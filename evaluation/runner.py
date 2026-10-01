"""Runs the evaluation dataset through the real LangGraph workflow.

Two modes
---------
offline (default; no API keys, no network)
    A *scripted* LLM and stubbed MCP tools are plugged into the real graph. This
    checks the harness and the deterministic parts of the system - input
    validation, the rule-based guardrail layer, schema validity, response shape,
    latency of the plumbing. It does NOT measure model quality: cases that need
    a real model (LLM guardrail, routing, constraint extraction, answer quality)
    are reported as "skipped".

live (needs GROQ_API_KEY)
    The real Groq model runs every LLM step. MCP tools stay stubbed by default
    so results depend on the model, not on flaky third-party APIs
    (``--real-mcp`` uses the real tools instead). Optional ``--judge`` adds LLM
    scoring of the final answer (see judge.py for its limits).

The graph always uses an in-memory checkpointer, so PostgreSQL is not needed.
"""

import os
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from unittest import mock

from evaluation import metrics as m
from evaluation.dataset import EVAL_CASES, EvalCase

DATASET_NOTE = (
    "Hand-written evaluation examples written for this project. Not real user data "
    "and not objective ground truth; expectations are the developer's intent."
)
JUDGE_NOTE = (
    "LLM-as-a-judge scores are an evaluation method, not objective truth: they are "
    "noisy and the judge is the same model family that wrote the answers."
)
LIVE_MODEL = "llama-3.3-70b-versatile"  # the model configured in backend.py


@contextmanager
def _environment(mode: str, real_mcp: bool) -> Iterator[Any]:
    """Wire the real graph to either scripted (offline) or real (live) components."""
    if mode == "offline":
        for name in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPENWEATHER_API_KEY"):
            os.environ.setdefault(name, "offline-evaluation-placeholder")
    elif not os.getenv("GROQ_API_KEY"):
        # backend.py loads .env on import, so the key can also come from .env.
        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("GROQ_API_KEY"):
            raise SystemExit("Live mode needs GROQ_API_KEY (environment or .env).")

    import backend
    import mcp_client
    from langgraph.checkpoint.memory import InMemorySaver

    from evaluation.stubs import FakeLLM, McpStubs

    patches = []
    if mode == "offline":
        fake = FakeLLM()
        patches += [
            mock.patch.object(backend, "llm", fake),
            mock.patch.object(mcp_client, "llm", fake),
        ]
    if mode == "offline" or not real_mcp:
        stubs = McpStubs()
        patches += [
            mock.patch.object(mcp_client, "weather_mcp_search", stubs.weather),
            mock.patch.object(mcp_client, "forecast_mcp_search", stubs.forecast),
            mock.patch.object(backend, "aviation_mcp_call", stubs.aviation),
            mock.patch.object(backend, "tavily_mcp_search", stubs.tavily),
        ]
    for patch in patches:
        patch.start()
    backend.set_travel_graph(backend.build_travel_graph(InMemorySaver()))
    try:
        yield backend
    finally:
        backend.set_travel_graph(None)
        for patch in reversed(patches):
            patch.stop()


def _run_case(backend, case: EvalCase, run_tag: str) -> tuple[dict[str, Any], float]:
    """Play every turn on one thread, approving each draft. Returns (final result, seconds)."""
    thread_id = f"eval-{run_tag}-{case.id}"
    started = time.perf_counter()
    result: dict[str, Any] = {}
    for message in case.turns:
        result = backend.run_travel_agent(message, thread_id)
        if result.get("requires_approval"):
            result = backend.resume_travel_agent(thread_id, approved=True)
    return result, time.perf_counter() - started


def _score_case(case: EvalCase, result: dict[str, Any], mode: str) -> list[m.MetricResult]:
    offline = mode == "offline"
    llm_reason = "needs a real model (run with --mode live)"

    results = []
    if offline and case.decided_by == "llm" and not case.expect_allowed:
        # The fake LLM always says "allowed", so this can't be checked offline.
        results.append(m.skipped("guardrail", llm_reason))
    else:
        results.append(m.check_guardrail(case, result))

    if not case.expect_allowed:
        results += [m.skipped(name, "request is expected to be blocked")
                    for name in ("routing", "constraints", "answer_quality")]
    elif offline:
        results += [m.skipped(name, llm_reason) for name in ("routing", "constraints", "answer_quality")]
    else:
        results += [m.check_routing(case, result), m.check_constraints(case, result), m.check_answer(case, result)]

    results.append(m.check_structured(case, result))
    return results


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["guardrail", "routing", "constraints", "structured_output", "answer_quality"]
    summary: dict[str, Any] = {}
    for name in metric_names:
        statuses = [r["metrics"][name]["status"] for r in rows if name in r["metrics"]]
        passed, failed = statuses.count(m.PASS), statuses.count(m.FAIL)
        summary[name] = {
            "passed": passed,
            "failed": failed,
            "skipped": statuses.count(m.SKIPPED),
            "pass_rate": round(passed / (passed + failed), 3) if passed + failed else None,
        }

    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    if latencies:
        summary["latency_s"] = {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "max": round(max(latencies), 3),
        }

    judged = [r["judge"] for r in rows if r.get("judge")]
    if judged:
        summary["judge"] = {
            key: round(statistics.mean(j[key] for j in judged), 2)
            for key in ("relevance", "groundedness", "completeness")
        } | {"answers_judged": len(judged)}
    return summary


def run_evaluation(
    mode: str = "offline",
    use_judge: bool = False,
    real_mcp: bool = False,
    only: list[str] | None = None,
) -> dict[str, Any]:
    if mode not in ("offline", "live"):
        raise ValueError("mode must be 'offline' or 'live'")
    if use_judge and mode != "live":
        raise ValueError("--judge needs --mode live (a scripted LLM cannot judge anything)")

    cases = [c for c in EVAL_CASES if not only or c.id in only]
    if not cases:
        raise ValueError(f"No evaluation case matches {only}")

    run_tag = datetime.now(timezone.utc).strftime("%H%M%S%f")
    rows: list[dict[str, Any]] = []

    with _environment(mode, real_mcp) as backend:
        for case in cases:
            row: dict[str, Any] = {
                "id": case.id,
                "category": case.category,
                "turns": len(case.turns),
                "latency_s": None,
                "metrics": {},
                "error": None,
                "judge": None,
            }
            try:
                result, seconds = _run_case(backend, case, run_tag)
                row["latency_s"] = round(seconds, 3)
                row["workflow_status"] = result.get("workflow_status")
                row["selected_agents"] = result.get("selected_agents", [])
                for metric in _score_case(case, result, mode):
                    row["metrics"][metric.name] = metric.as_dict()

                if use_judge and case.expect_allowed:
                    scores = _judge(backend, case, result)
                    row["judge"] = scores
            except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
                from observability import safe_error

                row["error"] = safe_error(exc)
                for name in ("guardrail", "structured_output"):
                    row["metrics"][name] = m.MetricResult(name, m.FAIL, f"case crashed: {row['error']}").as_dict()
            rows.append(row)

    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "llm": "scripted (no model quality is measured)" if mode == "offline" else LIVE_MODEL,
        "mcp_tools": "real" if (mode == "live" and real_mcp) else "stubbed",
        "checkpointer": "in-memory (PostgreSQL not used)",
        "judge_enabled": use_judge,
        "notes": [DATASET_NOTE] + ([JUDGE_NOTE] if use_judge else []),
        "cases": rows,
        "summary": summarise(rows),
    }


def _judge(backend, case: EvalCase, result: dict[str, Any]) -> dict[str, Any] | None:
    from evaluation.judge import judge_answer

    scores = judge_answer(" ".join(case.turns), result, backend._llm_text)
    return None if scores is None else scores.model_dump()
