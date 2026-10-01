"""Tests for the evaluation harness itself (dataset integrity, metrics, runner, judge).

These do NOT measure model quality - they check that the harness reports correctly,
including that it *fails* when the system misbehaves.
"""

import json

import pytest

import backend
from dashboard import AGENT_ORDER
from evaluation import metrics as m
from evaluation.dataset import EVAL_CASES
from evaluation.judge import JudgeScores, build_judge_prompt, judge_answer
from evaluation.runner import _run_case, _score_case, run_evaluation, summarise

CASES = {c.id: c for c in EVAL_CASES}


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------
def test_dataset_ids_are_unique_and_nonempty():
    ids = [c.id for c in EVAL_CASES]
    assert len(ids) == len(set(ids)) and len(ids) >= 15
    assert all(c.turns and all(isinstance(t, str) for t in c.turns) for c in EVAL_CASES)


def test_dataset_only_references_real_agents():
    for case in EVAL_CASES:
        for agent in (*case.must_include, *case.must_exclude):
            assert agent in AGENT_ORDER, (case.id, agent)
        assert not set(case.must_include) & set(case.must_exclude), case.id


def test_dataset_covers_every_behaviour_the_spec_asks_for():
    categories = {c.category for c in EVAL_CASES}
    assert {"valid_full_trip", "valid_partial_request", "malformed", "prompt_injection",
            "unsafe", "off_topic", "conversation_memory", "false_positive_guard"} <= categories
    assert {c.decided_by for c in EVAL_CASES if not c.expect_allowed} == {"validation", "rules", "llm"}


def test_dataset_is_labelled_as_evaluation_data():
    import evaluation.dataset as dataset

    assert "not objective" in dataset.__doc__.lower() or "not real user traffic" in dataset.__doc__.lower()


# ---------------------------------------------------------------------------
# Metric functions (pure)
# ---------------------------------------------------------------------------
def test_guardrail_metric_pass_and_fail():
    blocked = CASES["injection_ignore"]
    assert m.check_guardrail(blocked, {"guardrail_allowed": False}).status == m.PASS
    assert m.check_guardrail(blocked, {"guardrail_allowed": True}).status == m.FAIL

    valid = CASES["trip_dubai"]
    assert m.check_guardrail(valid, {"guardrail_allowed": True}).status == m.PASS
    assert m.check_guardrail(valid, {"guardrail_allowed": False}).status == m.FAIL


def test_guardrail_metric_fails_if_a_blocked_request_leaks_results():
    result = {"guardrail_allowed": False, "flight_results": "some flights"}
    assert m.check_guardrail(CASES["unsafe_smuggling"], result).status == m.FAIL
    result = {"guardrail_allowed": False, "requires_approval": True}
    assert m.check_guardrail(CASES["unsafe_smuggling"], result).status == m.FAIL


def test_routing_metric():
    case = CASES["weather_only"]  # include weather; exclude flight/hotel/budget
    good = {"selected_agents": ["weather_agent", "itinerary_agent"]}
    assert m.check_routing(case, good).status == m.PASS
    missing = {"selected_agents": ["itinerary_agent"]}
    assert "missing" in m.check_routing(case, missing).detail
    extra = {"selected_agents": ["weather_agent", "flight_agent", "itinerary_agent"]}
    assert "unexpected" in m.check_routing(case, extra).detail
    assert m.check_routing(CASES["immigration_queue_wording"], good).status == m.SKIPPED


def test_constraints_metric_accepts_any_listed_spelling():
    case = CASES["followup_budget_change"]
    ok = {"trip_constraints": {"destination": "Goa", "duration": "5 days", "budget": "25,000 INR"}}
    assert m.check_constraints(case, ok).status == m.PASS
    stale = {"trip_constraints": {"destination": "Goa", "duration": "5 days", "budget": "30000"}}
    result = m.check_constraints(case, stale)
    assert result.status == m.FAIL and "budget" in result.detail
    forgot = {"trip_constraints": {"destination": "", "duration": "", "budget": "25000"}}
    assert m.check_constraints(case, forgot).status == m.FAIL


def test_answer_metric_heuristics():
    case = CASES["trip_dubai"]
    assert m.check_answer(case, {"answer": ""}).status == m.FAIL
    assert m.check_answer(case, {"answer": "Dubai " + "x" * 50}).status == m.FAIL  # too short
    assert m.check_answer(case, {"answer": "y" * 300}).status == m.FAIL  # never names Dubai
    assert m.check_answer(case, {"answer": "Day 1 in Dubai. " * 30}).status == m.PASS


def test_structured_metric_detects_leaks_and_bad_schemas(graph):
    good = backend.run_travel_agent("Plan a 5-day trip to Goa", "eval-struct")
    assert m.check_structured(CASES["trip_dubai"], good).status == m.PASS

    leaky = {**good, "flight_results": "Error executing tool x: appid=SECRET123"}
    assert m.check_structured(CASES["trip_dubai"], leaky).status == m.FAIL

    broken = {**good, "weather_data": {"status": "ok", "current": {"city": "Goa"}}}  # invalid current weather
    assert m.check_structured(CASES["trip_dubai"], broken).status == m.FAIL

    no_dashboard = {k: v for k, v in good.items() if k != "dashboard"}
    assert m.check_structured(CASES["trip_dubai"], no_dashboard).status == m.FAIL


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def test_offline_run_reports_honestly(monkeypatch):
    report = run_evaluation("offline")
    assert report["mode"] == "offline" and "scripted" in report["llm"]
    assert any("not objective ground truth" in n for n in report["notes"])
    assert len(report["cases"]) == len(EVAL_CASES)

    summary = report["summary"]
    assert summary["guardrail"]["failed"] == 0 and summary["guardrail"]["passed"] >= 15
    assert summary["structured_output"]["failed"] == 0
    # model-dependent metrics must be skipped, never faked, in offline mode
    for name in ("routing", "constraints", "answer_quality"):
        assert summary[name]["passed"] == 0 and summary[name]["pass_rate"] is None
    llm_cases = [r for r in report["cases"] if r["id"] in ("offtopic_code", "offtopic_recipe")]
    assert all(r["metrics"]["guardrail"]["status"] == m.SKIPPED for r in llm_cases)


def test_offline_run_does_not_leave_patches_behind():
    run_evaluation("offline", only=["trip_dubai"])
    assert backend._graph_is_override is False
    assert type(backend.llm).__name__ != "FakeLLM"


def test_evaluation_actually_fails_when_the_guardrail_is_broken(monkeypatch):
    import guardrails

    monkeypatch.setattr(guardrails, "rule_check", lambda text: None)  # rules layer disabled
    report = run_evaluation("offline", only=["injection_ignore", "unsafe_smuggling", "trip_dubai"])
    guardrail = report["summary"]["guardrail"]
    assert guardrail["failed"] == 2 and guardrail["passed"] == 1


def test_run_evaluation_validates_arguments():
    with pytest.raises(ValueError):
        run_evaluation("offline", use_judge=True)
    with pytest.raises(ValueError):
        run_evaluation("sideways")
    with pytest.raises(ValueError):
        run_evaluation("offline", only=["no-such-case"])


def test_live_scoring_path_with_a_scripted_supervisor(graph, fake_llm):
    """Exercises routing/constraints/answer scoring end to end (scripted, so the
    numbers say nothing about a real model - only that the scoring path works)."""
    case = CASES["followup_budget_change"]
    calls = {"n": 0}

    def supervisor(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"selected_agents": list(AGENT_ORDER),
                    "trip_constraints": {"destination": "Goa", "duration": "5 days", "budget": "30000"},
                    "reasoning": "full"}
        return {"selected_agents": ["budget_agent", "itinerary_agent"],
                "trip_constraints": {"budget": "25000"}, "reasoning": "budget change",
                "resolved_query": "5-day Goa trip under 25000"}

    fake_llm.supervisor = supervisor
    fake_llm.raw["expert travel planner"] = "Day 1 in Goa. " * 30
    fake_llm.raw["travel booking assistant"] = "Final Goa plan. " * 30

    result, seconds = _run_case(backend, case, "unit")
    scores = {r.name: r for r in _score_case(case, result, "live")}
    assert seconds >= 0
    assert scores["guardrail"].status == m.PASS
    assert scores["routing"].status == m.PASS
    assert scores["constraints"].status == m.PASS, scores["constraints"].detail
    assert scores["answer_quality"].status == m.PASS, scores["answer_quality"].detail
    assert scores["structured_output"].status == m.PASS


def test_live_scoring_flags_a_supervisor_that_forgets_the_conversation(graph, fake_llm):
    case = CASES["followup_budget_change"]
    fake_llm.supervisor = lambda prompt: {  # ignores context, keeps the OLD budget
        "selected_agents": ["itinerary_agent"],
        "trip_constraints": {"destination": "Goa", "duration": "5 days", "budget": "30000"},
        "reasoning": "x",
    }
    result, _ = _run_case(backend, case, "unit2")
    scores = {r.name: r for r in _score_case(case, result, "live")}
    assert scores["constraints"].status == m.FAIL
    assert scores["routing"].status == m.FAIL  # budget_agent was required


def test_summary_excludes_skipped_from_rates():
    rows = [
        {"latency_s": 1.0, "judge": None, "metrics": {"guardrail": {"status": "pass"}}},
        {"latency_s": 3.0, "judge": None, "metrics": {"guardrail": {"status": "fail"}}},
        {"latency_s": None, "judge": None, "metrics": {"guardrail": {"status": "skipped"}}},
    ]
    stats = summarise(rows)
    assert stats["guardrail"] == {"passed": 1, "failed": 1, "skipped": 1, "pass_rate": 0.5}
    assert stats["latency_s"]["max"] == 3.0
    assert stats["routing"]["pass_rate"] is None


def test_cli_writes_a_json_report(tmp_path, monkeypatch):
    import evaluation.run as cli

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    assert cli.main(["--mode", "offline", "--case", "trip_dubai"]) == 0
    (report_file,) = list((tmp_path / "results").glob("offline-*.json"))
    assert json.loads(report_file.read_text())["cases"][0]["id"] == "trip_dubai"


def test_cli_exit_code_is_nonzero_when_a_metric_fails(monkeypatch, tmp_path):
    import evaluation.run as cli
    import guardrails

    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(guardrails, "rule_check", lambda text: None)
    assert cli.main(["--mode", "offline", "--case", "injection_ignore", "--no-save"]) == 1


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------
def test_judge_prompt_contains_sources_and_answer():
    prompt = build_judge_prompt("Trip to Goa", {"answer": "FINAL", "weather_results": "29C cloudy"})
    assert "Trip to Goa" in prompt and "29C cloudy" in prompt and "FINAL" in prompt
    assert "did not run" in prompt  # missing sources are stated, not invented


def test_judge_parses_scores_and_never_raises():
    good = lambda system, prompt: '{"relevance": 5, "groundedness": 4, "completeness": 3, "comment": "ok"}'
    scores = judge_answer("req", {"answer": "a"}, good)
    assert isinstance(scores, JudgeScores) and scores.groundedness == 4

    assert judge_answer("req", {}, lambda s, p: "not json at all") is None
    assert judge_answer("req", {}, lambda s, p: '{"relevance": 9, "groundedness": 1, "completeness": 1}') is None

    def boom(system, prompt):
        raise RuntimeError("judge down")

    assert judge_answer("req", {}, boom) is None
