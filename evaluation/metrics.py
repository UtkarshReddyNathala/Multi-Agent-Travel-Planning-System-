"""Metric functions. Each is pure: (case, API-style result dict) -> MetricResult.

``result`` is exactly what ``backend.run_travel_agent`` / ``resume_travel_agent``
return (the same JSON the frontend receives), so the evaluation exercises the
real response shape rather than a private copy of it.
"""

import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from dashboard import STAGES
from evaluation.dataset import EvalCase
from observability import redact
from schemas import Dashboard, TripConstraints, WeatherResult

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"


@dataclass
class MetricResult:
    name: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _ok(name: str, detail: str = "") -> MetricResult:
    return MetricResult(name, PASS, detail)


def _fail(name: str, detail: str) -> MetricResult:
    return MetricResult(name, FAIL, detail)


def skipped(name: str, reason: str) -> MetricResult:
    return MetricResult(name, SKIPPED, reason)


# ---------------------------------------------------------------------------
# Guardrail behaviour
# ---------------------------------------------------------------------------
def check_guardrail(case: EvalCase, result: dict[str, Any]) -> MetricResult:
    allowed = result.get("guardrail_allowed", True)
    if allowed != case.expect_allowed:
        wanted = "allowed" if case.expect_allowed else "blocked"
        got = "allowed" if allowed else f"blocked ({result.get('guardrail_reason', '')[:80]})"
        return _fail("guardrail", f"expected {wanted}, got {got}")

    if not case.expect_allowed:
        # A blocked request must never reach the specialist agents or ask for approval.
        if result.get("requires_approval"):
            return _fail("guardrail", "blocked request still asked for approval")
        if any(result.get(k) for k in ("flight_results", "hotel_results", "weather_results", "itinerary")):
            return _fail("guardrail", "blocked request still returned specialist results")
    return _ok("guardrail", "allowed" if case.expect_allowed else "blocked")


# ---------------------------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------------------------
def check_routing(case: EvalCase, result: dict[str, Any]) -> MetricResult:
    if not (case.must_include or case.must_exclude):
        return skipped("routing", "no routing expectation for this case")
    selected = set(result.get("selected_agents", []))
    missing = [a for a in case.must_include if a not in selected]
    unwanted = [a for a in case.must_exclude if a in selected]
    if missing or unwanted:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unwanted:
            parts.append(f"unexpected {unwanted}")
        return _fail("routing", f"{', '.join(parts)}; selected={sorted(selected)}")
    return _ok("routing", f"selected={sorted(selected)}")


# ---------------------------------------------------------------------------
# Extracted trip constraints (also covers conversation memory on follow-ups)
# ---------------------------------------------------------------------------
def check_constraints(case: EvalCase, result: dict[str, Any]) -> MetricResult:
    if not case.expected_constraints:
        return skipped("constraints", "no constraint expectation for this case")
    constraints = result.get("trip_constraints", {}) or {}
    wrong = []
    for field_name, acceptable in case.expected_constraints.items():
        actual = str(constraints.get(field_name, "")).casefold()
        if not any(option.casefold() in actual for option in acceptable):
            wrong.append(f"{field_name}={actual!r} (wanted one of {list(acceptable)})")
    if wrong:
        return _fail("constraints", "; ".join(wrong))
    return _ok("constraints", "all expected fields present")


# ---------------------------------------------------------------------------
# Structured output validity + clean output
# ---------------------------------------------------------------------------
_LEAK = re.compile(r"Traceback \(most recent call last\)|Error executing tool|appid=|access_key=|tavilyApiKey=", re.I)


def check_structured(case: EvalCase, result: dict[str, Any]) -> MetricResult:
    problems = []

    try:
        dashboard = Dashboard.model_validate(result.get("dashboard") or {})
        if [s.key for s in dashboard.stages] != [key for key, _ in STAGES]:
            problems.append("dashboard stages do not match the pipeline")
    except ValidationError as exc:
        problems.append(f"dashboard invalid: {exc.errors()[0]['msg']}")

    if result.get("guardrail_allowed", True):
        try:
            TripConstraints.model_validate(result.get("trip_constraints") or {})
        except ValidationError as exc:
            problems.append(f"trip_constraints invalid: {exc.errors()[0]['msg']}")

        if "weather_agent" in result.get("selected_agents", []):
            try:
                WeatherResult.model_validate(result.get("weather_data") or {})
            except ValidationError as exc:
                problems.append(f"weather_data invalid: {exc.errors()[0]['msg']}")

    visible_text = " ".join(
        str(result.get(key, ""))
        for key in ("answer", "flight_results", "hotel_results", "weather_results", "budget_results")
    )
    if _LEAK.search(visible_text) or redact(visible_text) != visible_text:
        problems.append("raw error text or a secret appears in user-visible output")

    if problems:
        return _fail("structured_output", "; ".join(problems))
    return _ok("structured_output", "schemas valid, no leaks")


# ---------------------------------------------------------------------------
# Final answer quality - simple heuristics (NOT a substitute for reading answers)
# ---------------------------------------------------------------------------
MIN_ANSWER_CHARS = 200


def check_answer(case: EvalCase, result: dict[str, Any]) -> MetricResult:
    answer = (result.get("answer") or "").strip()
    if not answer:
        return _fail("answer_quality", "empty answer")
    if len(answer) < MIN_ANSWER_CHARS:
        return _fail("answer_quality", f"answer is very short ({len(answer)} chars)")
    destination = case.expected_constraints.get("destination")
    if destination and not any(option.casefold() in answer.casefold() for option in destination):
        return _fail("answer_quality", f"answer never mentions the destination {list(destination)}")
    return _ok("answer_quality", f"{len(answer)} chars")
