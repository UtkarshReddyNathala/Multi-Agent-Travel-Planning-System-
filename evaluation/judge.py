"""Optional LLM-as-a-judge scoring (live mode only, ``--judge``).

This is an EVALUATION METHOD, not objective truth: a language model reads the
source data the agents were given plus the final answer and scores it. Known
weaknesses - it is noisy, it can be lenient toward fluent text, and here the
judge is the same model family that wrote the answer (self-preference bias).
Treat the numbers as a rough signal to compare runs, never as a grade.
"""

from typing import Any, Callable

from pydantic import BaseModel, Field

from schemas import parse_json_object

JUDGE_SYSTEM = "You are a strict reviewer of travel-planning answers. Return strict JSON only."


class JudgeScores(BaseModel):
    relevance: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    comment: str = ""


def build_judge_prompt(request: str, result: dict[str, Any]) -> str:
    def block(title: str, key: str) -> str:
        text = str(result.get(key) or "").strip() or "(agent did not run or returned nothing)"
        return f"### {title}\n{text[:1500]}"

    return f"""
Score the FINAL ANSWER of a travel-planning assistant.

User request:
{request}

Source data the specialist agents produced (this is all the real data that existed):
{block("Flight data", "flight_results")}
{block("Hotel data", "hotel_results")}
{block("Weather data", "weather_results")}
{block("Budget assessment", "budget_results")}

FINAL ANSWER:
{str(result.get("answer") or "")[:4000]}

Score each from 1 (poor) to 5 (excellent):
- relevance: does the answer address what the user asked?
- groundedness: are concrete facts (prices, temperatures, flights, hotels) supported by the
  source data above, and is missing data acknowledged rather than invented?
- completeness: does it cover the parts of the request that the sources allow?

Return strict JSON only:
{{"relevance": 1, "groundedness": 1, "completeness": 1, "comment": "one short sentence"}}
""".strip()


def judge_answer(
    request: str,
    result: dict[str, Any],
    llm_call: Callable[[str, str], str],
) -> JudgeScores | None:
    """Return scores, or None when the judge call/parse fails (never raises)."""
    try:
        raw = llm_call(JUDGE_SYSTEM, build_judge_prompt(request, result))
        return JudgeScores.model_validate(parse_json_object(raw))
    except Exception:  # noqa: BLE001 - judging is best-effort and must never break a run
        return None
