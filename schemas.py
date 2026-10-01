"""Pydantic models for the data in this project that has a fixed shape:

* ``TripConstraints`` / ``SupervisorPlan`` - the JSON the supervisor LLM returns.
* ``WeatherResult`` - the fixed-shape dicts returned by ``custom_weather_mcp_server.py``.
* ``TraceEvent`` / ``Dashboard`` - execution metadata that the backend measures.

The flight, hotel, budget and itinerary agents return free text (from the LLM or
Tavily) with no fixed format, so they stay plain ``str``. Their run status is
still recorded with ``TraceEvent``.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# LLM JSON parsing
# ---------------------------------------------------------------------------
def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model reply."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model returned JSON that is not an object.")
    return parsed


# ---------------------------------------------------------------------------
# Supervisor output
# ---------------------------------------------------------------------------
_TEXT_FIELDS = ("destination", "origin", "duration", "budget", "travel_style")


class TripConstraints(BaseModel):
    """Trip details extracted from the request (same keys as ``_empty_constraints()``), with type checks."""

    model_config = ConfigDict(extra="ignore")

    destination: str = ""
    origin: str = ""
    duration: str = ""
    budget: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(default_factory=list)

    @field_validator(*_TEXT_FIELDS, mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        # LLMs sometimes return numbers (budget: 30000) or null for "unknown".
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("special_preferences", mode="before")
    @classmethod
    def _preferences(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


def merge_constraints(previous: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Follow-up turn: keep earlier details unless the new message changed them.

    A field the new turn left empty falls back to the previous value, so the LLM
    forgetting to repeat the destination cannot erase it.
    """
    prev = TripConstraints.model_validate(previous or {}).model_dump()
    curr = TripConstraints.model_validate(new or {}).model_dump()
    merged = {}
    for key in prev:
        merged[key] = curr[key] if curr[key] else prev[key]
    return merged


class SupervisorPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected_agents: list[str] = Field(default_factory=list)
    trip_constraints: TripConstraints = Field(default_factory=TripConstraints)
    reasoning: str = ""
    resolved_query: str = ""

    @field_validator("trip_constraints", mode="before")
    @classmethod
    def _constraints(cls, value: Any) -> Any:
        return value if isinstance(value, dict) else {}

    @field_validator("reasoning", "resolved_query", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("selected_agents", mode="before")
    @classmethod
    def _agents(cls, value: Any) -> list[str]:
        return [str(v) for v in value] if isinstance(value, (list, tuple)) else []


# ---------------------------------------------------------------------------
# Weather (fields taken from custom_weather_mcp_server.py)
# ---------------------------------------------------------------------------
class CurrentWeather(BaseModel):
    city: str
    temperature_c: float
    feels_like_c: float
    humidity: float
    condition: str
    wind_speed: float


class ForecastEntry(BaseModel):
    datetime: str
    temperature_c: float
    condition: str


class WeatherResult(BaseModel):
    """What the Weather Agent hands to the rest of the workflow (and to Redis)."""

    status: Literal["ok", "partial", "unavailable"]
    city: str
    current: CurrentWeather | None = None
    forecast: list[ForecastEntry] = Field(default_factory=list)
    error: str | None = None

    def to_prompt_text(self) -> str:
        """Readable text for the downstream LLM prompts (same two sections as before)."""
        if self.status == "unavailable":
            return (
                f"Live weather information for {self.city} is temporarily unavailable. "
                "Give general seasonal guidance and advise the traveler to verify "
                "the forecast before departure."
            )

        lines = ["Current Weather:"]
        if self.current:
            c = self.current
            lines.append(
                f"{c.city}: {c.temperature_c}°C (feels like {c.feels_like_c}°C), "
                f"{c.condition}, humidity {c.humidity}%, wind {c.wind_speed} m/s"
            )
        else:
            lines.append("Current conditions are unavailable.")

        lines += ["", "Forecast (3-hour steps):"]
        if self.forecast:
            lines += [
                f"- {f.datetime}: {f.temperature_c}°C, {f.condition}" for f in self.forecast
            ]
        else:
            lines.append("Forecast data is unavailable.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution trace / dashboard
# ---------------------------------------------------------------------------
class TraceEvent(BaseModel):
    """One measured node execution. ``status``: success | degraded | failed | blocked."""

    run_id: str = ""
    node: str
    status: str
    duration_ms: float | None = None
    detail: str = ""
    error: str | None = None
    cache: str | None = None  # hit | miss | unavailable | disabled (weather only)


class DashboardStage(BaseModel):
    key: str
    label: str
    status: str
    duration_ms: float | None = None
    detail: str = ""
    error: str | None = None
    cache: str | None = None


class Dashboard(BaseModel):
    run_id: str = ""
    workflow_status: str
    has_warnings: bool = False
    measured_ms: float = 0.0
    llm_calls: int = 0
    stages: list[DashboardStage]
