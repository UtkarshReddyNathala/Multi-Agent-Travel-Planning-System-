"""Scripted stand-ins for the LLM and the MCP tools.

Used by the pytest suite (via tests/conftest.py) and by the offline mode of the
evaluation harness. Nothing here talks to Groq, Tavily, AviationStack or
OpenWeather. The MCP stubs return the content-block shape that
langchain-mcp-adapters really returns (verified against the installed
version), e.g. ``[{"type": "text", "text": "<json string>"}]``.
"""

import json

from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------
def default_supervisor(prompt: str) -> dict:
    return {
        "selected_agents": [
            "flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent",
        ],
        "trip_constraints": {
            "destination": "Goa", "origin": "", "duration": "5 days",
            "budget": "30000", "travel_style": "", "special_preferences": [],
        },
        "reasoning": "Full trip request.",
    }


class FakeLLM:
    """Stands in for ChatGroq. Routes on the system prompt each agent uses."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.guardrail = lambda prompt: {"allowed": True, "category": "travel", "reason": ""}
        self.supervisor = default_supervisor
        self.fail_on: set[str] = set()  # substrings of system prompts that must raise
        self.raw: dict[str, str] = {}  # system-substring -> raw reply override

    def prompts_for(self, system_fragment: str) -> list[str]:
        return [p for s, p in self.calls if system_fragment in s]

    def invoke(self, messages):
        if isinstance(messages, str):
            system, prompt = "", messages
        else:
            system, prompt = messages[0].content, messages[-1].content
        self.calls.append((system, prompt))

        for fragment in self.fail_on:
            if fragment in system:
                raise RuntimeError("simulated LLM outage (key=gsk_abcdefghijklmnop)")
        for fragment, text in self.raw.items():
            if fragment in system:
                return AIMessage(content=text)

        if "input guardrail" in system:
            return AIMessage(content=json.dumps(self.guardrail(prompt)))
        if "route work" in system:
            return AIMessage(content=json.dumps(self.supervisor(prompt)))
        if "flight planner" in system:
            return AIMessage(content="FLIGHT GUIDANCE")
        if "budget analyst" in system:
            return AIMessage(content="BUDGET ASSESSMENT")
        if "expert travel planner" in system:
            return AIMessage(content="DRAFT ITINERARY")
        if "travel booking assistant" in system:
            return AIMessage(content="FINAL PLAN")
        return AIMessage(content="Goa")  # extract_destination (plain prompt)


# ---------------------------------------------------------------------------
# MCP stubs (real adapter result shape)
# ---------------------------------------------------------------------------
def blocks(payload) -> list[dict]:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return [{"type": "text", "text": text, "id": "lc_test"}]


CURRENT = {
    "city": "Goa", "temperature_c": 29.5, "feels_like_c": 33.1,
    "humidity": 78, "condition": "scattered clouds", "wind_speed": 4.2,
}
FORECAST = {
    "city": "Goa",
    "forecast": [
        {"datetime": "2026-09-20 15:00:00", "temperature_c": 30.1, "condition": "light rain"},
        {"datetime": "2026-09-20 18:00:00", "temperature_c": 28.4, "condition": "light rain"},
    ],
}


class McpStubs:
    def __init__(self):
        self.calls = {"weather": 0, "forecast": 0, "aviation": 0, "tavily": 0}
        self.fail = set()  # subset of the keys above
        self.current_reply = blocks(CURRENT)
        self.forecast_reply = blocks(FORECAST)

    async def weather(self, city):
        self.calls["weather"] += 1
        if "weather" in self.fail:
            raise RuntimeError("Error executing tool get_current_weather: 404 city not found appid=SECRET123")
        return self.current_reply

    async def forecast(self, city):
        self.calls["forecast"] += 1
        if "forecast" in self.fail:
            raise TimeoutError("forecast timed out")
        return self.forecast_reply

    async def aviation(self, tool_name, tool_args=None):
        self.calls["aviation"] += 1
        if "aviation" in self.fail:
            raise RuntimeError("uvx was not found")
        return blocks(f"{tool_name}: DEL, GOI, BOM")

    async def tavily(self, query):
        self.calls["tavily"] += 1
        if "tavily" in self.fail:
            raise RuntimeError("https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-SECRETKEY failed")
        return blocks("Hotel Alpha - beachfront, Hotel Beta - budget")
