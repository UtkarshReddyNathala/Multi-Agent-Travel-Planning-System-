import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import threading
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from config import llm_timeout
from dashboard import AGENT_ORDER, build_dashboard
from errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    NoPendingApprovalError,
    ThreadNotFoundError,
)
from guardrails import run_guardrail
from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    mcp_result_text,
)
from observability import logger, safe_error, traced_node
from schemas import (
    SupervisorPlan,
    WeatherResult,
    merge_constraints,
    parse_json_object,
)
from weather_service import get_weather


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM (Groq, Llama 3.3 70B)
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    timeout=llm_timeout(),  # ChatGroq has no request timeout by default
)

# =========================
# State shared by every step of the workflow
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Specialist agent results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # Budget result and human-approval (HITL) state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int

    # Conversation memory: the previous turn of this thread (read from the
    # checkpoint when a run starts) and the full request the supervisor builds
    # for this turn, e.g. "Change the budget to 25000" + the earlier trip details.
    previous_query: str
    previous_constraints: dict[str, Any]
    resolved_query: str

    # Weather data, errors and execution tracking (for the dashboard)
    weather_data: dict[str, Any]
    workflow_error: str
    run_id: str
    execution_trace: Annotated[list[dict[str, Any]], operator.add]


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = set(AGENT_ORDER)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    return parse_json_object(text)


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def _effective_query(state: TravelState) -> str:
    """The request the specialists work on.

    On a follow-up turn this is the supervisor's standalone version of the
    request (earlier trip details + the change); otherwise it is the user's message.
    """
    return state.get("resolved_query") or state["user_query"]


def _same_place(a: str, b: str) -> bool:
    a, b = (a or "").strip().casefold(), (b or "").strip().casefold()
    return not a or not b or a in b or b in a


def _trip_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """True when the destination or origin differs from the earlier turn."""
    return not (
        _same_place(previous.get("destination", ""), current.get("destination", ""))
        and _same_place(previous.get("origin", ""), current.get("origin", ""))
    )


# =========================
# Input Guardrail (Input Validation -> Rules -> LLM, see guardrails.py)
# =========================
@traced_node("guardrail", catch=False)
def guardrail_agent(state: TravelState):
    outcome = run_guardrail(
        state["user_query"],
        previous_context=(state.get("previous_query") or "")[:600],
        llm_call=_llm_text,
    )
    decision = outcome.decision
    llm_calls = state.get("llm_calls", 0) + (1 if outcome.llm_used and decision.layer == "llm" else 0)

    validation_event = {
        "node": "input_validation",
        "status": "success" if outcome.validation_passed else "blocked",
        "duration_ms": None,
        "detail": "" if outcome.validation_passed else decision.reason,
    }

    if not decision.allowed:
        reason = decision.reason or "This request was blocked by the travel input guardrail."
        events = [validation_event]
        if outcome.validation_passed:
            events.append(
                {
                    "status": "blocked",
                    "detail": f"{decision.category} ({decision.layer}): {reason}",
                }
            )
        # trip_constraints and resolved_query are left as they are, so an
        # off-topic message in the middle of a conversation doesn't erase the trip.
        return {
            "user_query": outcome.cleaned_text,
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
            "_trace": events,
        }

    degraded = decision.category == "guardrail_unavailable"
    return {
        "user_query": outcome.cleaned_text,
        "guardrail_allowed": True,
        "guardrail_reason": decision.reason,
        "llm_calls": llm_calls,
        "_trace": [
            validation_event,
            {
                "status": "degraded" if degraded else "success",
                "detail": (
                    "LLM guardrail unavailable - only the deterministic checks were applied"
                    if degraded
                    else f"allowed ({decision.layer})"
                ),
            },
        ],
    }


# =========================
# Supervisor Agent
# =========================
@traced_node("supervisor", catch=False)
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)
    previous_query = (state.get("previous_query") or "").strip()
    previous_constraints = state.get("previous_constraints") or {}
    has_context = bool(previous_query)

    context_block = ""
    resolved_field = ""
    if has_context:
        context_block = f"""
Conversation context - an earlier turn of this same conversation:
Previous trip request: {previous_query}
Previous trip constraints: {previous_constraints}

If the new user message is a follow-up that changes or refines that trip (for
example a new budget, duration, or preference), keep every earlier detail the
user did not change and set "resolved_query" to ONE complete standalone trip
request that includes the change. Select every agent whose output is affected
by the change. If the new message is about a different trip, ignore the earlier
context. Never invent details that appear in neither message.
"""
        resolved_field = ',\n  "resolved_query": ""'

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included
{context_block}
Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""{resolved_field}
}}

User request:
{query}
"""

    fallback_error = None
    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        plan = SupervisorPlan.model_validate(parse_json_object(supervisor_raw))
        selected_agents = [name for name in AGENT_ORDER if name in plan.selected_agents]

        # The itinerary agent combines the results of whichever agents were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        new_constraints = plan.trip_constraints.model_dump()
        constraints = (
            merge_constraints(previous_constraints, new_constraints)
            if has_context
            else new_constraints
        )
        reasoning = plan.reasoning
        resolved_query = (
            (plan.resolved_query or f"{previous_query}\nFollow-up from the user: {query}")
            if has_context
            else query
        )
        llm_calls += 1
    except Exception as exc:
        fallback_error = safe_error(exc)
        logger.warning("Supervisor fallback used: %s", fallback_error)
        # Safe fallback: run every specialist agent.
        selected_agents = AGENT_ORDER.copy()
        constraints = (
            merge_constraints(previous_constraints, {}) if has_context else _empty_constraints()
        )
        reasoning = (
            "Supervisor parsing failed, so the full travel workflow "
            "was selected as a safe fallback."
        )
        resolved_query = (
            f"{previous_query}\nFollow-up from the user: {query}" if has_context else query
        )

    update: dict[str, Any] = {
        "guardrail_allowed": True,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "resolved_query": resolved_query,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }

    detail = "Selected: " + ", ".join(selected_agents)
    if has_context:
        detail += " | follow-up: earlier trip details reused"

    # A new destination or origin makes every earlier specialist result out of date.
    if has_context and _trip_changed(previous_constraints, constraints):
        update["selected_agents"] = AGENT_ORDER.copy()
        update.update(
            flight_results="",
            hotel_results="",
            weather_results="",
            weather_data={},
            budget_results="",
            itinerary="",
        )
        detail += " | destination/origin changed: earlier results dropped, all agents re-run"

    update["_trace"] = {
        "status": "degraded" if fallback_error else "success",
        "detail": detail + (" | fallback: full workflow" if fallback_error else ""),
        "error": fallback_error,
    }
    return update


def route_from_guardrail(state: TravelState) -> str:
    return "supervisor" if state.get("guardrail_allowed", True) else "guardrail_blocked"


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent (AviationStack MCP + LLM)
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

FLIGHT_UNAVAILABLE = (
    "Live flight and airport data is temporarily unavailable. Provide only "
    "general, clearly labelled non-live route and airline guidance, and advise "
    "the traveler to verify options and fares directly with airlines."
)


@traced_node("flight_agent")
def flight_agent(state: TravelState):
    logger.info("Flight agent started")
    query = _effective_query(state)
    error = None

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=mcp_result_text(airports)[:3000],
            airline_data=mcp_result_text(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        # Log a redacted error and show a friendly message, never the raw exception.
        error = safe_error(exc)
        logger.warning("Flight agent degraded: %s", error)
        flight_data = FLIGHT_UNAVAILABLE

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        # Only count successful LLM responses (none if this agent failed)
        "llm_calls": state.get("llm_calls", 0) + (0 if error else 1),
        "_trace": {
            "status": "degraded" if error else "success",
            "detail": "live flight data unavailable" if error else "AviationStack data + LLM guidance",
            "error": error,
        },
    }


# =========================
# Hotel Agent (Tavily MCP search)
# =========================
HOTEL_UNAVAILABLE = (
    "Live hotel search is temporarily unavailable. "
    "Provide general accommodation and neighborhood "
    "guidance based on the destination and clearly "
    "label it as non-live advice."
)


@traced_node("hotel_agent")
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{_effective_query(state)}"
    )
    error = None

    try:
        hotel_results = mcp_result_text(
            asyncio.run(
                tavily_mcp_search(query)
            )
        )
        if not hotel_results:
            raise ValueError("Tavily returned no results.")

    except Exception as exc:
        error = safe_error(exc)
        logger.warning("Hotel agent degraded: %s", error)
        hotel_results = HOTEL_UNAVAILABLE

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        # The hotel agent only calls Tavily (no LLM), so the LLM count doesn't change
        "llm_calls": state.get("llm_calls", 0),
        "_trace": {
            "status": "degraded" if error else "success",
            "detail": "live hotel search unavailable" if error else "Tavily search results",
            "error": error,
        },
    }


# =========================
# Weather Agent (weather MCP server, cached in Redis via weather_service.py)
# =========================
@traced_node("weather_agent")
def weather_agent(state: TravelState):
    city = ""
    cache_status = None
    llm_used = False

    try:
        # Use the destination found by the supervisor; only ask the LLM
        # to extract it if the supervisor didn't find one.
        city = ((state.get("trip_constraints") or {}).get("destination") or "").strip()
        if not city:
            city = extract_destination(_effective_query(state))
            llm_used = True

        weather, cache_status = get_weather(city)
    except Exception as exc:
        logger.warning("Weather agent degraded: %s", safe_error(exc))
        weather = WeatherResult(
            status="unavailable",
            city=city or "the destination",
            error=safe_error(exc),
        )

    return {
        "weather_results": weather.to_prompt_text(),
        "weather_data": weather.model_dump(),
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + (1 if llm_used else 0),
        "_trace": {
            "status": "success" if weather.status == "ok" else "degraded",
            "detail": f"{weather.city}: {weather.status}",
            "error": weather.error,
            "cache": cache_status,
        },
    }


# =========================
# Budget Agent (LLM)
# =========================
@traced_node("budget_agent")
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{_effective_query(state)}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    error = None
    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are a practical travel budget analyst."),
                HumanMessage(content=prompt),
            ]
        )
        budget_results = response.content
    except Exception as exc:
        error = safe_error(exc)
        logger.warning("Budget agent failed: %s", error)
        budget_results = (
            "Budget analysis is unavailable right now. Do not invent cost "
            "figures; tell the traveler to verify prices before booking."
        )

    return {
        "budget_results": budget_results,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + (0 if error else 1),
        "_trace": {
            "status": "failed" if error else "success",
            "detail": "budget analysis unavailable" if error else "LLM budget assessment",
            "error": error,
        },
    }


# =========================
# Itinerary Agent (builds the draft plan from the selected agents' results)
# =========================
ITINERARY_FAILED_MESSAGE = (
    "The draft itinerary could not be created because the AI service is "
    "temporarily unavailable. Please try again in a moment."
)


@traced_node("itinerary_agent")
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{_effective_query(state)}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel planner."),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:
        # With no draft there is nothing to approve, so end the run cleanly
        # (see route_after_itinerary) instead of crashing or asking for approval.
        error = safe_error(exc)
        logger.error("Itinerary agent failed: %s", error)
        return {
            "itinerary": "",
            "workflow_error": ITINERARY_FAILED_MESSAGE,
            "final_response": ITINERARY_FAILED_MESSAGE,
            "messages": [AIMessage(content=ITINERARY_FAILED_MESSAGE)],
            "llm_calls": state.get("llm_calls", 0),
            "_trace": {
                "status": "failed",
                "detail": "draft itinerary could not be generated",
                "error": error,
            },
        }

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def route_after_itinerary(state: TravelState) -> str:
    return END if state.get("workflow_error") else "human_approval"


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Don't wrap interrupt() in try/except: LangGraph uses it to pause the run.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent (writes the final plan, applying the human's feedback)
# =========================
@traced_node("final_agent")
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    latest_message = ""
    if state.get("resolved_query") and state["resolved_query"] != state["user_query"]:
        latest_message = f"\nLatest user message: {state['user_query']}"

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{_effective_query(state)}{latest_message}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a professional AI travel booking assistant."
                ),
                HumanMessage(content=final_prompt),
            ]
        )
    except Exception as exc:
        # If this fails, return the approved draft instead of losing the whole run.
        error = safe_error(exc)
        logger.error("Final agent failed, returning the draft: %s", error)
        note = (
            "_The final polishing step was unavailable, so this is the draft itinerary."
            + (
                " Your revision feedback could not be applied._"
                if not state.get("approved", False)
                else "_"
            )
        )
        fallback = f"{note}\n\n{state.get('itinerary', '')}"
        return {
            "final_response": fallback,
            "messages": [AIMessage(content=fallback)],
            "_trace": {
                "status": "degraded",
                "detail": "final polishing unavailable - draft returned",
                "error": error,
            },
        }

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
#
#   START -> guardrail -> supervisor -> specialist agents -> itinerary
#                 |                                             |
#                 v                                   human_approval -> final -> END
#         guardrail_blocked -> END
# =========================
def build_graph() -> StateGraph:
    builder = StateGraph(TravelState)

    builder.add_node("guardrail", guardrail_agent)
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("guardrail_blocked", guardrail_blocked_agent)
    builder.add_node("flight_agent", flight_agent)
    builder.add_node("hotel_agent", hotel_agent)
    builder.add_node("weather_agent", weather_agent)
    builder.add_node("budget_agent", budget_agent)
    builder.add_node("itinerary_agent", itinerary_agent)
    builder.add_node("human_approval", human_approval_agent)
    builder.add_node("final_agent", final_agent)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_from_guardrail,
        {"supervisor": "supervisor", "guardrail_blocked": "guardrail_blocked"},
    )
    builder.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

    for agent in ("flight_agent", "hotel_agent", "weather_agent", "budget_agent"):
        builder.add_conditional_edges(agent, route_after_agent(agent), ROUTE_MAP)

    builder.add_conditional_edges(
        "itinerary_agent",
        route_after_itinerary,
        {"human_approval": "human_approval", END: END},
    )
    builder.add_edge("human_approval", "final_agent")
    builder.add_edge("final_agent", END)
    builder.add_edge("guardrail_blocked", END)
    return builder


graph = build_graph()


def build_travel_graph(checkpointer):
    """Compile the workflow with any LangGraph checkpointer (Postgres, in-memory...)."""
    return build_graph().compile(checkpointer=checkpointer)


# =========================
# PostgreSQL Checkpointer (saves each conversation so it can pause and resume)
#
# The connection opens on first use, not when the module is imported, so
# importing it doesn't need a database and a database problem becomes a clear
# 503 error instead of a crash. A dropped connection is reopened on the next request.
# =========================
_travel_graph = None
_conn = None
_graph_is_override = False
_graph_lock = threading.Lock()


def get_travel_graph():
    global _travel_graph, _conn

    with _graph_lock:
        if _travel_graph is not None:
            return _travel_graph

        try:
            connection = psycopg.connect(
                get_database_url(),
                autocommit=True,
                row_factory=dict_row,
            )
            checkpointer = PostgresSaver(connection)
            checkpointer.setup()
        except ValueError as exc:  # DATABASE_URL missing
            raise DependencyUnavailableError(
                "The database is not configured (DATABASE_URL is missing)."
            ) from exc
        except psycopg.Error as exc:
            logger.error("Could not connect to PostgreSQL: %s", safe_error(exc))
            raise DependencyUnavailableError(
                "The database is unavailable. Please try again shortly."
            ) from exc

        _conn = connection
        _travel_graph = build_travel_graph(checkpointer)
        return _travel_graph


def set_travel_graph(compiled_graph) -> None:
    """Use a pre-built graph (tests, offline evaluation) instead of PostgreSQL."""
    global _travel_graph, _conn, _graph_is_override

    with _graph_lock:
        _travel_graph = compiled_graph
        _conn = None
        _graph_is_override = compiled_graph is not None


def _reset_graph() -> None:
    """Forget a broken PostgreSQL connection so the next request reconnects."""
    global _travel_graph, _conn

    with _graph_lock:
        if _graph_is_override:
            return
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _travel_graph = None
        _conn = None


_DB_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError)


def _read_state(config: dict[str, Any]):
    """Read the thread's checkpoint, reconnecting once if the database connection dropped."""
    for attempt in (1, 2):
        try:
            return get_travel_graph().get_state(config)
        except _DB_ERRORS as exc:
            logger.warning("Checkpoint read failed (attempt %s): %s", attempt, safe_error(exc))
            _reset_graph()
            if attempt == 2:
                raise DependencyUnavailableError(
                    "The database connection was lost. Please try again."
                ) from exc


def _invoke(payload, config: dict[str, Any]):
    """Run the graph. Database failures become a DependencyUnavailableError (503)."""
    try:
        return get_travel_graph().invoke(payload, config=config)
    except _DB_ERRORS as exc:
        logger.error("Database error during workflow: %s", safe_error(exc))
        _reset_graph()
        raise DependencyUnavailableError(
            "The database connection was lost. Please try again."
        ) from exc


def _run_config(thread_id: str, run_name: str) -> dict[str, Any]:
    """LangGraph run settings. run_name, tags and metadata are only used when LangSmith is on."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
        "tags": ["tripmate"],
        "metadata": {"thread_id": thread_id},
    }


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)
    blocked = result.get("guardrail_allowed", True) is False

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    # A blocked message keeps the earlier trip in memory, but must not show
    # that trip's results as if they were the answer to the blocked message.
    def specialist(key: str) -> str:
        return "" if blocked else result.get(key, "")

    dashboard = build_dashboard(result, awaiting_approval=interrupt_payload is not None)

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": specialist("flight_results"),
        "hotel_results": specialist("hotel_results"),
        "weather_results": specialist("weather_results"),
        "weather_data": {} if blocked else result.get("weather_data", {}),
        "budget_results": specialist("budget_results"),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else ("" if blocked else result.get("itinerary", ""))
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": {} if blocked else result.get("trip_constraints", {}),
        "resolved_query": "" if blocked else result.get("resolved_query", ""),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
        "workflow_status": dashboard["workflow_status"],
        "dashboard": dashboard,
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval.

    Calling it again with the same thread_id is a follow-up turn: the previous
    trip (from the thread's checkpoint) is passed to the guardrail and supervisor.
    """
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = _run_config(thread_id, "travel_plan")

    previous = (_read_state(config).values or {})
    previous_query = previous.get("resolved_query", "") or ""

    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "run_id": uuid.uuid4().hex,
        "previous_query": previous_query,
        "previous_constraints": previous.get("trip_constraints", {}) if previous_query else {},
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "selected_agents": [],
        "supervisor_reasoning": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "workflow_error": "",
        "llm_calls": 0,
    }

    if not previous_query:
        # New conversation: start with empty results. On a follow-up they are
        # NOT reset, so agents that don't re-run keep their earlier results.
        initial_state.update(
            trip_constraints=_empty_constraints(),
            flight_results="",
            hotel_results="",
            weather_results="",
            weather_data={},
            budget_results="",
            itinerary="",
        )

    result = _invoke(initial_state, config)
    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise InvalidRequestError("thread_id is required to resume a travel plan.")

    config = _run_config(thread_id, "travel_plan_resume")

    # Without this check, resuming a finished thread would silently do nothing,
    # and resuming an unknown thread would fail with an unclear KeyError.
    snapshot = _read_state(config)
    if not snapshot.values:
        raise ThreadNotFoundError()
    if "human_approval" not in (snapshot.next or ()):
        raise NoPendingApprovalError()

    result = _invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config,
    )

    return _serialize_result(result, thread_id)
