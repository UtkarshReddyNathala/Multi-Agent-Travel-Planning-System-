"""Builds the Agent Execution Dashboard payload from real workflow state.

Nothing is estimated: each stage's status comes from a trace event measured by
``observability.traced_node`` during the current run, or from the graph state
(selected agents, guardrail result, human approval).
"""

from typing import Any

from schemas import Dashboard, DashboardStage

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

STAGES = [
    ("input_validation", "Input validation"),
    ("guardrail", "Guardrail"),
    ("supervisor", "Supervisor"),
    ("flight_agent", "Flight Agent"),
    ("hotel_agent", "Hotel Agent"),
    ("weather_agent", "Weather Agent"),
    ("budget_agent", "Budget Agent"),
    ("itinerary_agent", "Itinerary Agent"),
    ("human_approval", "Human approval"),
    ("final_agent", "Final response"),
]


def build_dashboard(state: dict[str, Any], awaiting_approval: bool) -> dict[str, Any]:
    run_id = state.get("run_id", "")
    blocked = state.get("guardrail_allowed", True) is False
    failed_run = bool(state.get("workflow_error"))
    selected = set(state.get("selected_agents", []))

    # Only events of the current run; the last one wins if a node ever repeats.
    events: dict[str, dict[str, Any]] = {}
    for event in state.get("execution_trace", []):
        if event.get("run_id") == run_id:
            events[event["node"]] = event

    stages: list[DashboardStage] = []
    for key, label in STAGES:
        event = events.get(key)
        if event:
            stages.append(
                DashboardStage(
                    key=key,
                    label=label,
                    status=event["status"],
                    duration_ms=event.get("duration_ms"),
                    detail=event.get("detail", ""),
                    error=event.get("error"),
                    cache=event.get("cache"),
                )
            )
            continue

        if key in AGENT_ORDER:
            if blocked or (failed_run and key == "itinerary_agent"):
                status = "not_run"
            elif key not in selected:
                status = "not_selected"
            else:
                status = "pending"
        elif key == "human_approval":
            if awaiting_approval:
                status = "awaiting"
            elif blocked or failed_run:
                status = "not_run"
            elif state.get("final_response") and "final_agent" in events:
                status = "approved" if state.get("approved") else "revision_requested"
            else:
                status = "pending"
        else:  # input_validation, guardrail, supervisor, final_agent
            status = "not_run" if blocked or failed_run else "pending"

        stages.append(DashboardStage(key=key, label=label, status=status))

    if blocked:
        workflow_status = "blocked"
    elif failed_run:
        workflow_status = "failed"
    elif awaiting_approval:
        workflow_status = "awaiting_approval"
    elif "final_agent" in events:
        workflow_status = "completed"
    else:
        workflow_status = "in_progress"

    measured = sum(e.get("duration_ms") or 0 for e in events.values())
    has_warnings = any(e["status"] in ("degraded", "failed") for e in events.values())

    return Dashboard(
        run_id=run_id,
        workflow_status=workflow_status,
        has_warnings=has_warnings,
        measured_ms=round(measured, 1),
        llm_calls=int(state.get("llm_calls", 0) or 0),
        stages=stages,
    ).model_dump()
