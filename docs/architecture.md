# Architecture

This guide explains how TripMate AI works inside: the workflow, each agent, the guardrail, memory, caching, error handling and the dashboard.

## Workflow

The workflow is a LangGraph `StateGraph`. Every step (**node**) reads and updates one shared state, and the routing functions in `backend.py` decide which node runs next.

```text
START → guardrail ──blocked──→ guardrail_blocked → END
            │
          allowed
            ↓
        supervisor → [flight → hotel → weather → budget] → itinerary → human_approval → final → END
                     (only the agents the supervisor picked,
                      always in this order)
```

| Node | What it does | Uses |
|---|---|---|
| `guardrail` | Checks the message: validation, rules, then an LLM classifier | Groq LLM |
| `supervisor` | Picks the agents, extracts the trip details, handles follow-ups | Groq LLM |
| `flight_agent` | Airport and airline data plus LLM guidance | AviationStack MCP, LLM |
| `hotel_agent` | Hotel search | Tavily MCP |
| `weather_agent` | Current weather and forecast (cached) | Weather MCP server, Redis |
| `budget_agent` | Checks if the trip fits the budget | LLM |
| `itinerary_agent` | Writes the draft plan from the agents' results | LLM |
| `human_approval` | Pauses the workflow with `interrupt()` until the user responds | - |
| `final_agent` | Writes the final plan, applying the user's feedback | LLM |

The state is saved after every step in PostgreSQL (`PostgresSaver`). That is what lets a draft wait for approval and still be there after a server restart.

## Guardrail (`guardrails.py`)

Every message goes through three layers, from cheapest to most expensive:

| Layer | Function | What it checks |
|---|---|---|
| 1. Input validation | `validate_input()` | Empty, too long (`MAX_QUERY_CHARS`) or symbol-only messages. Control characters are removed. |
| 2. Rules | `rule_check()` | Regex patterns for prompt injection ("ignore previous instructions", "developer mode") and clearly unsafe requests (smuggling, forged passports or visas) |
| 3. LLM classifier | `llm_check()` | Is this really a travel request? |

How the layers are designed:

- **The rules are deliberately narrow.** Words like "avoid" and "bypass" aren't blocked, because "how do I avoid immigration queues?" is a normal travel question.
- **The LLM sees the previous turn**, so a follow-up like "change the budget to 25000" isn't judged off-topic.
- **The user's text is wrapped in `<user_request>` tags**, so the model can tell it apart from the instructions.
- **The LLM's answer is parsed strictly with Pydantic.** A naive `bool("false")` is `True` in Python, which would let blocked requests through.
- **If the LLM fails, the request is allowed (fail open)**, so real users aren't blocked by an outage. The dashboard marks the guardrail as *degraded*, so this is visible.

## Supervisor and conversation memory

The supervisor returns JSON (validated as `SupervisorPlan`) with the agents to run, the trip details (`TripConstraints`) and a short reason.

Sending a message with the same `thread_id` makes it a follow-up:

```text
Turn 1: "Plan a 5-day trip to Goa under 30000"   → trip details saved in the checkpoint
Turn 2: "Change the budget to 25000"  (same thread_id)
          → the supervisor sees turn 1 and returns only the change
          → merge_constraints() keeps Goa and 5 days
          → only the affected agents run again; the others keep their earlier results
          → a new destination or origin clears all earlier results and runs every agent
```

- An off-topic message in the middle of a conversation is blocked **without erasing the trip**.
- If the supervisor's JSON can't be parsed, it falls back to running **every agent**.
- The web UI keeps the `thread_id` in `localStorage`; **Start new conversation** clears it.

## MCP tools (`mcp_client.py`)

| Server | Transport | Used by |
|---|---|---|
| Tavily | Streamable HTTP | Hotel agent |
| AviationStack | stdio, started with `uvx aviationstack-mcp` | Flight agent |
| Weather (`custom_weather_mcp_server.py`) | stdio, a Python subprocess | Weather agent |

- Each tool is loaded from its own server only, so a broken weather server can't break a hotel search.
- Every MCP call has a timeout (`MCP_TIMEOUT_SECONDS`).
- MCP errors are **raised** (`handle_tool_error = False`) instead of being returned as text, so agents never treat an error message as real data.

## Redis weather cache (`cache.py`, `weather_service.py`)

```text
Weather agent → get_weather(city)
                   │
        Redis GET weather:<normalised city>
           ├─ HIT  → cached WeatherResult (no MCP call)
           └─ MISS → weather MCP (current + forecast, in parallel)
                       → validate → Redis SETEX with TTL (only if complete) → WeatherResult
```

- **Only weather is cached.** It's small, shared by many users and fine if slightly old. Flight and hotel results depend on the full request, so they aren't cached.
- **City names are normalised**: `Goa`, `GOA` and ` goa. ` all use the key `weather:goa`.
- **Failed or partial results are never cached**, so the next request tries again.
- **Redis is optional.** If it's down, the live result is used and Redis is skipped for 30 seconds, so a dead cache doesn't slow down every request.
- The dashboard shows the cache result: `hit`, `miss`, `unavailable` or `disabled`.

## Error handling

| Area | How errors are handled |
|---|---|
| Each agent | If one agent fails, it returns "temporarily unavailable", its dashboard stage shows *degraded*, and the run continues |
| Final agent | If it fails after approval, the approved draft is returned instead of losing the run |
| Timeouts | Every LLM call (`LLM_TIMEOUT_SECONDS`) and MCP call (`MCP_TIMEOUT_SECONDS`) has a limit |
| Secrets | `redact()` removes API keys from every log line and user message. This matters because the Tavily key is part of the MCP URL. |
| PostgreSQL | Connects on first use (not at import), reconnects once if the connection dropped, and returns `503` if it's down |
| API | Clear error codes: `400`, `404`, `409`, `503`, and a generic `500` |
| Browser | The plan is rendered from Markdown and sanitised with DOMPurify, so HTML or script in LLM or web output can't run |

## Execution dashboard (`dashboard.py`, `observability.py`)

Every agent is wrapped with `traced_node`, which records a trace event for each run: status, measured duration, error and cache result. `build_dashboard()` turns these events into the `dashboard` object in every API response, and the UI shows it.

- Every value is measured during the current run. Nothing is estimated.
- Agents that haven't run show as *pending*, *not selected* or *not run*.
- The LLM-call count only includes successful LLM responses.

## LangSmith tracing (optional)

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace every LangGraph step, LLM call and tool call. Runs are named and tagged with the `thread_id`, and `/health` shows the tracing status. Without it, the app works exactly the same. If tracing is on but LangSmith can't be reached, the workflow still completes. Note that traces include user messages and model output.

## Settings

All settings come from environment variables (see `.env.example`).

| Setting | What it is for | Default |
|---|---|---|
| `GROQ_API_KEY` | LLM (required) | - |
| `TAVILY_API_KEY` | Hotel search | - |
| `OPENWEATHER_API_KEY` | Weather MCP server | - |
| `AVIATION_STACK_API_KEY` | Flight data (optional; flights show as unavailable without it) | - |
| `DATABASE_URL` | PostgreSQL for memory and checkpoints (`sslmode=require` is added unless set) | - |
| `REDIS_URL` | Weather cache (unset = no caching) | unset |
| `WEATHER_CACHE_TTL_SECONDS` | How long a weather result is cached | `3600` |
| `MCP_TIMEOUT_SECONDS`, `LLM_TIMEOUT_SECONDS` | Time limit for one tool or LLM call | `60` |
| `MAX_QUERY_CHARS` | Longest message accepted | `2000` |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional tracing | off |

## API errors

Errors always have the shape `{"success": false, "error": "...", "error_code": "..."}`.

| Code | Meaning |
|---|---|
| `400` | Invalid input |
| `404` | Unknown conversation (`thread_id`) |
| `409` | No draft waiting for approval |
| `503` | Database unavailable |
| `500` | Unexpected error (details only in the server log) |
