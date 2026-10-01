from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from observability import setup_logging, logger, safe_error, langsmith_status

setup_logging()

# LangSmith is optional and driven only by environment variables.
_langsmith = langsmith_status()
if _langsmith == "enabled_but_no_api_key":
    logger.warning(
        "LangSmith tracing is switched on but LANGSMITH_API_KEY is not set; "
        "traces cannot be uploaded. The app itself is unaffected."
    )
else:
    logger.info("LangSmith tracing: %s", _langsmith)

from backend import run_travel_agent, resume_travel_agent  # noqa: E402
from cache import cache_health  # noqa: E402
from errors import TripMateError  # noqa: E402

# Lets the synchronous agent functions call async MCP helpers
# while running inside FastAPI's event loop.
import nest_asyncio  # noqa: E402

nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="2.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Standard error response: {success: false, error, error_code}."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
            "error_code": code,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return _error_response(400, "invalid_request", "Message cannot be empty.")

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except TripMateError as exc:
        logger.warning("Travel request rejected (%s): %s", exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)

    except Exception as exc:
        # The full (redacted) error goes to the server log. The user only gets a
        # generic message, so internal details and API keys are never exposed.
        logger.error("Unexpected error in /api/travel: %s", safe_error(exc), exc_info=True)
        return _error_response(
            500,
            "internal_error",
            "Something went wrong while planning your trip. Please try again.",
        )


@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return _error_response(
                400,
                "invalid_request",
                "Please provide revision feedback when rejecting the draft.",
            )

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except TripMateError as exc:
        logger.warning("Approval request rejected (%s): %s", exc.code, exc.message)
        return _error_response(exc.status_code, exc.code, exc.message)

    except Exception as exc:
        logger.error("Unexpected error in /api/travel/approve: %s", safe_error(exc), exc_info=True)
        return _error_response(
            500,
            "internal_error",
            "Something went wrong while processing your approval. Please try again.",
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripMate AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
            "redis_weather_cache",
            "conversation_memory",
            "execution_dashboard",
        ],
        # Redis and LangSmith are optional; the app runs without either.
        "redis": cache_health(),
        "langsmith": langsmith_status(),
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
