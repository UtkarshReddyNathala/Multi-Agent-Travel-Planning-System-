"""Logging, secret redaction, optional LangSmith tracing and per-node tracing.

Nothing in this module is required for the core workflow: LangSmith is switched
on purely through environment variables and the app runs identically without it.
"""

import logging
import os
import re
import time
from functools import wraps
from typing import Any, Callable

from langgraph.errors import GraphBubbleUp

logger = logging.getLogger("tripmate")

try:  # langsmith ships as a dependency of langchain-core; the shim is defensive.
    from langsmith import traceable
except Exception:  # pragma: no cover

    def traceable(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    """Configure the root logger once (uvicorn does not configure it for us)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
_SECRET_ENV_NAMES = (
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "OPENWEATHER_API_KEY",
    "AVIATION_STACK_API_KEY",
    "AVIATIONSTACK_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)

# The Tavily key is part of the MCP URL, and OpenWeather and AviationStack
# send keys in the URL too, so error messages can contain them. Remove them.
_REDACTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(tavilyApiKey=)[^&\s'\"]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(appid=)[^&\s'\"]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(access_key=)[^&\s'\"]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(api[_-]?key=)[^&\s'\"]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(postgres(?:ql)?://[^:/\s]+:)[^@\s]+(@)"), r"\1[REDACTED]\2"),
    (re.compile(r"\bgsk_[A-Za-z0-9]+"), "[REDACTED]"),
    (re.compile(r"\btvly-[A-Za-z0-9\-]+"), "[REDACTED]"),
]


def redact(text: str) -> str:
    """Remove API keys / DB passwords from text before it is logged or shown."""
    text = str(text)
    for name in _SECRET_ENV_NAMES:
        value = os.getenv(name)
        if value and len(value) >= 6:
            text = text.replace(value, "[REDACTED]")
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def safe_error(exc: BaseException, limit: int = 200) -> str:
    """Short, secret-free description of an exception for logs and the dashboard."""
    return redact(f"{type(exc).__name__}: {exc}")[:limit]


# ---------------------------------------------------------------------------
# LangSmith (optional)
# ---------------------------------------------------------------------------
def langsmith_status() -> str:
    """'disabled' | 'enabled' | 'enabled_but_no_api_key' (from env vars only)."""
    flags = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")
    if not any(os.getenv(name, "").strip().lower() in ("1", "true", "yes") for name in flags):
        return "disabled"
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        return "enabled"
    return "enabled_but_no_api_key"


# ---------------------------------------------------------------------------
# Per-node execution trace (feeds the dashboard)
# ---------------------------------------------------------------------------
def make_event(
    run_id: str,
    node: str,
    status: str,
    duration_ms: float | None = None,
    detail: str = "",
    error: str | None = None,
    cache: str | None = None,
) -> dict[str, Any]:
    from schemas import TraceEvent  # local import avoids a circular import

    return TraceEvent(
        run_id=run_id,
        node=node,
        status=status,
        duration_ms=None if duration_ms is None else round(duration_ms, 1),
        detail=detail,
        error=error,
        cache=cache,
    ).model_dump()


def traced_node(name: str, *, catch: bool = True) -> Callable:
    """Wrap a LangGraph node so every execution leaves a *measured* trace event.

    A node reports richer information by adding a ``_trace`` entry to the dict it
    returns: ``{"status": ..., "detail": ..., "error": ..., "cache": ...}`` (or a
    list of such dicts, each optionally overriding ``node``/``duration_ms``).

    With ``catch=True`` an unexpected exception is logged and recorded as a failed
    event instead of aborting the whole run. ``GraphBubbleUp`` (which LangGraph
    uses for ``interrupt()``) is always re-raised.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            run_id = state.get("run_id", "") if hasattr(state, "get") else ""
            try:
                update = fn(state) or {}
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if not catch:
                    raise
                logger.exception("Node '%s' failed unexpectedly", name)
                elapsed = (time.perf_counter() - start) * 1000
                return {
                    "execution_trace": [
                        make_event(run_id, name, "failed", elapsed, error=safe_error(exc))
                    ]
                }

            elapsed = (time.perf_counter() - start) * 1000
            meta = update.pop("_trace", None)
            if meta is None:
                meta = {"status": "success"}
            metas = meta if isinstance(meta, list) else [meta]

            events = []
            for item in metas:
                item = dict(item)
                events.append(
                    make_event(
                        run_id,
                        item.pop("node", name),
                        item.pop("status", "success"),
                        item.pop("duration_ms", elapsed),
                        **item,
                    )
                )
            update["execution_trace"] = events
            return update

        return wrapper

    return decorator
