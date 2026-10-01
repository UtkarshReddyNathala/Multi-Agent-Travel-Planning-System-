"""App settings, read from environment variables.

Each value is read by a function (not stored once at import), so it is always
current and tests can change it.
"""

import os


def _number_env(name: str, default: float, cast=float, minimum: float = 0):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
    except ValueError:
        return default
    return value if value > minimum else default


def redis_url() -> str | None:
    """Redis is optional. When REDIS_URL is unset, caching is simply disabled."""
    url = (os.getenv("REDIS_URL") or "").strip()
    return url or None


def weather_cache_ttl() -> int:
    """Seconds a weather result stays in Redis (default: 1 hour)."""
    return _number_env("WEATHER_CACHE_TTL_SECONDS", 3600, cast=int)


def mcp_timeout() -> float:
    """Seconds allowed for a single MCP call (loading a tool or invoking it)."""
    return _number_env("MCP_TIMEOUT_SECONDS", 60.0)


def llm_timeout() -> float:
    """Seconds allowed for a single Groq request."""
    return _number_env("LLM_TIMEOUT_SECONDS", 60.0)


def max_query_chars() -> int:
    """Longest user message the guardrail accepts."""
    return _number_env("MAX_QUERY_CHARS", 2000, cast=int)
