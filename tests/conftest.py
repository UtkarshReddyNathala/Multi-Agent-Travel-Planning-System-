"""Shared fixtures: scripted fake LLM, MCP stubs, fake Redis, in-memory graph.

Nothing here talks to Groq, Tavily, AviationStack, OpenWeather, PostgreSQL or
Redis. The MCP stubs return the content-block shape that langchain-mcp-adapters
really returns (verified against the installed version), e.g.
``[{"type": "text", "text": "<json string>"}]``.
"""

import os

# Must be set before `backend` / `mcp_client` are imported (they read the key at import).
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("OPENWEATHER_API_KEY", "test-weather-key")

import pytest
import redis as redis_lib
from langgraph.checkpoint.memory import InMemorySaver

import backend
import mcp_client
import weather_service
from cache import RedisCache


from evaluation.stubs import (  # noqa: E402,F401  (re-exported for the tests)
    CURRENT,
    FORECAST,
    FakeLLM,
    McpStubs,
    blocks,
    default_supervisor,
)


# ---------------------------------------------------------------------------
# Fake Redis
# ---------------------------------------------------------------------------
class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.gets = 0
        self.setexes = 0

    def get(self, key):
        self.gets += 1
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.setexes += 1
        self.store[key] = value
        self.ttls[key] = ttl
        return True

    def ping(self):
        return True


class DownRedis:
    def __init__(self):
        self.attempts = 0

    def get(self, key):
        self.attempts += 1
        raise redis_lib.ConnectionError("Error 111 connecting to redis:6379. Connection refused.")

    def setex(self, *args):
        self.attempts += 1
        raise redis_lib.ConnectionError("Connection refused")

    def ping(self):
        raise redis_lib.ConnectionError("Connection refused")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("REDIS_URL", "WEATHER_CACHE_TTL_SECONDS", "LANGSMITH_TRACING",
                 "LANGCHAIN_TRACING_V2", "MAX_QUERY_CHARS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(backend, "llm", llm)
    monkeypatch.setattr(mcp_client, "llm", llm)
    return llm


@pytest.fixture
def mcp(monkeypatch):
    stubs = McpStubs()
    monkeypatch.setattr(mcp_client, "weather_mcp_search", stubs.weather)
    monkeypatch.setattr(mcp_client, "forecast_mcp_search", stubs.forecast)
    monkeypatch.setattr(backend, "aviation_mcp_call", stubs.aviation)
    monkeypatch.setattr(backend, "tavily_mcp_search", stubs.tavily)
    return stubs


@pytest.fixture
def fake_redis(monkeypatch):
    client = FakeRedis()
    monkeypatch.setattr(weather_service, "get_cache", lambda: RedisCache(client))
    return client


@pytest.fixture
def graph(fake_llm, mcp):
    """The real workflow compiled with an in-memory checkpointer (no PostgreSQL)."""
    compiled = backend.build_travel_graph(InMemorySaver())
    backend.set_travel_graph(compiled)
    yield compiled
    backend.set_travel_graph(None)
