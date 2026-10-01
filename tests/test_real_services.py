"""Optional tests against REAL PostgreSQL and REAL Redis.

Skipped by default. Point them at throw-away services to run them:

    TRIPMATE_TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/tripmate?sslmode=disable \\
    TRIPMATE_TEST_REDIS_URL=redis://localhost:6379/15 \\
    pytest tests/test_real_services.py

The LLM and MCP tools are still scripted (no Groq/Tavily/OpenWeather calls); what is
real here is the PostgreSQL checkpointer and the Redis cache.
"""

import os
import uuid

import pytest

import backend
import cache as cache_module
import weather_service

DB_URL = os.getenv("TRIPMATE_TEST_DATABASE_URL")
REDIS_URL = os.getenv("TRIPMATE_TEST_REDIS_URL")

needs_postgres = pytest.mark.skipif(not DB_URL, reason="set TRIPMATE_TEST_DATABASE_URL to run")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="set TRIPMATE_TEST_REDIS_URL to run")


# ---------------------------------------------------------------------------
# PostgreSQL checkpointer
# ---------------------------------------------------------------------------
@pytest.fixture
def postgres_graph(monkeypatch, fake_llm, mcp):
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    backend.set_travel_graph(None)  # forget any in-memory graph; connect lazily
    backend._reset_graph()
    thread_ids: list[str] = []
    yield thread_ids
    try:
        saver = backend.get_travel_graph().checkpointer
        for thread_id in thread_ids:
            saver.delete_thread(thread_id)
    finally:
        backend._reset_graph()


@needs_postgres
def test_draft_survives_a_restart_and_can_be_approved(postgres_graph):
    thread_id = f"it-{uuid.uuid4().hex}"
    postgres_graph.append(thread_id)

    draft = backend.run_travel_agent("Plan a 5-day trip to Goa under 30000", thread_id)
    assert draft["requires_approval"] is True

    backend._reset_graph()  # simulates the server process restarting

    final = backend.resume_travel_agent(thread_id, approved=True)
    assert final["workflow_status"] == "completed" and final["approved"] is True


@needs_postgres
def test_follow_up_memory_is_read_back_from_postgres(postgres_graph):
    thread_id = f"it-{uuid.uuid4().hex}"
    postgres_graph.append(thread_id)

    backend.run_travel_agent("Plan a 5-day trip to Goa under 30000", thread_id)
    backend._reset_graph()
    second = backend.run_travel_agent("Change the budget to 25000", thread_id)

    assert second["trip_constraints"]["destination"] == "Goa"
    assert second["trip_constraints"]["duration"] == "5 days"


@needs_postgres
def test_unknown_thread_and_finished_thread_are_reported(postgres_graph):
    from errors import NoPendingApprovalError, ThreadNotFoundError

    with pytest.raises(ThreadNotFoundError):
        backend.resume_travel_agent(f"it-missing-{uuid.uuid4().hex}", approved=True)

    thread_id = f"it-{uuid.uuid4().hex}"
    postgres_graph.append(thread_id)
    backend.run_travel_agent("Plan a 5-day trip to Goa", thread_id)
    backend.resume_travel_agent(thread_id, approved=True)
    with pytest.raises(NoPendingApprovalError):
        backend.resume_travel_agent(thread_id, approved=True)


@needs_postgres
def test_missing_database_is_a_controlled_error(monkeypatch, fake_llm, mcp):
    from errors import DependencyUnavailableError

    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:9/none?sslmode=disable")
    backend.set_travel_graph(None)
    backend._reset_graph()
    with pytest.raises(DependencyUnavailableError) as caught:
        backend.run_travel_agent("Plan a trip to Goa", "it-no-db")
    assert "nothing" not in str(caught.value)  # the password is not echoed


# ---------------------------------------------------------------------------
# Redis cache
# ---------------------------------------------------------------------------
@pytest.fixture
def real_cache(monkeypatch, mcp):
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setattr(cache_module, "_cache", None)
    city = f"Zz Testcity {uuid.uuid4().hex[:8]}"
    yield city
    client = cache_module.get_cache()._client
    client.delete(weather_service.weather_cache_key(city))


@needs_redis
def test_real_redis_miss_then_hit_with_ttl(real_cache, mcp, monkeypatch):
    monkeypatch.setenv("WEATHER_CACHE_TTL_SECONDS", "120")
    first, first_status = weather_service.get_weather(real_cache)
    second, second_status = weather_service.get_weather(real_cache)

    assert (first_status, second_status) == ("miss", "hit")
    assert first.status == "ok" and second.model_dump() == first.model_dump()
    assert mcp.calls["weather"] == 1  # the second lookup never reached the MCP tool

    ttl = cache_module.get_cache()._client.ttl(weather_service.weather_cache_key(real_cache))
    assert 0 < ttl <= 120


@needs_redis
def test_real_redis_case_and_spacing_share_one_entry(real_cache, mcp):
    weather_service.get_weather(real_cache)
    _, status = weather_service.get_weather(f"  {real_cache.upper()}. ")
    assert status == "hit"


@needs_redis
def test_cache_health_reports_connected():
    os.environ["REDIS_URL"] = REDIS_URL
    try:
        cache_module._cache = None
        assert cache_module.cache_health() == "connected"
    finally:
        os.environ.pop("REDIS_URL", None)
        cache_module._cache = None
