"""Redis cache-aside behaviour: HIT, MISS, TTL, failures, and the weather integration."""

import json

import pytest

import backend
import cache as cache_module
import weather_service
from cache import DISABLED, HIT, MISS, UNAVAILABLE, RedisCache
from schemas import WeatherResult
from tests.conftest import DownRedis, FakeRedis, blocks


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_loader(value="fresh"):
    calls = []

    def loader():
        calls.append(1)
        return value

    loader.calls = calls
    return loader


ENC, DEC = (lambda v: {"v": v}), (lambda d: d["v"])


# ============================================================ generic cache-aside
def test_miss_calls_the_service_stores_with_ttl_then_hit_skips_it():
    redis = FakeRedis()
    cache = RedisCache(redis)
    loader = make_loader("fresh")

    value, status = cache.get_or_fetch("k", 900, loader, encode=ENC, decode=DEC)
    assert (value, status) == ("fresh", MISS)
    assert len(loader.calls) == 1
    assert redis.setexes == 1 and redis.ttls["k"] == 900  # SETEX with the requested TTL
    assert json.loads(redis.store["k"]) == {"v": "fresh"}

    value, status = cache.get_or_fetch("k", 900, loader, encode=ENC, decode=DEC)
    assert (value, status) == ("fresh", HIT)
    assert len(loader.calls) == 1  # external service NOT called again
    assert redis.setexes == 1


def test_should_cache_false_is_not_stored():
    redis = FakeRedis()
    cache = RedisCache(redis)
    cache.get_or_fetch("k", 60, make_loader("bad"), encode=ENC, decode=DEC, should_cache=lambda v: False)
    assert redis.setexes == 0 and "k" not in redis.store


def test_loader_error_propagates_and_stores_nothing():
    redis = FakeRedis()

    def failing():
        raise RuntimeError("upstream down")

    with pytest.raises(RuntimeError):
        RedisCache(redis).get_or_fetch("k", 60, failing, encode=ENC, decode=DEC)
    assert redis.setexes == 0


@pytest.mark.parametrize("bad_entry", ["{not json", '{"unexpected": 1}'])
def test_corrupt_or_wrong_shape_entry_is_treated_as_a_miss_and_overwritten(bad_entry):
    redis = FakeRedis()
    redis.store["k"] = bad_entry
    loader = make_loader("fresh")
    value, status = RedisCache(redis).get_or_fetch("k", 60, loader, encode=ENC, decode=DEC)
    assert (value, status) == ("fresh", MISS) and len(loader.calls) == 1
    assert json.loads(redis.store["k"]) == {"v": "fresh"}


# ============================================================ Redis unavailable
def test_redis_down_does_not_crash_and_calls_the_live_service():
    loader = make_loader("fresh")
    value, status = RedisCache(DownRedis()).get_or_fetch("k", 60, loader, encode=ENC, decode=DEC)
    assert (value, status) == ("fresh", UNAVAILABLE)
    assert len(loader.calls) == 1


def test_redis_is_skipped_during_cooldown_then_retried():
    clock, down = Clock(), DownRedis()
    cache = RedisCache(down, cooldown_seconds=30, clock=clock)

    cache.get_or_fetch("k", 60, make_loader(), encode=ENC, decode=DEC)
    assert down.attempts == 1
    cache.get_or_fetch("k", 60, make_loader(), encode=ENC, decode=DEC)
    assert down.attempts == 1  # cooldown: no new connection attempt

    clock.now += 31
    cache.get_or_fetch("k", 60, make_loader(), encode=ENC, decode=DEC)
    assert down.attempts == 2  # cooldown over: tries again


def test_failure_when_storing_still_returns_the_fresh_value():
    class SetexFails(FakeRedis):
        def setex(self, *a):
            raise OSError("write failed")

    value, status = RedisCache(SetexFails()).get_or_fetch("k", 60, make_loader("fresh"), encode=ENC, decode=DEC)
    assert (value, status) == ("fresh", MISS)


def test_redis_recovers_after_an_outage():
    clock, redis = Clock(), FakeRedis()
    cache = RedisCache(redis, cooldown_seconds=10, clock=clock)
    cache._client.get = lambda k: (_ for _ in ()).throw(ConnectionError("down"))
    assert cache.get_or_fetch("k", 60, make_loader(), encode=ENC, decode=DEC)[1] == UNAVAILABLE
    redis2 = FakeRedis()
    cache._client = redis2
    clock.now += 11
    assert cache.get_or_fetch("k", 60, make_loader(), encode=ENC, decode=DEC)[1] == MISS
    assert cache.get_or_fetch("k", 60, make_loader(), encode=ENC, decode=DEC)[1] == HIT


# ============================================================ get_cache() / health
def test_no_redis_url_means_caching_is_disabled():
    assert cache_module.get_cache() is None
    assert cache_module.cache_health() == DISABLED


def test_malformed_redis_url_disables_caching_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "not-a-redis-url")
    monkeypatch.setattr(cache_module, "_cache", None)
    assert cache_module.get_cache() is None


def test_unreachable_redis_reports_unavailable_health(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")  # nothing listens on port 1
    monkeypatch.setattr(cache_module, "_cache", None)
    assert isinstance(cache_module.get_cache(), RedisCache)
    assert cache_module.cache_health() == UNAVAILABLE


# ============================================================ weather integration
def test_cache_key_is_normalised():
    keys = {weather_service.weather_cache_key(c) for c in ("Goa", " goa ", "GOA,", "Goa.")}
    assert keys == {"weather:goa"}
    assert weather_service.weather_cache_key("New   Delhi") == "weather:new delhi"


def test_weather_miss_then_hit(mcp):
    redis = FakeRedis()
    cache = RedisCache(redis)

    first, status1 = weather_service.get_weather("Goa", cache=cache)
    assert status1 == MISS and first.status == "ok"
    assert mcp.calls["weather"] == 1 and mcp.calls["forecast"] == 1
    assert redis.ttls["weather:goa"] == 3600  # default TTL
    assert WeatherResult.model_validate_json(redis.store["weather:goa"]) == first

    second, status2 = weather_service.get_weather("Goa", cache=cache)
    assert status2 == HIT and second == first
    assert mcp.calls["weather"] == 1 and mcp.calls["forecast"] == 1  # MCP untouched


def test_differently_spelled_destinations_share_one_cache_entry(mcp):
    cache = RedisCache(FakeRedis())
    for spelling in ("Goa", "goa", "  GOA  ", "Goa."):
        weather_service.get_weather(spelling, cache=cache)
    assert mcp.calls["weather"] == 1


def test_ttl_is_configurable(mcp, monkeypatch):
    monkeypatch.setenv("WEATHER_CACHE_TTL_SECONDS", "120")
    redis = FakeRedis()
    weather_service.get_weather("Goa", cache=RedisCache(redis))
    assert redis.ttls["weather:goa"] == 120


def test_invalid_ttl_setting_falls_back_to_default(mcp, monkeypatch):
    monkeypatch.setenv("WEATHER_CACHE_TTL_SECONDS", "soon")
    redis = FakeRedis()
    weather_service.get_weather("Goa", cache=RedisCache(redis))
    assert redis.ttls["weather:goa"] == 3600


def test_redis_down_still_returns_live_weather(mcp):
    result, status = weather_service.get_weather("Goa", cache=RedisCache(DownRedis()))
    assert status == UNAVAILABLE and result.status == "ok"
    assert result.current.temperature_c == 29.5


def test_redis_not_configured_still_returns_live_weather(mcp):
    result, status = weather_service.get_weather("Goa", cache=None)
    assert status == DISABLED and result.status == "ok"


def test_failed_lookups_are_not_cached_so_the_next_request_retries(mcp):
    redis = FakeRedis()
    cache = RedisCache(redis)
    mcp.fail |= {"weather", "forecast"}
    result, status = weather_service.get_weather("Atlantis", cache=cache)
    assert result.status == "unavailable" and status == MISS
    assert "Atlantis" in result.to_prompt_text() and "temporarily unavailable" in result.to_prompt_text()
    assert redis.setexes == 0

    mcp.fail.clear()
    result, _ = weather_service.get_weather("Atlantis", cache=cache)
    assert result.status == "ok"


def test_partial_results_are_not_cached(mcp):
    redis = FakeRedis()
    mcp.fail.add("forecast")
    result, _ = weather_service.get_weather("Goa", cache=RedisCache(redis))
    assert result.status == "partial" and result.current is not None and result.forecast == []
    assert redis.setexes == 0
    assert "Forecast data is unavailable" in result.to_prompt_text()


@pytest.mark.parametrize(
    "reply",
    [blocks("Error executing tool get_current_weather: boom"), blocks("{}"), blocks(""), [], blocks({"city": "Goa"})],
)
def test_malformed_tool_responses_become_unavailable_not_exceptions(mcp, reply):
    mcp.current_reply = reply
    mcp.forecast_reply = reply
    result, _ = weather_service.get_weather("Goa", cache=None)
    assert result.status == "unavailable" and result.error


def test_workflow_reports_cache_status_on_the_dashboard(graph, mcp, fake_redis):
    plan = "Plan a 5-day trip to Goa"
    first = backend.run_travel_agent(plan, "cache-1")
    second = backend.run_travel_agent(plan, "cache-2")

    def weather_stage(result):
        return next(s for s in result["dashboard"]["stages"] if s["key"] == "weather_agent")

    assert weather_stage(first)["cache"] == "miss"
    assert weather_stage(second)["cache"] == "hit"
    assert mcp.calls["weather"] == 1  # the second plan reused Redis
    assert second["weather_data"]["status"] == "ok"


def test_workflow_survives_redis_outage(graph, mcp, monkeypatch):
    monkeypatch.setattr(weather_service, "get_cache", lambda: RedisCache(DownRedis()))
    result = backend.run_travel_agent("Plan a 5-day trip to Goa", "cache-down")
    stage = next(s for s in result["dashboard"]["stages"] if s["key"] == "weather_agent")
    assert stage["status"] == "success" and stage["cache"] == "unavailable"
    assert result["requires_approval"] is True
