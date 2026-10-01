"""Weather lookup used by the Weather Agent: Redis cache in front of the weather MCP.

    Weather Agent
         |
    get_weather(city)
         |
    Redis GET  weather:<normalised city>
         |-- HIT  --> cached WeatherResult
         '-- MISS --> weather MCP tools (get_current_weather + get_forecast)
                         |
                      Redis SETEX (TTL) - only for complete, successful results
                         |
                      WeatherResult

The cache lives here, outside the MCP layer, so the MCP server stays simple.
"""

import asyncio
import re
import unicodedata

import mcp_client
from cache import DISABLED, get_cache
from config import weather_cache_ttl
from observability import logger, safe_error, traceable
from schemas import CurrentWeather, ForecastEntry, WeatherResult

_UNSET = object()


def normalize_city(city: str) -> str:
    """'  Goa, ' / 'GOA' / 'goa' -> 'goa' so spelling of case/spacing shares one key."""
    text = unicodedata.normalize("NFKC", city or "")
    text = text.casefold().strip().strip("\"'`.,;:!? ")
    return re.sub(r"\s+", " ", text)


def weather_cache_key(city: str) -> str:
    return f"weather:{normalize_city(city)}"


async def _fetch_live(city: str) -> WeatherResult:
    """Call the existing weather MCP tools and validate what they return."""
    current_raw, forecast_raw = await asyncio.gather(
        mcp_client.weather_mcp_search(city),
        mcp_client.forecast_mcp_search(city),
        return_exceptions=True,
    )

    errors: list[str] = []
    current = None
    forecast: list[ForecastEntry] = []

    if isinstance(current_raw, BaseException):
        errors.append(f"current weather: {safe_error(current_raw)}")
    else:
        try:
            current = CurrentWeather.model_validate(mcp_client.mcp_result_json(current_raw))
        except Exception as exc:
            errors.append(f"current weather: {safe_error(exc)}")

    if isinstance(forecast_raw, BaseException):
        errors.append(f"forecast: {safe_error(forecast_raw)}")
    else:
        try:
            payload = mcp_client.mcp_result_json(forecast_raw)
            forecast = [ForecastEntry.model_validate(item) for item in payload["forecast"]]
        except Exception as exc:
            errors.append(f"forecast: {safe_error(exc)}")

    if current is not None and forecast:
        status = "ok"
    elif current is not None or forecast:
        status = "partial"
    else:
        status = "unavailable"

    if errors:
        logger.warning("Weather MCP problem for '%s': %s", city, "; ".join(errors))

    return WeatherResult(
        status=status,
        city=city,
        current=current,
        forecast=forecast,
        error="; ".join(errors)[:300] if errors else None,
    )


@traceable(run_type="chain", name="get_weather")
def get_weather(city: str, cache=_UNSET) -> tuple[WeatherResult, str]:
    """Return ``(WeatherResult, cache_status)``. Never raises for data problems.

    ``cache_status`` is one of: hit | miss | unavailable | disabled.
    """
    if cache is _UNSET:
        cache = get_cache()

    def load() -> WeatherResult:
        return asyncio.run(_fetch_live(city))

    if cache is None:
        return load(), DISABLED

    result, status = cache.get_or_fetch(
        weather_cache_key(city),
        weather_cache_ttl(),
        load,
        encode=lambda r: r.model_dump(),
        decode=WeatherResult.model_validate,
        # Never cache failures or half-results: the next request should retry.
        should_cache=lambda r: r.status == "ok",
    )
    return result, status
