"""Redis cache-aside helper.

Redis is used ONLY as a short-lived cache for repeated external-data requests
(currently: weather). It does not store LangGraph state, conversation memory or
anything permanent - that is PostgreSQL's job.

    Check Redis (GET)
        |
    Cache HIT? --yes--> return cached value
        |no
    call the external service (loader)
        |
    store in Redis with a TTL (SETEX)
        |
    return the fresh value

Redis is an optimisation, never a single point of failure: any Redis problem is
logged and treated as "no cache", so the live weather service is still called.
"""

import json
import threading
import time
from typing import Any, Callable

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from config import redis_url
from observability import logger, safe_error

_UNAVAILABLE = object()

# Cache status values reported to the dashboard / traces.
HIT = "hit"
MISS = "miss"
UNAVAILABLE = "unavailable"  # Redis configured but failing -> bypassed
DISABLED = "disabled"  # REDIS_URL not set -> no caching


class RedisCache:
    def __init__(
        self,
        client: Any,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client = client
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._down_until = 0.0

    # -- internals ---------------------------------------------------------
    def _safe(self, operation: Callable[[], Any]) -> Any:
        """Run a Redis call; on any failure log it and return _UNAVAILABLE.

        After a failure Redis is skipped for ``cooldown_seconds`` so a dead Redis
        does not add a connection attempt to every single request.
        """
        if self._clock() < self._down_until:
            return _UNAVAILABLE
        try:
            return operation()
        except Exception as exc:
            self._down_until = self._clock() + self._cooldown
            logger.warning(
                "Redis unavailable, bypassing cache for %.0fs: %s",
                self._cooldown,
                safe_error(exc),
            )
            return _UNAVAILABLE

    # -- public API --------------------------------------------------------
    def get_or_fetch(
        self,
        key: str,
        ttl: int,
        loader: Callable[[], Any],
        *,
        encode: Callable[[Any], Any],
        decode: Callable[[Any], Any],
        should_cache: Callable[[Any], bool] = lambda value: True,
    ) -> tuple[Any, str]:
        """Cache-aside lookup. Returns ``(value, status)``.

        ``encode`` turns a value into JSON-serialisable data, ``decode`` turns
        cached data back into a value (and must raise if the data is invalid,
        which is then treated as a miss). ``loader`` errors propagate to the caller.
        """
        # 1. Check Redis
        cached = self._safe(lambda: self._client.get(key))

        if cached is not _UNAVAILABLE and cached is not None:
            # 2. Cache HIT
            try:
                return decode(json.loads(cached)), HIT
            except Exception as exc:
                logger.warning("Ignoring invalid cache entry %s: %s", key, safe_error(exc))

        status = UNAVAILABLE if cached is _UNAVAILABLE else MISS

        # 3. Cache MISS (or Redis down): call the live weather service
        fresh = loader()

        # 4. Store the result with a TTL, then return it
        if status == MISS and should_cache(fresh):
            payload = json.dumps(encode(fresh))
            self._safe(lambda: self._client.setex(key, ttl, payload))
        return fresh, status


# ---------------------------------------------------------------------------
# Shared instance (created lazily from REDIS_URL)
# ---------------------------------------------------------------------------
_cache: RedisCache | None = None
_cache_url: str | None = None
_lock = threading.Lock()


def get_cache() -> RedisCache | None:
    """Return the shared cache, or None when REDIS_URL is not configured."""
    global _cache, _cache_url
    url = redis_url()
    if not url:
        return None
    with _lock:
        if _cache is None or _cache_url != url:
            try:
                client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                    retry=Retry(NoBackoff(), 0),  # fail fast: a cache must never slow requests
                )
            except Exception as exc:  # e.g. malformed REDIS_URL
                logger.error("Invalid REDIS_URL, caching disabled: %s", safe_error(exc))
                return None
            _cache, _cache_url = RedisCache(client), url
        return _cache


def cache_health() -> str:
    """'disabled' | 'connected' | 'unavailable' - used by /health."""
    cache = get_cache()
    if cache is None:
        return DISABLED
    try:
        cache._client.ping()
        return "connected"
    except Exception:
        return UNAVAILABLE
