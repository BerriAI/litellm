from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Final, Protocol, TypeAlias

from fastapi import Request
from pydantic import TypeAdapter, ValidationError
from redis.exceptions import RedisError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.constants import (
    KEY_REVOKE_MAX_REQUESTS_PER_SOURCE,
    KEY_REVOKE_THROTTLE_CACHE_KEY_PREFIX,
    KEY_REVOKE_THROTTLE_MAX_TRACKED_SOURCES,
    KEY_REVOKE_WINDOW_SECONDS,
)
from litellm.proxy.auth.login_throttle import TRUSTED_PROXY_RANGES_KEY, declared_proxy_ranges, source_group
from litellm.proxy.auth.network import TrustedProxyConfig, resolve_client_ip

_UNKNOWN_SOURCE: Final = "unknown"
_REDIS_FAILURES: Final = (RedisError, RedisCircuitBreakerOpenError, OSError, asyncio.TimeoutError, ValidationError)

_COUNT_LUA: Final = (
    "local count = redis.call('INCR', KEYS[1]) "
    "if redis.call('TTL', KEYS[1]) < 0 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end "
    "return {count, redis.call('TTL', KEYS[1])}"
)

_Hit: TypeAlias = tuple[int, int]
_HIT: Final = TypeAdapter[_Hit](_Hit)

_COUNTERS: Final = InMemoryCache(
    max_size_in_memory=KEY_REVOKE_THROTTLE_MAX_TRACKED_SOURCES, default_ttl=KEY_REVOKE_WINDOW_SECONDS
)


class CounterStore(Protocol):
    def increment_cache(self, key: str, value: float, *, ttl: int) -> float: ...


class ScriptRunner(Protocol):
    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]: ...


@cache
def warn_key_revoke_allowance_is_per_peer() -> None:
    verbose_proxy_logger.warning(
        "%s is not set or not a valid list of ranges, so POST /key/revoke counts its hourly request allowance per "
        "connecting address, and behind a load balancer every client shares one allowance. Set it to the address "
        "ranges of the proxies in front of LiteLLM, or to an empty list when clients connect directly, to count "
        "each client separately.",
        TRUSTED_PROXY_RANGES_KEY,
    )


@dataclass(frozen=True, slots=True)
class KeyRevokeRefused:
    retry_after: int


@dataclass(frozen=True, slots=True)
class KeyRevokeThrottle:
    source: str
    counters: CounterStore
    redis_cache: ScriptRunner | None

    @classmethod
    def from_request(
        cls,
        request: Request,
        general_settings: Mapping[str, object] | None,
        redis_cache: ScriptRunner | None,
        counters: CounterStore = _COUNTERS,
    ) -> KeyRevokeThrottle:
        proxies: Final = None if general_settings is None else declared_proxy_ranges(general_settings)
        if proxies is None:
            warn_key_revoke_allowance_is_per_peer()
        client_ip, _ = resolve_client_ip(
            request, TrustedProxyConfig(use_forwarded_for=bool(proxies), trusted_proxy_cidrs=proxies or ())
        )
        return cls(source=source_group(client_ip or _UNKNOWN_SOURCE), counters=counters, redis_cache=redis_cache)

    async def check(self) -> KeyRevokeRefused | None:
        count, retry_after = await self._count()
        if count <= KEY_REVOKE_MAX_REQUESTS_PER_SOURCE:
            return None
        return KeyRevokeRefused(retry_after=retry_after)

    @with_service_target("key_revoke_throttle")
    async def _count(self) -> _Hit:
        key: Final = f"{KEY_REVOKE_THROTTLE_CACHE_KEY_PREFIX}:{self.source}"
        if self.redis_cache is not None:
            try:
                hit: Final = _HIT.validate_python(
                    await self.redis_cache.async_register_script(_COUNT_LUA)([key], [KEY_REVOKE_WINDOW_SECONDS])
                )
            except _REDIS_FAILURES as err:
                verbose_proxy_logger.warning(
                    "Redis failed while counting POST /key/revoke requests; using this worker's own count until it "
                    "recovers: %s",
                    err,
                )
            else:
                return hit
        return int(self.counters.increment_cache(key, 1, ttl=KEY_REVOKE_WINDOW_SECONDS)), KEY_REVOKE_WINDOW_SECONDS
