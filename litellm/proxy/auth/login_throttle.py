"""Failed-login accounting for the Admin UI sign-in path.

Counts failed credential checks per (username, source address) over a fixed window and
denies further attempts with 429 once the count reaches the limit. Built per request by
``LoginThrottle.from_request`` because it carries that request's resolved source address,
and because the coordination cache is assigned at startup and can be reassigned later.
"""

import hashlib
from collections.abc import Awaitable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.network import TrustedProxyConfig, normalize_cidr_ranges, resolve_client_ip
from litellm.proxy.auth.trusted_proxy_utils import TRUSTED_PROXY_RANGES_KEY
from litellm.secret_managers.main import get_secret_bool

DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS: Final = 10
DEFAULT_FAILED_LOGIN_WINDOW_SECONDS: Final = 900

_CACHE_KEY_PREFIX: Final = "login_fail"
_UNKNOWN_SOURCE: Final = "unknown"
_MAX_LOGGED_USERNAME_CHARS: Final = 128

_MAX_TRACKED_LOGIN_SOURCES: Final = 10_000

_FAILED_LOGIN_CACHE: Final = DualCache(
    in_memory_cache=InMemoryCache(max_size_in_memory=_MAX_TRACKED_LOGIN_SOURCES),
    default_in_memory_ttl=DEFAULT_FAILED_LOGIN_WINDOW_SECONDS,
)
_NO_SETTINGS: Final = MappingProxyType({})


def _int_setting(name: str, value: object, default: int, minimum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        verbose_proxy_logger.warning(
            "general_settings.%s=%r is not an integer >= %s; using the default of %s", name, value, minimum, default
        )
        return default
    return value


def _as_count(cached: object) -> int:
    return int(cached) if isinstance(cached, int | float) and not isinstance(cached, bool) else 0


@dataclass(frozen=True, slots=True)
class LoginThrottle:
    """Fixed-window failed-login accounting for one request's source address."""

    client_ip: str
    max_attempts: int
    window_seconds: int
    cache: DualCache
    redis_cache: RedisCache | None = None
    enabled: bool = True

    @classmethod
    def from_request(cls, request: Request) -> "LoginThrottle":
        """Build the throttle for this request from the live proxy settings and caches."""
        from litellm.proxy.proxy_server import general_settings, redis_usage_cache

        settings: Final = general_settings or _NO_SETTINGS
        cidrs: Final = normalize_cidr_ranges(
            settings.get(TRUSTED_PROXY_RANGES_KEY), setting_name=TRUSTED_PROXY_RANGES_KEY
        )
        resolved, _ = resolve_client_ip(
            request, TrustedProxyConfig(use_forwarded_for=bool(cidrs), trusted_proxy_cidrs=cidrs)
        )
        return cls(
            client_ip=resolved or _UNKNOWN_SOURCE,
            max_attempts=_int_setting(
                "max_failed_login_attempts",
                settings.get("max_failed_login_attempts"),
                DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS,
                1,
            ),
            window_seconds=_int_setting(
                "failed_login_window_seconds",
                settings.get("failed_login_window_seconds"),
                DEFAULT_FAILED_LOGIN_WINDOW_SECONDS,
                1,
            ),
            cache=_FAILED_LOGIN_CACHE,
            redis_cache=redis_usage_cache,
            enabled=not get_secret_bool("LITELLM_DISABLE_LOGIN_RATE_LIMIT"),
        )

    @staticmethod
    def _loggable(username: str) -> str:
        """The username with anything that could forge a log line removed."""
        return "".join(c for c in username if c.isprintable())[:_MAX_LOGGED_USERNAME_CHARS]

    def _key(self, username: str) -> str:
        identity: Final = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
        return f"{_CACHE_KEY_PREFIX}:{identity}:{self.client_ip}"

    async def _outcome(self, work: Awaitable[object]) -> object:
        try:
            return await work
        except Exception as exc:  # noqa: BLE001  # an unreachable cache must never deny a valid credential
            verbose_proxy_logger.warning("login attempt accounting unavailable: %s", exc)
            return None

    async def _failures(self, key: str) -> int:
        store: Final = self.cache if self.redis_cache is None else self.redis_cache
        return _as_count(await self._outcome(store.async_get_cache(key=key)))

    async def _ensure_expiry(self, key: str) -> None:
        """Give the counter an expiry if it somehow has none.

        Redis commits the increment before setting the TTL, so a failure in between can
        leave a counter that never expires. Nothing increments the key again once the
        limit is reached, so without this the pair would stay refused indefinitely.
        """
        redis_cache: Final = self.redis_cache
        if redis_cache is None:
            return
        if isinstance(await self._outcome(redis_cache.async_get_ttl(key)), int):
            return
        await self._outcome(redis_cache.async_increment(key, 0, ttl=self.window_seconds))

    async def raise_if_blocked(self, username: str) -> None:
        """Deny before the database lookup and before the password comparison."""
        if not self.enabled:
            return
        key: Final = self._key(username)
        if await self._failures(key) < self.max_attempts:
            return
        await self._ensure_expiry(key)
        verbose_proxy_logger.warning(
            "Admin UI sign-in attempts exhausted for username=%s source=%s; %s attempts in %ss, retry after %ss",
            self._loggable(username),
            self.client_ip,
            self.max_attempts,
            self.window_seconds,
            self.window_seconds,
        )
        raise ProxyException(
            message="Too many failed sign-in attempts. Try again later.",
            type=ProxyErrorTypes.auth_error,
            param="max_failed_login_attempts",
            code=429,
            headers={"Retry-After": str(self.window_seconds)},  # mutable-ok: ProxyException coerces header values
        )

    async def record_failure(self, username: str) -> None:
        """Count one rejected credential guess against this username and source."""
        if not self.enabled:
            return
        key: Final = self._key(username)
        redis_cache: Final = self.redis_cache
        if redis_cache is not None:
            await self._outcome(redis_cache.async_increment(key, 1, ttl=self.window_seconds))
            await self._ensure_expiry(key)
            return
        await self._outcome(self.cache.async_increment_cache(key=key, value=1, ttl=self.window_seconds))

    async def clear(self, username: str) -> None:
        """Drop the bucket after a successful sign-in."""
        if not self.enabled:
            return
        key: Final = self._key(username)
        redis_cache: Final = self.redis_cache
        if redis_cache is not None:
            await self._outcome(redis_cache.async_delete_cache(key))
        await self._outcome(self.cache.async_delete_cache(key=key))
