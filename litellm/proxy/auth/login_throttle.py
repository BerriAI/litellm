"""Failed-login accounting for the Admin UI sign-in path.

Counts failed credential checks over a fixed window against two independent keys, the
username on its own and the source address on its own, so that one username attacked from
many sources and one source spraying many usernames are both counted. Repeated failures
are answered slowly, doubling from one second, and refused with 429 once either counter
reaches its limit. Built per request by ``LoginThrottle.from_request`` because it carries
that request's resolved source address, and because the coordination cache is assigned at
startup and can be reassigned later.
"""

import asyncio
import hashlib
from collections.abc import Awaitable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NamedTuple, NoReturn

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.network import TrustedProxyConfig, normalize_cidr_ranges, resolve_client_ip
from litellm.proxy.auth.trusted_proxy_utils import TRUSTED_PROXY_RANGES_KEY
from litellm.secret_managers.main import get_secret_bool

DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS: Final = 50
DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_SOURCE: Final = 250
DEFAULT_FAILED_LOGIN_WINDOW_SECONDS: Final = 900

USERNAME_DELAY_ONSET: Final = 3
SOURCE_DELAY_ONSET: Final = 25
FIRST_DELAY_SECONDS: Final = 1.0
MAX_DELAY_SECONDS: Final = 30.0
MAX_CONCURRENT_DELAYS_PER_SOURCE: Final = 5

_MAX_DELAY_DOUBLINGS: Final = 16

_CACHE_KEY_PREFIX: Final = "login_fail"
_UNKNOWN_SOURCE: Final = "unknown"
_MAX_LOGGED_USERNAME_CHARS: Final = 128

_MAX_TRACKED_LOGIN_USERNAMES: Final = 10_000
_MAX_TRACKED_LOGIN_SOURCES: Final = 10_000


def _bounded_store(max_entries: int) -> DualCache:
    return DualCache(
        in_memory_cache=InMemoryCache(max_size_in_memory=max_entries),
        default_in_memory_ttl=DEFAULT_FAILED_LOGIN_WINDOW_SECONDS,
    )


# Separate stores: eviction is earliest-expiring-first, so in one shared store a spray of
# fresh usernames would evict the source counter that is meant to stop that same spray.
_FAILED_LOGIN_USERNAME_CACHE: Final = _bounded_store(_MAX_TRACKED_LOGIN_USERNAMES)
_FAILED_LOGIN_SOURCE_CACHE: Final = _bounded_store(_MAX_TRACKED_LOGIN_SOURCES)
_NO_SETTINGS: Final = MappingProxyType({})

_DELAYS_IN_FLIGHT: Final[dict[str, int]] = {}  # mutable-ok: per-source slots taken and released around each held delay


async def _sleep(seconds: float) -> None:
    """The wait a rejected sign-in is held for. Replaced in tests so the suite pays no wall clock."""
    await asyncio.sleep(seconds)


class FailureCounts(NamedTuple):
    """Failures recorded so far in this window against each of the two keys."""

    username: int
    source: int


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
    """Fixed-window failed-login accounting for one request's username and source address."""

    client_ip: str
    max_attempts: int
    max_attempts_per_source: int
    window_seconds: int
    username_cache: DualCache
    source_cache: DualCache
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
            max_attempts_per_source=_int_setting(
                "max_failed_login_attempts_per_source",
                settings.get("max_failed_login_attempts_per_source"),
                DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_SOURCE,
                1,
            ),
            window_seconds=_int_setting(
                "failed_login_window_seconds",
                settings.get("failed_login_window_seconds"),
                DEFAULT_FAILED_LOGIN_WINDOW_SECONDS,
                1,
            ),
            username_cache=_FAILED_LOGIN_USERNAME_CACHE,
            source_cache=_FAILED_LOGIN_SOURCE_CACHE,
            redis_cache=redis_usage_cache,
            enabled=not get_secret_bool("LITELLM_DISABLE_LOGIN_RATE_LIMIT", False),
        )

    @staticmethod
    def _loggable(username: str) -> str:
        """The username with anything that could forge a log line removed."""
        return "".join(c for c in username if c.isprintable())[:_MAX_LOGGED_USERNAME_CHARS]

    @staticmethod
    def _username_key(username: str) -> str:
        identity: Final = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
        return f"{_CACHE_KEY_PREFIX}:user:{identity}"

    def _source_key(self) -> str:
        return f"{_CACHE_KEY_PREFIX}:source:{self.client_ip}"

    async def _outcome(self, work: Awaitable[object]) -> object:
        try:
            return await work
        except Exception as exc:  # noqa: BLE001  # an unreachable cache must never deny a valid credential
            verbose_proxy_logger.warning("login attempt accounting unavailable: %s", exc)
            return None

    async def _failures(self, store: DualCache, key: str) -> int:
        """The larger of the shared and the process-local count, so a Redis outage degrades
        to per-worker accounting instead of switching the control off."""
        local: Final = _as_count(await self._outcome(store.async_get_cache(key=key)))
        redis_cache: Final = self.redis_cache
        if redis_cache is None:
            return local
        return max(local, _as_count(await self._outcome(redis_cache.async_get_cache(key))))

    async def _remaining_window(self, key: str) -> int:
        """Seconds until this counter expires, repairing a counter left without an expiry.

        Redis commits the increment before setting the TTL, so a failure in between can
        leave a counter that never expires. Nothing increments the key again once the
        limit is reached, so without the repair the key would stay refused indefinitely.
        """
        redis_cache: Final = self.redis_cache
        if redis_cache is None:
            return self.window_seconds
        ttl: Final = await self._outcome(redis_cache.async_get_ttl(key))
        if isinstance(ttl, int) and ttl > 0:
            return min(ttl, self.window_seconds)
        await self._outcome(redis_cache.async_increment(key, 0, ttl=self.window_seconds))
        return self.window_seconds

    def _refused(self, retry_after: int, param: str) -> ProxyException:
        return ProxyException(
            message="Too many failed sign-in attempts. Try again later.",
            type=ProxyErrorTypes.auth_error,
            param=param,
            code=429,
            headers={"Retry-After": str(retry_after)},  # mutable-ok: ProxyException coerces header values
        )

    async def _refuse(self, key: str, scope: str, param: str, username: str, failures: int, limit: int) -> NoReturn:
        retry_after: Final = await self._remaining_window(key)
        verbose_proxy_logger.warning(
            "Admin UI sign-in attempts exhausted for %s; username=%s source=%s failures=%s limit=%s window=%ss",
            scope,
            self._loggable(username),
            self.client_ip,
            failures,
            limit,
            self.window_seconds,
        )
        raise self._refused(retry_after, param)

    async def raise_if_blocked(self, username: str) -> None:
        """Refuse before the database lookup and before the invite-link password hash."""
        if not self.enabled:
            return
        username_key: Final = self._username_key(username)
        source_key: Final = self._source_key()
        username_failures: Final = await self._failures(self.username_cache, username_key)
        if username_failures >= self.max_attempts:
            await self._refuse(
                username_key, "username", "max_failed_login_attempts", username, username_failures, self.max_attempts
            )
        source_failures: Final = await self._failures(self.source_cache, source_key)
        if source_failures >= self.max_attempts_per_source:
            await self._refuse(
                source_key,
                "source address",
                "max_failed_login_attempts_per_source",
                username,
                source_failures,
                self.max_attempts_per_source,
            )

    async def _bump(self, store: DualCache, key: str) -> int:
        local: Final = _as_count(
            await self._outcome(store.async_increment_cache(key=key, value=1, ttl=self.window_seconds))
        )
        redis_cache: Final = self.redis_cache
        if redis_cache is None:
            return local
        shared: Final = _as_count(await self._outcome(redis_cache.async_increment(key, 1, ttl=self.window_seconds)))
        await self._remaining_window(key)
        return max(local, shared)

    async def record_failure(self, username: str) -> FailureCounts:
        """Count one rejected credential guess against this username and against this source."""
        if not self.enabled:
            return FailureCounts(username=0, source=0)
        return FailureCounts(
            username=await self._bump(self.username_cache, self._username_key(username)),
            source=await self._bump(self.source_cache, self._source_key()),
        )

    @staticmethod
    def delay_seconds(counts: FailureCounts) -> float:
        """Seconds to hold a rejected attempt for, doubling per failure past whichever onset is further along."""
        steps: Final = min(
            max(counts.username - USERNAME_DELAY_ONSET, counts.source - SOURCE_DELAY_ONSET),
            _MAX_DELAY_DOUBLINGS,
        )
        if steps < 0:
            return 0.0
        return min(FIRST_DELAY_SECONDS * float(2**steps), MAX_DELAY_SECONDS)

    async def delay_for(self, username: str, counts: FailureCounts) -> None:
        """Hold this rejected attempt open before answering it, so guessing costs wall-clock time.

        Only ever reached once the credentials are known to be wrong, so a valid password is
        never delayed. Sources are capped at ``MAX_CONCURRENT_DELAYS_PER_SOURCE`` held
        connections; over that, the attempt is refused immediately instead of parking a socket.
        """
        if not self.enabled:
            return
        delay: Final = self.delay_seconds(counts)
        if delay <= 0:
            return
        in_flight: Final = _DELAYS_IN_FLIGHT.get(self.client_ip, 0)
        if in_flight >= MAX_CONCURRENT_DELAYS_PER_SOURCE:
            verbose_proxy_logger.warning(
                "Admin UI sign-in attempts held concurrently exhausted; username=%s source=%s in_flight=%s",
                self._loggable(username),
                self.client_ip,
                in_flight,
            )
            raise self._refused(int(MAX_DELAY_SECONDS), "concurrent_failed_logins")
        _DELAYS_IN_FLIGHT[self.client_ip] = in_flight + 1
        try:
            await _sleep(delay)
        finally:
            remaining: Final = _DELAYS_IN_FLIGHT.get(self.client_ip, 1) - 1
            if remaining > 0:
                _DELAYS_IN_FLIGHT[self.client_ip] = remaining
            else:
                _DELAYS_IN_FLIGHT.pop(self.client_ip, None)

    async def clear(self, username: str) -> None:
        """Drop the username counter after a successful sign-in.

        The source counter is left alone. It is shared by every account behind that address,
        so one success there says nothing about the other attempts it is counting.
        """
        if not self.enabled:
            return
        key: Final = self._username_key(username)
        redis_cache: Final = self.redis_cache
        if redis_cache is not None:
            await self._outcome(redis_cache.async_delete_cache(key))
        await self._outcome(self.username_cache.async_delete_cache(key=key))
