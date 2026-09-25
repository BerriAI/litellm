"""Failed-login accounting for the Admin UI sign-in path.

Wrong passwords are counted over a short window per source address and per source-and-username
pair; too many in one window blocks that key for a fixed time. While a key is blocked every attempt
from it, right or wrong, is refused with 429 before the password is checked. A blocked pair stops
counting against its source, so one script stuck on one account does not block the whole office.
Recovery is the master key over the API, which never passes through here, or waiting out the block.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Final, Literal, NamedTuple, Protocol, TypeAlias

from fastapi import Request, status
from pydantic import TypeAdapter, ValidationError
from redis.exceptions import RedisError

from litellm._logging import verbose_proxy_logger
from litellm.caching._redis_scripts import login_block_ttls, record_login_failure
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError, RedisScriptClient
from litellm.constants import (
    EMPTY_MAPPING,
    LOGIN_THROTTLE_CACHE_KEY_PREFIX,
    LOGIN_THROTTLE_MAX_TRACKED_BLOCKS,
    LOGIN_THROTTLE_MAX_TRACKED_COUNTERS,
    LOGIN_THROTTLE_NOT_BLOCKED,
    LOGIN_THROTTLE_UNKNOWN_SOURCE,
)
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.network import TrustedProxyConfig, resolve_client_ip
from litellm.secret_managers.main import get_secret_bool

DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_SOURCE: Final = 10
DEFAULT_FAILED_LOGIN_WINDOW_SECONDS: Final = 60
DEFAULT_FAILED_LOGIN_BLOCK_SECONDS: Final = 300

IPV6_SOURCE_PREFIX_LENGTH: Final = 64
EXEMPT: Final = 0

SOURCE_LIMIT_KEY: Final = "max_failed_login_attempts_per_source"
SOURCE_LIMIT_OVERRIDES_KEY: Final = "max_failed_login_attempts_per_source_overrides"
WINDOW_KEY: Final = "failed_login_window_seconds"
BLOCK_KEY: Final = "failed_login_block_seconds"
TRUSTED_PROXY_RANGES_KEY: Final = "trusted_proxy_ranges"

_REDIS_FAILURES: Final = (RedisError, RedisCircuitBreakerOpenError, OSError, asyncio.TimeoutError)
_LOCAL_BLOCK_EXPIRY: Final = TypeAdapter[float | None](float | None)
_SOURCE_LIMIT_OVERRIDES: Final = TypeAdapter[Mapping[str, object]](Mapping[str, object])
_RANGE_ENTRIES: Final = TypeAdapter[tuple[object, ...]](tuple[object, ...])

Scope: TypeAlias = Literal["user", "source"]

_BlockTtls: TypeAlias = tuple[int, int]
_LUA_BLOCK_TTLS: Final = TypeAdapter[_BlockTtls](_BlockTtls)
_Network: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network


class LocalStore(Protocol):
    """The per-worker store behind the counters and blocks; ``InMemoryCache`` satisfies it."""

    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: float, *, ttl: int) -> None: ...

    def increment_cache(self, key: str, value: float, *, ttl: int) -> float: ...

    def delete_cache(self, key: str) -> None: ...


_COUNTERS: Final = InMemoryCache(
    max_size_in_memory=LOGIN_THROTTLE_MAX_TRACKED_COUNTERS, default_ttl=DEFAULT_FAILED_LOGIN_WINDOW_SECONDS
)
_BLOCKS: Final = InMemoryCache(
    max_size_in_memory=LOGIN_THROTTLE_MAX_TRACKED_BLOCKS, default_ttl=DEFAULT_FAILED_LOGIN_BLOCK_SECONDS
)


@cache
def _rate_limit_disabled() -> bool:
    return get_secret_bool("LITELLM_DISABLE_LOGIN_RATE_LIMIT", default_value=False) is True


@cache
def warn_login_counters_are_per_worker(num_workers: str) -> None:
    verbose_proxy_logger.warning(
        "Running %s workers but Redis is not configured. Failed Admin UI sign-in attempts are counted "
        "per worker, so the effective limits are %s times the configured values. Configure Redis "
        "to share one count across workers.",
        num_workers,
        num_workers,
    )


@cache
def warn_source_login_limit_is_off() -> None:
    verbose_proxy_logger.warning(
        "%s is not set or not a valid list of ranges, so failed Admin UI sign-in attempts are limited per "
        "source address and username only. Set it to the address ranges of the proxies in front of LiteLLM, "
        "or to an empty list when clients connect directly, to also limit each source address across usernames.",
        TRUSTED_PROXY_RANGES_KEY,
    )


def declared_proxy_ranges(settings: Mapping[str, object]) -> tuple[str, ...] | None:
    """What the operator says fronts LiteLLM: the proxy ranges, an empty tuple for none, None when unsaid.

    Only a declared topology makes the source address trustworthy enough to limit across usernames.
    An unset key, a value that is not a list of ranges, or a list with an entry that is not an address
    or range leaves it unknown and the source scope off.
    """
    entries: Final = _configured_range_entries(settings.get(TRUSTED_PROXY_RANGES_KEY))
    if entries is None or any(_parse_network(entry, TRUSTED_PROXY_RANGES_KEY) is None for entry in entries):
        return None
    return entries


def _configured_range_entries(raw_ranges: object) -> tuple[str, ...] | None:
    """Every configured entry, blanks included, so a stray empty string fails validation like any other typo."""
    if raw_ranges is None:
        return None
    if isinstance(raw_ranges, str):
        return tuple(part.strip() for part in raw_ranges.split(","))
    try:
        return tuple(str(entry).strip() for entry in _RANGE_ENTRIES.validate_python(raw_ranges))
    except ValidationError:
        verbose_proxy_logger.warning(
            "Invalid %s value: expected a list of address ranges, got %s",
            TRUSTED_PROXY_RANGES_KEY,
            type(raw_ranges).__name__,
        )
        return None


def _positive_int(raw: object, key: str, default: int) -> int:
    if raw is None:
        return default
    try:
        value: Final = int(str(raw))
    except (TypeError, ValueError):
        verbose_proxy_logger.warning("Invalid %s value %r; using %s", key, raw, default)
        return default
    if value < 1:
        verbose_proxy_logger.warning("Invalid %s value %s (must be >= 1); using %s", key, value, default)
        return default
    return value


def _int_setting(settings: Mapping[str, object], key: str, default: int) -> int:
    return _positive_int(settings.get(key), key, default)


def _override_limit(raw: object, default: int) -> int:
    """A per-address override: a limit of 1 or more, or ``EXEMPT`` (0) to leave that address unlimited."""
    if str(raw).strip() == str(EXEMPT):
        return EXEMPT
    return _positive_int(raw, SOURCE_LIMIT_OVERRIDES_KEY, default)


def _parse_address(client_ip: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address as it is limited and counted: an IPv4-mapped IPv6 address is its IPv4 address."""
    try:
        address: Final = ipaddress.ip_address(client_ip)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _parse_network(raw_range: str, setting_name: str = SOURCE_LIMIT_OVERRIDES_KEY) -> _Network | None:
    try:
        return ipaddress.ip_network(raw_range.strip(), strict=False)
    except ValueError:
        verbose_proxy_logger.warning("Invalid address or range %r in %s; skipping", raw_range, setting_name)
        return None


def _precedence(network: _Network, limit: int) -> tuple[int, bool, int]:
    """Sort key for competing overrides: the longest prefix wins, then an exemption, then the higher limit."""
    return (network.prefixlen, limit == EXEMPT, limit)


def _source_limit(settings: Mapping[str, object], client_ip: str) -> int:
    """Failure allowance for this address: the most specific configured range containing it, else the default.

    ``EXEMPT`` (0) means the operator opted this address out of both limits. Between equivalent keys such as
    ``1.2.3.4`` and ``1.2.3.4/32`` an exemption wins, then the higher limit.
    """
    default: Final = _int_setting(settings, SOURCE_LIMIT_KEY, DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_SOURCE)
    raw_overrides: Final = settings.get(SOURCE_LIMIT_OVERRIDES_KEY)
    if raw_overrides is None:
        return default
    try:
        overrides: Final = _SOURCE_LIMIT_OVERRIDES.validate_python(raw_overrides)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Invalid %s value; expected a mapping of address or range to limit", SOURCE_LIMIT_OVERRIDES_KEY
        )
        return default
    address: Final = _parse_address(client_ip)
    if address is None:
        return default
    matches: Final = sorted(
        _precedence(network, _override_limit(raw_limit, default))
        for raw_range, raw_limit in overrides.items()
        if (network := _parse_network(raw_range)) is not None and address in network
    )
    return matches[-1][-1] if matches else default


def user_limit_for(source_limit: int) -> int:
    """Failures allowed for one username from one address: half the address allowance, rounded down, at least 1."""
    return max(source_limit // 2, 1)


def source_group(client_ip: str) -> str:
    """The bucket an address is counted in: IPv4 as is, IPv6 by its /64, so one prefix holder cannot rotate."""
    address: Final = _parse_address(client_ip)
    if address is None:
        return client_ip
    if isinstance(address, ipaddress.IPv6Address):
        return str(ipaddress.ip_network((address, IPV6_SOURCE_PREFIX_LENGTH), strict=False))
    return str(address)


class _Keys(NamedTuple):
    pair_counter: str
    pair_block: str
    source_counter: str
    source_block: str


@dataclass(frozen=True, slots=True)
class Block:
    scope: Scope
    retry_after: int


@dataclass(frozen=True, slots=True)
class LoginThrottle:
    """Failed-login limits for one request's source address.

    ``source_limit`` is None when the source scope is off: ``trusted_proxy_ranges`` is unset, so the peer
    address may be a shared ingress. An empty list means clients connect directly and the peer is the source.
    ``user_limit`` is derived from the address allowance either way, see ``user_limit_for``. An address whose
    override is ``EXEMPT`` gets a disabled throttle: nothing is counted or blocked for it.
    """

    client_ip: str
    source_limit: int | None
    user_limit: int
    window_seconds: int
    block_seconds: int
    counters: LocalStore
    blocks: LocalStore
    redis_cache: RedisCache | None = None
    enabled: bool = True

    @classmethod
    def from_request(
        cls,
        request: Request,
        general_settings: Mapping[str, object] | None,
        redis_cache: RedisCache | None,
    ) -> LoginThrottle:
        settings: Final[Mapping[str, object]] = general_settings if general_settings is not None else EMPTY_MAPPING
        proxies: Final = declared_proxy_ranges(settings)
        resolved, _ = resolve_client_ip(
            request, TrustedProxyConfig(use_forwarded_for=bool(proxies), trusted_proxy_cidrs=proxies or ())
        )
        source_limit: Final = _source_limit(settings, resolved or LOGIN_THROTTLE_UNKNOWN_SOURCE)
        exempt: Final = source_limit == EXEMPT
        return cls(
            client_ip=resolved or LOGIN_THROTTLE_UNKNOWN_SOURCE,
            source_limit=source_limit if proxies is not None and resolved is not None and not exempt else None,
            user_limit=user_limit_for(source_limit),
            window_seconds=_int_setting(settings, WINDOW_KEY, DEFAULT_FAILED_LOGIN_WINDOW_SECONDS),
            block_seconds=_int_setting(settings, BLOCK_KEY, DEFAULT_FAILED_LOGIN_BLOCK_SECONDS),
            counters=_COUNTERS,
            blocks=_BLOCKS,
            redis_cache=redis_cache,
            enabled=not exempt and not _rate_limit_disabled(),
        )

    def _keys(self, username: str) -> _Keys:
        group: Final = source_group(self.client_ip)
        user: Final = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
        return _Keys(
            pair_counter=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{group}}}:user:{user}",
            pair_block=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{group}}}:block:user:{user}",
            source_counter=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{group}}}:source",
            source_block=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{group}}}:block:source",
        )

    async def attempt(self, username: str) -> LoginAttempt:
        """Refuses a blocked key before any credential is looked at; otherwise hands back the attempt to settle."""
        if not self.enabled:
            return LoginAttempt(throttle=self, username=username)
        block: Final = await self._active_block(self._keys(username))
        if block is None:
            return LoginAttempt(throttle=self, username=username)
        verbose_proxy_logger.warning(
            "Admin UI sign-in refused: the %s is blocked for %s more seconds; username=%r source=%s",
            block.scope,
            block.retry_after,
            username,
            self.client_ip,
        )
        raise self.refused(block.retry_after)

    async def _active_block(self, keys: _Keys) -> Block | None:
        local: Final = self._local_block_ttls(keys)
        shared: Final = await self._shared_block_ttls(keys)
        user_ttl: Final = max(local[0], shared[0])
        source_ttl: Final = max(local[1], shared[1])
        if self.source_limit is not None and source_ttl > 0:
            return Block(scope="source", retry_after=source_ttl)
        if user_ttl > 0:
            return Block(scope="user", retry_after=user_ttl)
        return None

    async def _shared_block_ttls(self, keys: _Keys) -> _BlockTtls:
        if self.redis_cache is None:
            return LOGIN_THROTTLE_NOT_BLOCKED
        try:
            return _LUA_BLOCK_TTLS.validate_python(await login_block_ttls(RedisScriptClient(self.redis_cache), *keys))
        except _REDIS_FAILURES as err:
            self._warn_redis(err)
            return LOGIN_THROTTLE_NOT_BLOCKED

    def _local_block_ttls(self, keys: _Keys) -> _BlockTtls:
        return self._local_block_ttl(keys.pair_block), self._local_block_ttl(keys.source_block)

    def _local_block_ttl(self, block_key: str) -> int:
        expires_at: Final = _LOCAL_BLOCK_EXPIRY.validate_python(self.blocks.get_cache(block_key))
        if expires_at is None:
            return 0
        return max(math.ceil(expires_at - time.time()), 0)

    async def record_failure(self, username: str) -> _BlockTtls:
        keys: Final = self._keys(username)
        source_limit: Final = self.source_limit or 0
        if self.redis_cache is not None:
            try:
                return _LUA_BLOCK_TTLS.validate_python(
                    await record_login_failure(
                        RedisScriptClient(self.redis_cache),
                        *keys,
                        pair_limit=self.user_limit,
                        source_limit=source_limit,
                        window_seconds=self.window_seconds,
                        block_seconds=self.block_seconds,
                    )
                )
            except _REDIS_FAILURES as err:
                self._warn_redis(err)
        user_block: Final = self._local_bump(keys.pair_counter, keys.pair_block, self.user_limit)
        if source_limit == 0 or user_block > 0:
            return user_block, 0
        return user_block, self._local_bump(keys.source_counter, keys.source_block, source_limit)

    def _local_bump(self, count_key: str, block_key: str, limit: int) -> int:
        blocked: Final = self._local_block_ttl(block_key)
        if blocked > 0:
            return blocked
        count: Final = int(self.counters.increment_cache(count_key, 1, ttl=self.window_seconds))
        if count <= limit:
            return 0
        self.blocks.set_cache(block_key, time.time() + self.block_seconds, ttl=self.block_seconds)
        return self.block_seconds

    async def clear_pair(self, username: str) -> None:
        pair_counter: Final = self._keys(username).pair_counter
        if self.redis_cache is not None:
            try:
                await self.redis_cache.async_delete_cache(pair_counter)
            except _REDIS_FAILURES as err:
                self._warn_redis(err)
        self.counters.delete_cache(pair_counter)

    def _warn_redis(self, err: Exception) -> None:
        verbose_proxy_logger.warning(
            "Redis failed while counting Admin UI sign-in attempts; using this worker's own counters "
            "until it recovers: %s",
            err,
        )

    @staticmethod
    def refused(retry_after: int) -> ProxyException:
        return ProxyException(
            message="Too many failed sign-in attempts. Try again later.",
            type=ProxyErrorTypes.auth_error,
            param="username",
            code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},  # mutable-ok: ProxyException writes into its headers dict
        )


@dataclass(frozen=True, slots=True)
class LoginAttempt:
    throttle: LoginThrottle
    username: str

    async def succeeded(self) -> None:
        if not self.throttle.enabled:
            return
        await self.throttle.clear_pair(self.username)

    async def failed(self) -> None:
        if not self.throttle.enabled:
            return
        user_block, source_block = await self.throttle.record_failure(self.username)
        if user_block == 0 and source_block == 0:
            return
        verbose_proxy_logger.warning(
            "Admin UI sign-in blocked for %s seconds after too many failures; scope=%s username=%r source=%s",
            user_block or source_block,
            "user" if user_block else "source",
            self.username,
            self.throttle.client_ip,
        )
