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
from typing import Final, Literal, NamedTuple, NoReturn, Protocol, TypeAlias

from fastapi import Request, status
from pydantic import TypeAdapter, ValidationError
from redis.exceptions import RedisError

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.constants import (
    EMPTY_MAPPING,
    LOGIN_THROTTLE_CACHE_KEY_PREFIX,
    LOGIN_THROTTLE_MAX_TRACKED_BLOCKS,
    LOGIN_THROTTLE_MAX_TRACKED_COUNTERS,
    LOGIN_THROTTLE_NOT_BLOCKED,
    LOGIN_THROTTLE_UNKNOWN_SOURCE,
)
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.network import TrustedProxyConfig, normalize_cidr_ranges, resolve_client_ip
from litellm.secret_managers.main import get_secret_bool

DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_SOURCE: Final = 10
DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_USER: Final = 5
DEFAULT_FAILED_LOGIN_WINDOW_SECONDS: Final = 60
DEFAULT_FAILED_LOGIN_BLOCK_SECONDS: Final = 300

IPV6_SOURCE_PREFIX_LENGTH: Final = 64

SOURCE_LIMIT_KEY: Final = "max_failed_login_attempts_per_source"
SOURCE_LIMIT_OVERRIDES_KEY: Final = "max_failed_login_attempts_per_source_overrides"
USER_LIMIT_KEY: Final = "max_failed_login_attempts_per_user"
WINDOW_KEY: Final = "failed_login_window_seconds"
BLOCK_KEY: Final = "failed_login_block_seconds"
TRUSTED_PROXY_RANGES_KEY: Final = "trusted_proxy_ranges"

_REDIS_FAILURES: Final = (RedisError, RedisCircuitBreakerOpenError, OSError, asyncio.TimeoutError)
_LOCAL_BLOCK_EXPIRY: Final = TypeAdapter[float | None](float | None)
_SOURCE_LIMIT_OVERRIDES: Final = TypeAdapter[Mapping[str, object]](Mapping[str, object])

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


# KEYS: pair counter, pair block, source counter, source block (one cluster slot via the source hash tag)
# ARGV: pair limit, source limit (0 = source scope off), window seconds, block seconds
# Both scripts return {pair block TTL, source block TTL}; 0 or below means not blocked
_BLOCK_TTLS_LUA: Final = "return {redis.call('TTL', KEYS[2]), redis.call('TTL', KEYS[4])}"
_RECORD_FAILURE_LUA: Final = (
    "local function bump(count_key, block_key, limit) "
    "local blocked = redis.call('TTL', block_key) "
    "if blocked > 0 then return blocked end "
    "local count = redis.call('INCR', count_key) "
    "if redis.call('TTL', count_key) < 0 then redis.call('EXPIRE', count_key, ARGV[3]) end "
    "if count > limit then redis.call('SET', block_key, '1', 'EX', ARGV[4]) return tonumber(ARGV[4]) end "
    "return 0 end "
    "local user_block = bump(KEYS[1], KEYS[2], tonumber(ARGV[1])) "
    "local source_block = 0 "
    "if tonumber(ARGV[2]) > 0 and user_block == 0 then "
    "source_block = bump(KEYS[3], KEYS[4], tonumber(ARGV[2])) end "
    "return {user_block, source_block}"
)

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
        "%s is not set, so failed Admin UI sign-in attempts are limited per source address and username "
        "only. Set it to the address ranges of the proxies in front of LiteLLM, or to an empty list when "
        "clients connect directly, to also limit each source address across usernames.",
        TRUSTED_PROXY_RANGES_KEY,
    )


def declared_proxy_ranges(settings: Mapping[str, object]) -> tuple[str, ...] | None:
    """What the operator says fronts LiteLLM: the proxy ranges, an empty tuple for none, None when unsaid.

    Only a declared topology makes the source address trustworthy enough to limit across usernames.
    An unset key, or a value that is not a list of ranges, leaves it unknown and the source scope off.
    """
    raw_ranges: Final = settings.get(TRUSTED_PROXY_RANGES_KEY)
    if isinstance(raw_ranges, (list, tuple, set)) and not raw_ranges:
        return ()
    cidrs: Final = tuple(normalize_cidr_ranges(raw_ranges, setting_name=TRUSTED_PROXY_RANGES_KEY))
    return cidrs or None


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


def _parse_address(client_ip: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The address as it is limited and counted: an IPv4-mapped IPv6 address is its IPv4 address."""
    try:
        address: Final = ipaddress.ip_address(client_ip)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _parse_network(raw_range: str) -> _Network | None:
    try:
        return ipaddress.ip_network(raw_range.strip(), strict=False)
    except ValueError:
        verbose_proxy_logger.warning(
            "Invalid address or range %r in %s; skipping", raw_range, SOURCE_LIMIT_OVERRIDES_KEY
        )
        return None


def _source_limit(settings: Mapping[str, object], client_ip: str) -> int:
    """Failure allowance for this address: the most specific configured range containing it, else the default."""
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
        (network.prefixlen, _positive_int(raw_limit, SOURCE_LIMIT_OVERRIDES_KEY, default))
        for raw_range, raw_limit in overrides.items()
        if (network := _parse_network(raw_range)) is not None and address in network
    )
    return matches[-1][1] if matches else default


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
        settings: Final = general_settings if general_settings is not None else EMPTY_MAPPING
        proxies: Final = declared_proxy_ranges(settings)
        resolved, _ = resolve_client_ip(
            request, TrustedProxyConfig(use_forwarded_for=bool(proxies), trusted_proxy_cidrs=proxies or ())
        )
        return cls(
            client_ip=resolved or LOGIN_THROTTLE_UNKNOWN_SOURCE,
            source_limit=_source_limit(settings, resolved) if proxies is not None and resolved is not None else None,
            user_limit=_int_setting(settings, USER_LIMIT_KEY, DEFAULT_MAX_FAILED_LOGIN_ATTEMPTS_PER_USER),
            window_seconds=_int_setting(settings, WINDOW_KEY, DEFAULT_FAILED_LOGIN_WINDOW_SECONDS),
            block_seconds=_int_setting(settings, BLOCK_KEY, DEFAULT_FAILED_LOGIN_BLOCK_SECONDS),
            counters=_COUNTERS,
            blocks=_BLOCKS,
            redis_cache=redis_cache,
            enabled=not _rate_limit_disabled(),
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
        self.refuse(block.retry_after)

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
            return _LUA_BLOCK_TTLS.validate_python(
                await self.redis_cache.async_register_script(_BLOCK_TTLS_LUA)(keys, ())
            )
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
                    await self.redis_cache.async_register_script(_RECORD_FAILURE_LUA)(
                        keys, (self.user_limit, source_limit, self.window_seconds, self.block_seconds)
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
    def refuse(retry_after: int) -> NoReturn:
        raise ProxyException(
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
