import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, NamedTuple, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.constants import LOGIN_THROTTLE_CACHE_KEY_PREFIX
from litellm.redis.transport.contract import RedisFailure, RedisTransport

BlockTtls: TypeAlias = tuple[int, int]

_BLOCK_TTLS: Final = TypeAdapter[BlockTtls](BlockTtls)

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


class LoginKeys(NamedTuple):
    pair_counter: str
    pair_block: str
    source_counter: str
    source_block: str


def login_keys(source_group: str, username: str) -> LoginKeys:
    user: Final = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
    return LoginKeys(
        pair_counter=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{source_group}}}:user:{user}",
        pair_block=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{source_group}}}:block:user:{user}",
        source_counter=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{source_group}}}:source",
        source_block=f"{LOGIN_THROTTLE_CACHE_KEY_PREFIX}:{{{source_group}}}:block:source",
    )


class LoginThrottleStore(Protocol):
    async def blocks(self, source_group: str, username: str) -> BlockTtls | RedisFailure: ...

    async def record_failure(
        self,
        source_group: str,
        username: str,
        *,
        user_limit: int,
        source_limit: int,
        window: timedelta,
        block: timedelta,
    ) -> BlockTtls | RedisFailure: ...

    async def clear(self, source_group: str, username: str) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportLoginThrottleStore:
    transport: RedisTransport

    async def blocks(self, source_group: str, username: str) -> BlockTtls | RedisFailure:
        return _ttls(await self.transport.eval(_BLOCK_TTLS_LUA, login_keys(source_group, username), ()))

    async def record_failure(
        self,
        source_group: str,
        username: str,
        *,
        user_limit: int,
        source_limit: int,
        window: timedelta,
        block: timedelta,
    ) -> BlockTtls | RedisFailure:
        args: Final = (user_limit, source_limit, _whole_seconds(window), _whole_seconds(block))
        return _ttls(await self.transport.eval(_RECORD_FAILURE_LUA, login_keys(source_group, username), args))

    async def clear(self, source_group: str, username: str) -> None | RedisFailure:
        deleted: Final = await self.transport.delete(login_keys(source_group, username).pair_counter)
        return deleted if isinstance(deleted, RedisFailure) else None


def _whole_seconds(ttl: timedelta) -> int:
    return ttl // timedelta(seconds=1)


def _ttls(reply: object) -> BlockTtls | RedisFailure:
    if isinstance(reply, RedisFailure):
        return reply
    try:
        return _BLOCK_TTLS.validate_python(reply, strict=True)
    except ValidationError as e:
        return RedisFailure(kind="decode", message=f"login throttle reply is not two block TTLs: {e}")
