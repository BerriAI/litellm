"""Redis Lua scripts compiled by redis-lua-py from redis_scripts.scripts. Do not edit.

Each script is a function taking a redis-py client, sync or async, and then the
arguments it was written with. Its Lua is the constant of the same name in
capitals.

Regenerate with:

    python -m redis_lua_py generate redis_scripts.scripts --out litellm/caching/_redis_scripts.py
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from typing import Any, Protocol, Union, overload
from weakref import WeakKeyDictionary

__all__ = [
    "BATCH_RATE_LIMIT",
    "CHECK_AND_INCREMENT_BY_N",
    "CLAIM_AFFINITY_PIN",
    "DELETE_IF_OWNER",
    "INCREMENT_SESSION_ITERATIONS",
    "INCREMENT_SESSION_SPEND",
    "INCREMENT_TOKENS",
    "INCREMENT_WITH_FLOOR",
    "LOGIN_BLOCK_TTLS",
    "PARALLEL_ACQUIRE",
    "PARALLEL_COUNT",
    "PARALLEL_RELEASE",
    "PEXPIRE_IF_OWNER",
    "POP_RESERVATION",
    "RECORD_LOGIN_FAILURE",
    "REFUND_ENQUEUED_TOKENS",
    "RESERVE_ENQUEUED_TOKENS",
    "SAVE_RESERVATION",
    "SET_MAX",
    "WINDOW_GUARDED_TOKEN_INCREMENT",
    "batch_rate_limit",
    "check_and_increment_by_n",
    "claim_affinity_pin",
    "delete_if_owner",
    "increment_session_iterations",
    "increment_session_spend",
    "increment_tokens",
    "increment_with_floor",
    "login_block_ttls",
    "parallel_acquire",
    "parallel_count",
    "parallel_release",
    "pexpire_if_owner",
    "pop_reservation",
    "record_login_failure",
    "refund_enqueued_tokens",
    "reserve_enqueued_tokens",
    "save_reservation",
    "set_max",
    "window_guarded_token_increment",
]


#: What a generated signature accepts for a key, the same as redis-py does.
_Key = Union[str, bytes, memoryview]

#: What a generated signature accepts for an argument it has no better type for.
_Arg = Union[str, bytes, memoryview, int, float]


class _SyncClient(Protocol):
    """Enough of a sync redis-py client to type its call before the async one.

    ``__enter__`` is the discriminator because every sync redis-py client has
    one and no async client does. Checked first, it keeps a wrapper that
    forwards everything through ``__getattr__`` -- which mypy takes to supply
    ``__aenter__`` as well -- from being typed as async.
    """

    def __enter__(self) -> Any: ...

    def register_script(self, script: str) -> Any: ...


class _AsyncClient(Protocol):
    """Enough of an async redis-py client to tell it from a sync one.

    Only used to type the two shapes of call. ``__aenter__`` is the
    discriminator because every async client has one and no sync client does,
    on every supported redis-py -- ``aclose`` only arrived in redis-py 5.
    """

    async def __aenter__(self) -> Any: ...

    def register_script(self, script: str) -> Any: ...


def _encode(
    name: str, value: object, error: type[Exception] = TypeError
) -> str | bytes | memoryview:
    """Render a Python value as a Redis argument.

    Redis has no argument types: everything on the wire is a byte string. This
    only accepts values whose string form is unambiguous, so that a stray None
    or object fails here rather than arriving in Lua as something surprising.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, bytes, memoryview)):
        return value
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    raise error(
        f"argument {name!r} is a {type(value).__name__}, which has no Redis representation. "
        "Pass a str, bytes, int, float or bool."
    )


def _items(name: str, value: object, error: type[Exception] = TypeError) -> list[Any]:
    """The elements passed for a list parameter.

    A string is iterable too, and splitting a key into characters is never
    what was meant, so it is refused rather than spread.
    """
    if isinstance(value, (str, bytes, memoryview)) or not isinstance(value, Iterable):
        raise error(
            f"argument {name!r} takes a list, got a {type(value).__name__}. "
            "Wrap a single value in a list."
        )
    return list(value)


def _encode_all(
    name: str, value: object, error: type[Exception] = TypeError
) -> list[str | bytes | memoryview]:
    """The elements passed for a list of arguments, each rendered for Redis."""
    return [_encode(name, item, error) for item in _items(name, value, error)]


def _is_cluster_pipeline(client: object) -> bool:
    cls = type(client)
    return cls.__name__ == "ClusterPipeline" and cls.__module__.startswith("redis.")


def _registered(lua: str, registry: WeakKeyDictionary[Any, Any], client: Any) -> Any:
    """The redis-py Script for this source on this client, registered once.

    redis-py's own Script object already implements the EVALSHA-then-EVAL
    dance and the NOSCRIPT retry, so this defers to it rather than
    reimplementing script caching.
    """
    try:
        registered = registry.get(client)
    except TypeError:  # a client that does not support weak references
        return client.register_script(lua)
    if registered is None:
        registered = client.register_script(lua)
        registry[client] = registered
    return registered


def _run(
    lua: str,
    registry: WeakKeyDictionary[Any, Any],
    client: Any,
    keys: list[Any],
    argv: list[Any],
) -> Any:
    """Run a script: a value from a sync client, an awaitable from an async one."""
    if _is_cluster_pipeline(client):
        # redis-py refuses EVALSHA on a cluster pipeline, and a queued
        # EVALSHA could not recover from NOSCRIPT at execute time anyway,
        # so the source travels with the command.
        return client.eval(lua, len(keys), *keys, *argv)
    return _registered(lua, registry, client)(keys=keys, args=argv, client=client)


# increment_with_floor -- KEYS: key; ARGV: amount, ttl
INCREMENT_WITH_FLOOR = """\
-- increment_with_floor
-- Generated by redis-lua-py from redis_scripts/scripts.py:22. Do not edit.
local key = KEYS[1]
local amount = ARGV[1]
local ttl = ARGV[2]
local count = redis.call('INCRBY', key, amount)
if count < 0 then
  count = redis.call('INCRBY', key, -count)
end
if redis.call('TTL', key) < 0 then
  redis.call('EXPIRE', key, ttl)
end
return count
"""
_INCREMENT_WITH_FLOOR_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def increment_with_floor(
    client: _SyncClient,
    /,
    key: _Key,
    amount: int,
    ttl: int,
) -> int: ...
@overload
def increment_with_floor(
    client: _AsyncClient,
    /,
    key: _Key,
    amount: int,
    ttl: int,
) -> Awaitable[int]: ...
@overload
def increment_with_floor(client: Any, /, key: _Key, amount: int, ttl: int) -> int: ...
def increment_with_floor(client: Any, /, key: _Key, amount: int, ttl: int) -> Any:
    """Add ``amount`` to ``key``, clamp the result at zero, and give a key without one ``ttl``."""
    return _run(
        INCREMENT_WITH_FLOOR,
        _INCREMENT_WITH_FLOOR_CLIENTS,
        client,
        [key],
        [_encode("amount", amount), _encode("ttl", ttl)],
    )


# set_max -- KEYS: key; ARGV: value, ttl
SET_MAX = """\
-- set_max
-- Generated by redis-lua-py from redis_scripts/scripts.py:33. Do not edit.
-- A Redis command with nothing to return hands Lua false, not nil, so an
-- `is None` test has to accept both. Taking v as an argument also means the
-- operand is evaluated once, not once per comparison.
local function __isnil(v)
  return v == nil or v == false
end
local key = KEYS[1]
local value = ARGV[1]
local ttl = tonumber(ARGV[2])
local current = redis.call('GET', key)
if __isnil(current) or tonumber(current) < tonumber(value) then
  redis.call('SET', key, value)
  if ttl > 0 then
    redis.call('EXPIRE', key, ttl)
  end
  return value
end
return current
"""
_SET_MAX_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def set_max(client: _SyncClient, /, key: _Key, value: str, ttl: int) -> bytes: ...
@overload
def set_max(
    client: _AsyncClient,
    /,
    key: _Key,
    value: str,
    ttl: int,
) -> Awaitable[bytes]: ...
@overload
def set_max(client: Any, /, key: _Key, value: str, ttl: int) -> bytes: ...
def set_max(client: Any, /, key: _Key, value: str, ttl: int) -> Any:
    """Store ``value`` unless ``key`` already holds a number at least as large; return what is stored.

    A ``ttl`` of 0 leaves the key's expiry alone.
    """
    return _run(
        SET_MAX,
        _SET_MAX_CLIENTS,
        client,
        [key],
        [_encode("value", value), _encode("ttl", ttl)],
    )


# claim_affinity_pin -- KEYS: key; ARGV: pin, ttl, eligible_json
CLAIM_AFFINITY_PIN = """\
-- claim_affinity_pin
-- Generated by redis-lua-py from redis_scripts/scripts.py:51. Do not edit.
-- Python truthiness: 0, '', empty tables and nil are all false.
local function __truthy(v)
  if v == nil or v == false then return false end
  if v == 0 or v == '' then return false end
  if type(v) == 'table' and next(v) == nil then return false end
  return true
end
-- A Redis command with nothing to return hands Lua false, not nil, so an
-- `is None` test has to accept both. Taking v as an argument also means the
-- operand is evaluated once, not once per comparison.
local function __isnil(v)
  return v == nil or v == false
end
-- dict.get: the value under a key, or the default when there is none.
local function __get(t, k, default)
  local v = t[k]
  if v == nil then return default end
  return v
end
local key = KEYS[1]
local pin = ARGV[1]
local ttl = ARGV[2]
local eligible_json = ARGV[3]
local matched, matches, stored
local current = redis.call('GET', key)
if __isnil(current) then
  redis.call('SET', key, pin, 'EX', ttl)
  return pin
end
if __truthy(eligible_json) then
  matched = false
  local function __try1()
    stored = cjson.decode(current)
    local __seq2 = cjson.decode(eligible_json)
    for __i2 = 1, #__seq2 do
      local eligible = __seq2[__i2]
      matches = true
      for name, value in pairs(eligible) do
        if __get(stored, name, nil) ~= value then
          matches = false
          do break end
        end
      end
      for name in pairs(stored) do
        if __isnil(__get(eligible, name, nil)) then
          matches = false
          do break end
        end
      end
      if matches then
        matched = true
        do break end
      end
    end
  end
  local __ok1, __r1 = pcall(__try1)
  if not __ok1 then
    matched = false
  end
  if matched then
    redis.call('EXPIRE', key, ttl)
    return current
  end
  redis.call('SET', key, pin, 'EX', ttl)
  return pin
end
if current == pin then
  redis.call('EXPIRE', key, ttl)
end
return current
"""
_CLAIM_AFFINITY_PIN_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def claim_affinity_pin(
    client: _SyncClient,
    /,
    key: _Key,
    pin: str,
    ttl: int,
    eligible_json: str,
) -> bytes: ...
@overload
def claim_affinity_pin(
    client: _AsyncClient,
    /,
    key: _Key,
    pin: str,
    ttl: int,
    eligible_json: str,
) -> Awaitable[bytes]: ...
@overload
def claim_affinity_pin(
    client: Any,
    /,
    key: _Key,
    pin: str,
    ttl: int,
    eligible_json: str,
) -> bytes: ...
def claim_affinity_pin(
    client: Any,
    /,
    key: _Key,
    pin: str,
    ttl: int,
    eligible_json: str,
) -> Any:
    """Return the pin stored at ``key``, claiming it for ``pin`` when there is none.

    With ``eligible_json`` empty, a stored pin always wins and is refreshed only when it
    equals ``pin``. Otherwise it is a JSON list of objects, and a stored pin that is not
    exactly one of them is replaced by ``pin``.
    """
    return _run(
        CLAIM_AFFINITY_PIN,
        _CLAIM_AFFINITY_PIN_CLIENTS,
        client,
        [key],
        [
            _encode("pin", pin),
            _encode("ttl", ttl),
            _encode("eligible_json", eligible_json),
        ],
    )


# login_block_ttls -- KEYS: pair_counter, pair_block, source_counter, source_block; ARGV: none
LOGIN_BLOCK_TTLS = """\
-- login_block_ttls
-- Generated by redis-lua-py from redis_scripts/scripts.py:100. Do not edit.
local pair_counter = KEYS[1]
local pair_block = KEYS[2]
local source_counter = KEYS[3]
local source_block = KEYS[4]
return {redis.call('TTL', pair_block), redis.call('TTL', source_block)}
"""
_LOGIN_BLOCK_TTLS_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def login_block_ttls(
    client: _SyncClient,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
) -> list[int]: ...
@overload
def login_block_ttls(
    client: _AsyncClient,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
) -> Awaitable[list[int]]: ...
@overload
def login_block_ttls(
    client: Any,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
) -> list[int]: ...
def login_block_ttls(
    client: Any,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
) -> Any:
    return _run(
        LOGIN_BLOCK_TTLS,
        _LOGIN_BLOCK_TTLS_CLIENTS,
        client,
        [pair_counter, pair_block, source_counter, source_block],
        [],
    )


# record_login_failure -- KEYS: pair_counter, pair_block, source_counter, source_block; ARGV: pair_limit, source_limit, window_seconds, block_seconds
RECORD_LOGIN_FAILURE = """\
-- record_login_failure
-- Generated by redis-lua-py from redis_scripts/scripts.py:105. Do not edit.
local pair_counter = KEYS[1]
local pair_block = KEYS[2]
local source_counter = KEYS[3]
local source_block = KEYS[4]
local pair_limit = tonumber(ARGV[1])
local source_limit = tonumber(ARGV[2])
local window_seconds = ARGV[3]
local block_seconds = tonumber(ARGV[4])
local function bump(count_key, block_key, limit)
  local blocked = redis.call('TTL', block_key)
  if blocked > 0 then
    return blocked
  end
  local count = redis.call('INCR', count_key)
  if redis.call('TTL', count_key) < 0 then
    redis.call('EXPIRE', count_key, window_seconds)
  end
  if count > limit then
    redis.call('SET', block_key, '1', 'EX', block_seconds)
    return block_seconds
  end
  return 0
end
local pair_block_ttl = bump(pair_counter, pair_block, pair_limit)
local source_block_ttl = 0
if source_limit > 0 and pair_block_ttl == 0 then
  source_block_ttl = bump(source_counter, source_block, source_limit)
end
return {pair_block_ttl, source_block_ttl}
"""
_RECORD_LOGIN_FAILURE_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def record_login_failure(
    client: _SyncClient,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
    pair_limit: int,
    source_limit: int,
    window_seconds: int,
    block_seconds: int,
) -> list[int]: ...
@overload
def record_login_failure(
    client: _AsyncClient,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
    pair_limit: int,
    source_limit: int,
    window_seconds: int,
    block_seconds: int,
) -> Awaitable[list[int]]: ...
@overload
def record_login_failure(
    client: Any,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
    pair_limit: int,
    source_limit: int,
    window_seconds: int,
    block_seconds: int,
) -> list[int]: ...
def record_login_failure(
    client: Any,
    /,
    pair_counter: _Key,
    pair_block: _Key,
    source_counter: _Key,
    source_block: _Key,
    pair_limit: int,
    source_limit: int,
    window_seconds: int,
    block_seconds: int,
) -> Any:
    """Count one failed login, blocking a scope that goes over its limit.

    A ``source_limit`` of 0 turns the source scope off, and a blocked pair stops
    counting against its source.
    """
    return _run(
        RECORD_LOGIN_FAILURE,
        _RECORD_LOGIN_FAILURE_CLIENTS,
        client,
        [pair_counter, pair_block, source_counter, source_block],
        [
            _encode("pair_limit", pair_limit),
            _encode("source_limit", source_limit),
            _encode("window_seconds", window_seconds),
            _encode("block_seconds", block_seconds),
        ],
    )


# reserve_enqueued_tokens -- KEYS: key; ARGV: amount, ttl, limit
RESERVE_ENQUEUED_TOKENS = """\
-- reserve_enqueued_tokens
-- Generated by redis-lua-py from redis_scripts/scripts.py:144. Do not edit.
-- Python truthiness: 0, '', empty tables and nil are all false.
local function __truthy(v)
  if v == nil or v == false then return false end
  if v == 0 or v == '' then return false end
  if type(v) == 'table' and next(v) == nil then return false end
  return true
end
-- Python's `a or b`: a when it is truthy by Python's rules, otherwise b.
local function __or(a, b)
  if __truthy(a) then return a end
  return b
end
local key = KEYS[1]
local amount = tonumber(ARGV[1])
local ttl = ARGV[2]
local limit = tonumber(ARGV[3])
local current = tonumber(__or(redis.call('GET', key), '0'))
if current + amount > limit then
  return {0, current}
end
local updated = redis.call('INCRBY', key, amount)
redis.call('EXPIRE', key, ttl)
return {1, updated}
"""
_RESERVE_ENQUEUED_TOKENS_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def reserve_enqueued_tokens(
    client: _SyncClient,
    /,
    key: _Key,
    amount: int,
    ttl: int,
    limit: int,
) -> list[int]: ...
@overload
def reserve_enqueued_tokens(
    client: _AsyncClient,
    /,
    key: _Key,
    amount: int,
    ttl: int,
    limit: int,
) -> Awaitable[list[int]]: ...
@overload
def reserve_enqueued_tokens(
    client: Any,
    /,
    key: _Key,
    amount: int,
    ttl: int,
    limit: int,
) -> list[int]: ...
def reserve_enqueued_tokens(
    client: Any,
    /,
    key: _Key,
    amount: int,
    ttl: int,
    limit: int,
) -> Any:
    """Add ``amount`` unless that takes the counter over ``limit``.

    Returns [1, new total] when reserved and [0, current total] when refused.
    """
    return _run(
        RESERVE_ENQUEUED_TOKENS,
        _RESERVE_ENQUEUED_TOKENS_CLIENTS,
        client,
        [key],
        [_encode("amount", amount), _encode("ttl", ttl), _encode("limit", limit)],
    )


# refund_enqueued_tokens -- KEYS: key; ARGV: amount
REFUND_ENQUEUED_TOKENS = """\
-- refund_enqueued_tokens
-- Generated by redis-lua-py from redis_scripts/scripts.py:158. Do not edit.
local key = KEYS[1]
local amount = ARGV[1]
if redis.call('DECRBY', key, amount) <= 0 then
  redis.call('DEL', key)
end
return 1
"""
_REFUND_ENQUEUED_TOKENS_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def refund_enqueued_tokens(client: _SyncClient, /, key: _Key, amount: int) -> int: ...
@overload
def refund_enqueued_tokens(
    client: _AsyncClient,
    /,
    key: _Key,
    amount: int,
) -> Awaitable[int]: ...
@overload
def refund_enqueued_tokens(client: Any, /, key: _Key, amount: int) -> int: ...
def refund_enqueued_tokens(client: Any, /, key: _Key, amount: int) -> Any:
    """Take ``amount`` back off the counter, deleting it once nothing is reserved."""
    return _run(
        REFUND_ENQUEUED_TOKENS,
        _REFUND_ENQUEUED_TOKENS_CLIENTS,
        client,
        [key],
        [_encode("amount", amount)],
    )


# save_reservation -- KEYS: key; ARGV: record, ttl
SAVE_RESERVATION = """\
-- save_reservation
-- Generated by redis-lua-py from redis_scripts/scripts.py:166. Do not edit.
local key = KEYS[1]
local record = ARGV[1]
local ttl = ARGV[2]
redis.call('SET', key, record, 'EX', ttl)
return 1
"""
_SAVE_RESERVATION_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def save_reservation(
    client: _SyncClient,
    /,
    key: _Key,
    record: str,
    ttl: int,
) -> int: ...
@overload
def save_reservation(
    client: _AsyncClient,
    /,
    key: _Key,
    record: str,
    ttl: int,
) -> Awaitable[int]: ...
@overload
def save_reservation(client: Any, /, key: _Key, record: str, ttl: int) -> int: ...
def save_reservation(client: Any, /, key: _Key, record: str, ttl: int) -> Any:
    return _run(
        SAVE_RESERVATION,
        _SAVE_RESERVATION_CLIENTS,
        client,
        [key],
        [_encode("record", record), _encode("ttl", ttl)],
    )


# pop_reservation -- KEYS: key; ARGV: ttl
POP_RESERVATION = """\
-- pop_reservation
-- Generated by redis-lua-py from redis_scripts/scripts.py:172. Do not edit.
-- Python truthiness: 0, '', empty tables and nil are all false.
local function __truthy(v)
  if v == nil or v == false then return false end
  if v == 0 or v == '' then return false end
  if type(v) == 'table' and next(v) == nil then return false end
  return true
end
local key = KEYS[1]
local ttl = ARGV[1]
local value = redis.call('GET', key)
if __truthy(value) then
  redis.call('SET', key, '', 'EX', ttl)
end
return value
"""
_POP_RESERVATION_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def pop_reservation(client: _SyncClient, /, key: _Key, ttl: int) -> bytes | None: ...
@overload
def pop_reservation(
    client: _AsyncClient,
    /,
    key: _Key,
    ttl: int,
) -> Awaitable[bytes | None]: ...
@overload
def pop_reservation(client: Any, /, key: _Key, ttl: int) -> bytes | None: ...
def pop_reservation(client: Any, /, key: _Key, ttl: int) -> Any:
    """Return the saved reservation and leave an empty marker for ``ttl``, so it pops once."""
    return _run(
        POP_RESERVATION,
        _POP_RESERVATION_CLIENTS,
        client,
        [key],
        [_encode("ttl", ttl)],
    )


# increment_session_iterations -- KEYS: key; ARGV: ttl
INCREMENT_SESSION_ITERATIONS = """\
-- increment_session_iterations
-- Generated by redis-lua-py from redis_scripts/scripts.py:184. Do not edit.
local key = KEYS[1]
local ttl = ARGV[1]
local current = redis.call('INCR', key)
if current == 1 then
  redis.call('EXPIRE', key, ttl)
end
return current
"""
_INCREMENT_SESSION_ITERATIONS_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def increment_session_iterations(
    client: _SyncClient,
    /,
    key: _Key,
    ttl: int,
) -> int: ...
@overload
def increment_session_iterations(
    client: _AsyncClient,
    /,
    key: _Key,
    ttl: int,
) -> Awaitable[int]: ...
@overload
def increment_session_iterations(client: Any, /, key: _Key, ttl: int) -> int: ...
def increment_session_iterations(client: Any, /, key: _Key, ttl: int) -> Any:
    """Count one more iteration, starting the ``ttl`` on the first."""
    return _run(
        INCREMENT_SESSION_ITERATIONS,
        _INCREMENT_SESSION_ITERATIONS_CLIENTS,
        client,
        [key],
        [_encode("ttl", ttl)],
    )


# increment_session_spend -- KEYS: key; ARGV: amount, ttl
INCREMENT_SESSION_SPEND = """\
-- increment_session_spend
-- Generated by redis-lua-py from redis_scripts/scripts.py:196. Do not edit.
local key = KEYS[1]
local amount = ARGV[1]
local ttl = ARGV[2]
local existed = redis.call('EXISTS', key)
local new_value = redis.call('INCRBYFLOAT', key, amount)
if existed == 0 then
  redis.call('EXPIRE', key, ttl)
end
return new_value
"""
_INCREMENT_SESSION_SPEND_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def increment_session_spend(
    client: _SyncClient,
    /,
    key: _Key,
    amount: str,
    ttl: int,
) -> bytes: ...
@overload
def increment_session_spend(
    client: _AsyncClient,
    /,
    key: _Key,
    amount: str,
    ttl: int,
) -> Awaitable[bytes]: ...
@overload
def increment_session_spend(
    client: Any,
    /,
    key: _Key,
    amount: str,
    ttl: int,
) -> bytes: ...
def increment_session_spend(client: Any, /, key: _Key, amount: str, ttl: int) -> Any:
    """Add ``amount`` to the session's spend, starting the ``ttl`` when the key is new."""
    return _run(
        INCREMENT_SESSION_SPEND,
        _INCREMENT_SESSION_SPEND_CLIENTS,
        client,
        [key],
        [_encode("amount", amount), _encode("ttl", ttl)],
    )


# delete_if_owner -- KEYS: key; ARGV: token
DELETE_IF_OWNER = """\
-- delete_if_owner
-- Generated by redis-lua-py from redis_scripts/scripts.py:210. Do not edit.
local key = KEYS[1]
local token = ARGV[1]
if redis.call('GET', key) == token then
  return redis.call('DEL', key)
end
return 0
"""
_DELETE_IF_OWNER_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def delete_if_owner(client: _SyncClient, /, key: _Key, token: str) -> int: ...
@overload
def delete_if_owner(
    client: _AsyncClient,
    /,
    key: _Key,
    token: str,
) -> Awaitable[int]: ...
@overload
def delete_if_owner(client: Any, /, key: _Key, token: str) -> int: ...
def delete_if_owner(client: Any, /, key: _Key, token: str) -> Any:
    """Delete the lock only while it still holds ``token``, so an expired holder cannot free the next one's."""
    return _run(
        DELETE_IF_OWNER,
        _DELETE_IF_OWNER_CLIENTS,
        client,
        [key],
        [_encode("token", token)],
    )


# pexpire_if_owner -- KEYS: key; ARGV: token, milliseconds
PEXPIRE_IF_OWNER = """\
-- pexpire_if_owner
-- Generated by redis-lua-py from redis_scripts/scripts.py:218. Do not edit.
local key = KEYS[1]
local token = ARGV[1]
local milliseconds = ARGV[2]
if redis.call('GET', key) == token then
  return redis.call('PEXPIRE', key, milliseconds)
end
return 0
"""
_PEXPIRE_IF_OWNER_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def pexpire_if_owner(
    client: _SyncClient,
    /,
    key: _Key,
    token: str,
    milliseconds: int,
) -> int: ...
@overload
def pexpire_if_owner(
    client: _AsyncClient,
    /,
    key: _Key,
    token: str,
    milliseconds: int,
) -> Awaitable[int]: ...
@overload
def pexpire_if_owner(
    client: Any,
    /,
    key: _Key,
    token: str,
    milliseconds: int,
) -> int: ...
def pexpire_if_owner(client: Any, /, key: _Key, token: str, milliseconds: int) -> Any:
    return _run(
        PEXPIRE_IF_OWNER,
        _PEXPIRE_IF_OWNER_CLIENTS,
        client,
        [key],
        [_encode("token", token), _encode("milliseconds", milliseconds)],
    )


# batch_rate_limit -- KEYS: *keys; ARGV: now, window_size
BATCH_RATE_LIMIT = """\
-- batch_rate_limit
-- Generated by redis-lua-py from redis_scripts/scripts.py:228. Do not edit.
-- A Redis command with nothing to return hands Lua false, not nil, so an
-- `is None` test has to accept both. Taking v as an argument also means the
-- operand is evaluated once, not once per comparison.
local function __isnil(v)
  return v == nil or v == false
end
-- Python's v[i:j], for a string or a list: bounds count from 0, a negative
-- bound counts back from the end, and a missing one goes all the way.
local function __slice(v, i, j)
  local n = #v
  if i == nil then i = 0 elseif i < 0 then i = math.max(n + i, 0) elseif i > n then i = n end
  if j == nil then j = n elseif j < 0 then j = math.max(n + j, 0) elseif j > n then j = n end
  if type(v) == 'string' then return string.sub(v, i + 1, j) end
  local out = {}
  for k = i + 1, j do out[#out + 1] = v[k] end
  return out
end
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local counter, counter_key, prefix, window_key, window_start
local results = {}
for i = 0, #keys - 1, 2 do
  window_key = keys[i + 1]
  counter_key = keys[i + 1 + 1]
  window_start = redis.call('GET', window_key)
  if __isnil(window_start) or now - tonumber(window_start) >= window_size then
    prefix = __slice(window_key, nil, -#':window')
    redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
    redis.call('SET', window_key, tostring(now))
    redis.call('SET', counter_key, 1)
    redis.call('EXPIRE', window_key, window_size)
    redis.call('EXPIRE', counter_key, window_size)
    results[#results + 1] = tostring(now)
    results[#results + 1] = 1
  else
    counter = redis.call('INCR', counter_key)
    if redis.call('TTL', counter_key) == -1 then
      redis.call('EXPIRE', counter_key, window_size)
    end
    results[#results + 1] = window_start
    results[#results + 1] = counter
  end
end
return results
"""
_BATCH_RATE_LIMIT_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def batch_rate_limit(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    now: int,
    window_size: int,
) -> list[bytes | int]: ...
@overload
def batch_rate_limit(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    now: int,
    window_size: int,
) -> Awaitable[list[bytes | int]]: ...
@overload
def batch_rate_limit(
    client: Any,
    /,
    keys: Iterable[_Key],
    now: int,
    window_size: int,
) -> list[bytes | int]: ...
def batch_rate_limit(
    client: Any,
    /,
    keys: Iterable[_Key],
    now: int,
    window_size: int,
) -> Any:
    """Count one request against each (window key, counter key) pair in ``keys``.

    A window that is missing or older than ``window_size`` restarts at ``now``, clearing
    the requests and tokens counters that share its prefix. Returns a flat
    [window start, counter, ...] per pair.
    """
    return _run(
        BATCH_RATE_LIMIT,
        _BATCH_RATE_LIMIT_CLIENTS,
        client,
        [*_items("keys", keys)],
        [_encode("now", now), _encode("window_size", window_size)],
    )


# check_and_increment_by_n -- KEYS: *keys; ARGV: *args
CHECK_AND_INCREMENT_BY_N = """\
-- check_and_increment_by_n
-- Generated by redis-lua-py from redis_scripts/scripts.py:261. Do not edit.
-- Python truthiness: 0, '', empty tables and nil are all false.
local function __truthy(v)
  if v == nil or v == false then return false end
  if v == 0 or v == '' then return false end
  if type(v) == 'table' and next(v) == nil then return false end
  return true
end
-- A Redis command with nothing to return hands Lua false, not nil, so an
-- `is None` test has to accept both. Taking v as an argument also means the
-- operand is evaluated once, not once per comparison.
local function __isnil(v)
  return v == nil or v == false
end
-- Python's `a or b`: a when it is truthy by Python's rules, otherwise b.
local function __or(a, b)
  if __truthy(a) then return a end
  return b
end
-- Python's v[i:j], for a string or a list: bounds count from 0, a negative
-- bound counts back from the end, and a missing one goes all the way.
local function __slice(v, i, j)
  local n = #v
  if i == nil then i = 0 elseif i < 0 then i = math.max(n + i, 0) elseif i > n then i = n end
  if j == nil then j = n elseif j < 0 then j = math.max(n + j, 0) elseif j > n then j = n end
  if type(v) == 'string' then return string.sub(v, i + 1, j) end
  local out = {}
  for k = i + 1, j do out[#out + 1] = v[k] end
  return out
end
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local args = {}
for __i2 = 1, #ARGV do
  args[#args + 1] = ARGV[__i2]
end
local active_window_start, blocked, counter_key, current_counter, increment, limit, new_counter, prefix, ttl, window_expired, window_key, window_size, window_start
local now = tonumber(redis.call('TIME')[1])
local descriptor_count = math.floor(#keys / 2)
local expired_windows = {}
local window_starts = {}
for i = 0, descriptor_count - 1 do
  window_key = keys[i * 2 + 1]
  counter_key = keys[i * 2 + 1 + 1]
  limit = tonumber(args[i * 4 + 1])
  increment = tonumber(args[i * 4 + 1 + 1])
  window_size = tonumber(args[i * 4 + 3 + 1])
  window_start = redis.call('GET', window_key)
  window_expired = __isnil(window_start) or now - tonumber(window_start) >= window_size
  current_counter = 0.0
  if not window_expired then
    current_counter = tonumber(__or(redis.call('GET', counter_key), 0))
  end
  if increment > 0 then
    blocked = current_counter + increment > limit
  else
    blocked = current_counter >= limit
  end
  if blocked then
    return {1, i + 1, current_counter, limit}
  end
  expired_windows[#expired_windows + 1] = window_expired
  window_starts[#window_starts + 1] = window_start
end
local results = {0}
local reset_windows = {}
for i = 0, descriptor_count - 1 do
  window_key = keys[i * 2 + 1]
  counter_key = keys[i * 2 + 1 + 1]
  increment = tonumber(args[i * 4 + 1 + 1])
  ttl = tonumber(args[i * 4 + 2 + 1])
  window_size = tonumber(args[i * 4 + 3 + 1])
  if __truthy(expired_windows[i + 1]) then
    active_window_start = now
    if reset_windows[window_key] == nil then
      prefix = __slice(window_key, nil, -#':window')
      redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
      reset_windows[window_key] = true
    end
    redis.call('SET', window_key, tostring(now))
    redis.call('SET', counter_key, increment)
    redis.call('EXPIRE', window_key, window_size)
    if ttl > 0 then
      redis.call('EXPIRE', counter_key, ttl)
    end
    results[#results + 1] = increment
  else
    active_window_start = tonumber(window_starts[i + 1])
    new_counter = redis.call('INCRBY', counter_key, increment)
    if redis.call('TTL', counter_key) == -1 and ttl > 0 then
      redis.call('EXPIRE', counter_key, ttl)
    end
    results[#results + 1] = new_counter
  end
  results[#results + 1] = active_window_start
end
return results
"""
_CHECK_AND_INCREMENT_BY_N_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def check_and_increment_by_n(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[int]: ...
@overload
def check_and_increment_by_n(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Awaitable[list[int]]: ...
@overload
def check_and_increment_by_n(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[int]: ...
def check_and_increment_by_n(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Any:
    """Check and increment one or more descriptors, all or nothing.

    ``keys`` holds a (window key, counter key) pair per descriptor, and ``args`` a
    (limit, increment, ttl seconds, window seconds) quadruple. If any descriptor would go
    over its limit, nothing is written and the result is
    [1, descriptor index (from 1), current counter, limit]. Otherwise it is
    [0, new counter, window start, ...] per descriptor.

    Windows are timed by the Redis server clock (TIME), not a client timestamp, so
    replicas with skewed wall clocks agree on when a window resets.
    """
    return _run(
        CHECK_AND_INCREMENT_BY_N,
        _CHECK_AND_INCREMENT_BY_N_CLIENTS,
        client,
        [*_items("keys", keys)],
        [*_encode_all("args", args)],
    )


# window_guarded_token_increment -- KEYS: *keys; ARGV: *args
WINDOW_GUARDED_TOKEN_INCREMENT = """\
-- window_guarded_token_increment
-- Generated by redis-lua-py from redis_scripts/scripts.py:331. Do not edit.
-- Python truthiness: 0, '', empty tables and nil are all false.
local function __truthy(v)
  if v == nil or v == false then return false end
  if v == 0 or v == '' then return false end
  if type(v) == 'table' and next(v) == nil then return false end
  return true
end
-- A Redis command with nothing to return hands Lua false, not nil, so an
-- `is None` test has to accept both. Taking v as an argument also means the
-- operand is evaluated once, not once per comparison.
local function __isnil(v)
  return v == nil or v == false
end
-- Python's `a or b`: a when it is truthy by Python's rules, otherwise b.
local function __or(a, b)
  if __truthy(a) then return a end
  return b
end
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local args = {}
for __i2 = 1, #ARGV do
  args[#args + 1] = ARGV[__i2]
end
local active_window_start, arg_base, counter_key, expected_window_start, increment, new_counter, ttl, window_key
local results = {}
for i = 0, #keys - 1, 2 do
  window_key = keys[i + 1]
  counter_key = keys[i + 1 + 1]
  arg_base = math.floor(i / 2) * 3
  expected_window_start = args[arg_base + 1]
  increment = tonumber(args[arg_base + 1 + 1])
  ttl = tonumber(args[arg_base + 2 + 1])
  active_window_start = redis.call('GET', window_key)
  if not __isnil(active_window_start) and active_window_start == expected_window_start then
    new_counter = redis.call('INCRBY', counter_key, increment)
    if redis.call('TTL', counter_key) == -1 and ttl > 0 then
      redis.call('EXPIRE', counter_key, ttl)
    end
    results[#results + 1] = 1
    results[#results + 1] = new_counter
  else
    results[#results + 1] = 0
    results[#results + 1] = tonumber(__or(redis.call('GET', counter_key), 0))
  end
end
return results
"""
_WINDOW_GUARDED_TOKEN_INCREMENT_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def window_guarded_token_increment(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[int]: ...
@overload
def window_guarded_token_increment(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Awaitable[list[int]]: ...
@overload
def window_guarded_token_increment(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[int]: ...
def window_guarded_token_increment(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Any:
    """Increment each counter only while its window is still the one the caller saw.

    ``keys`` holds (window key, counter key) pairs and ``args`` an (expected window start,
    increment, ttl seconds) triple per pair. Returns [1, new counter] per applied pair and
    [0, current counter] per pair whose window moved on.
    """
    return _run(
        WINDOW_GUARDED_TOKEN_INCREMENT,
        _WINDOW_GUARDED_TOKEN_INCREMENT_CLIENTS,
        client,
        [*_items("keys", keys)],
        [*_encode_all("args", args)],
    )


# parallel_acquire -- KEYS: *keys; ARGV: *args
PARALLEL_ACQUIRE = """\
-- parallel_acquire
-- Generated by redis-lua-py from redis_scripts/scripts.py:360. Do not edit.
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local args = {}
for __i2 = 1, #ARGV do
  args[#args + 1] = ARGV[__i2]
end
local in_flight, limit, slot_id, slot_ttl
local now = tonumber(redis.call('TIME')[1])
for i = 0, #keys - 1 do
  limit = tonumber(args[i * 3 + 1])
  slot_ttl = tonumber(args[i * 3 + 1 + 1])
  redis.call('ZREMRANGEBYSCORE', keys[i + 1], '-inf', now - slot_ttl)
  in_flight = redis.call('ZCARD', keys[i + 1])
  if in_flight + 1 > limit then
    return {1, i + 1, in_flight, limit}
  end
end
local results = {0}
for i = 0, #keys - 1 do
  slot_ttl = tonumber(args[i * 3 + 1 + 1])
  slot_id = args[i * 3 + 2 + 1]
  redis.call('ZADD', keys[i + 1], now, slot_id)
  redis.call('EXPIRE', keys[i + 1], slot_ttl)
  results[#results + 1] = redis.call('ZCARD', keys[i + 1])
end
return results
"""
_PARALLEL_ACQUIRE_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def parallel_acquire(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[int]: ...
@overload
def parallel_acquire(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Awaitable[list[int]]: ...
@overload
def parallel_acquire(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[int]: ...
def parallel_acquire(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Any:
    """Take one slot in each max_parallel_requests gauge, or none at all.

    A gauge is a sorted set of per-request slot ids scored by acquire time (Redis server
    clock). In-flight requests are counted with ZCARD after pruning slots older than the
    slot TTL, so unlike the windowed RPM/TPM counters a gauge is never reset while requests
    are in flight, a rejected request never occupies a slot, and a slot leaked by a crashed
    worker heals after the slot TTL even under continuous traffic.

    ``args`` holds a (limit, slot ttl seconds, slot id) triple per key. Returns
    [0, in flight, ...] per key on success and [1, key index (from 1), in flight, limit]
    when a gauge is full.
    """
    return _run(
        PARALLEL_ACQUIRE,
        _PARALLEL_ACQUIRE_CLIENTS,
        client,
        [*_items("keys", keys)],
        [*_encode_all("args", args)],
    )


# parallel_release -- KEYS: *keys; ARGV: *slot_ids
PARALLEL_RELEASE = """\
-- parallel_release
-- Generated by redis-lua-py from redis_scripts/scripts.py:392. Do not edit.
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local slot_ids = {}
for __i2 = 1, #ARGV do
  slot_ids[#slot_ids + 1] = ARGV[__i2]
end
local results = {}
for i = 0, #keys - 1 do
  redis.call('ZREM', keys[i + 1], slot_ids[i + 1])
  results[#results + 1] = redis.call('ZCARD', keys[i + 1])
end
return results
"""
_PARALLEL_RELEASE_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def parallel_release(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    slot_ids: Iterable[str],
) -> list[int]: ...
@overload
def parallel_release(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    slot_ids: Iterable[str],
) -> Awaitable[list[int]]: ...
@overload
def parallel_release(
    client: Any,
    /,
    keys: Iterable[_Key],
    slot_ids: Iterable[str],
) -> list[int]: ...
def parallel_release(
    client: Any,
    /,
    keys: Iterable[_Key],
    slot_ids: Iterable[str],
) -> Any:
    """Free this request's slot in each gauge, returning the in-flight count left per key.

    ZREM of an absent member is a no-op, so a release without a matching acquire
    (proxy-side rejection, double-fired callback, slot already expired) can never free a
    slot owned by another request.
    """
    return _run(
        PARALLEL_RELEASE,
        _PARALLEL_RELEASE_CLIENTS,
        client,
        [*_items("keys", keys)],
        [*_encode_all("slot_ids", slot_ids)],
    )


# parallel_count -- KEYS: *keys; ARGV: *slot_ttls
PARALLEL_COUNT = """\
-- parallel_count
-- Generated by redis-lua-py from redis_scripts/scripts.py:407. Do not edit.
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local slot_ttls = {}
for __i2 = 1, #ARGV do
  slot_ttls[#slot_ttls + 1] = tonumber(ARGV[__i2])
end
local now = tonumber(redis.call('TIME')[1])
local results = {}
for i = 0, #keys - 1 do
  redis.call('ZREMRANGEBYSCORE', keys[i + 1], '-inf', now - slot_ttls[i + 1])
  results[#results + 1] = redis.call('ZCARD', keys[i + 1])
end
return results
"""
_PARALLEL_COUNT_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def parallel_count(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    slot_ttls: Iterable[int],
) -> list[int]: ...
@overload
def parallel_count(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    slot_ttls: Iterable[int],
) -> Awaitable[list[int]]: ...
@overload
def parallel_count(
    client: Any,
    /,
    keys: Iterable[_Key],
    slot_ttls: Iterable[int],
) -> list[int]: ...
def parallel_count(
    client: Any,
    /,
    keys: Iterable[_Key],
    slot_ttls: Iterable[int],
) -> Any:
    """Read the in-flight count per gauge, pruning expired slots so leaked ones don't count."""
    return _run(
        PARALLEL_COUNT,
        _PARALLEL_COUNT_CLIENTS,
        client,
        [*_items("keys", keys)],
        [*_encode_all("slot_ttls", slot_ttls)],
    )


# increment_tokens -- KEYS: *keys; ARGV: *args
INCREMENT_TOKENS = """\
-- increment_tokens
-- Generated by redis-lua-py from redis_scripts/scripts.py:418. Do not edit.
local keys = {}
for __i1 = 1, #KEYS do
  keys[#keys + 1] = KEYS[__i1]
end
local args = {}
for __i2 = 1, #ARGV do
  args[#args + 1] = ARGV[__i2]
end
local key, new_value, ttl_seconds
local results = {}
for i = 0, #keys - 1 do
  key = keys[i + 1]
  new_value = redis.call('INCRBYFLOAT', key, tonumber(args[i * 2 + 1]))
  ttl_seconds = tonumber(args[i * 2 + 1 + 1])
  if ttl_seconds > 0 and redis.call('TTL', key) == -1 then
    redis.call('EXPIRE', key, ttl_seconds)
  end
  results[#results + 1] = new_value
end
return results
"""
_INCREMENT_TOKENS_CLIENTS: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


@overload
def increment_tokens(
    client: _SyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[bytes]: ...
@overload
def increment_tokens(
    client: _AsyncClient,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Awaitable[list[bytes]]: ...
@overload
def increment_tokens(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> list[bytes]: ...
def increment_tokens(
    client: Any,
    /,
    keys: Iterable[_Key],
    args: Iterable[str | int | float],
) -> Any:
    """INCRBYFLOAT each key, keeping any expiry it already has.

    ``args`` holds an (increment, ttl seconds) pair per key. A ttl above 0 is set only on
    a key that has no expiry yet, so repeated increments never push it back.
    """
    return _run(
        INCREMENT_TOKENS,
        _INCREMENT_TOKENS_CLIENTS,
        client,
        [*_items("keys", keys)],
        [*_encode_all("args", args)],
    )
