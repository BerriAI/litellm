"""The ``dynamic_queue`` contract shared by every worker through Redis.

Each operation is one Lua script call over five keys sharing a hash tag, so it is atomic and lands on one Redis
Cluster slot. Time is Redis ``TIME``, the one clock every worker agrees on. Waiters sort by member text, so the order
is exact for any integer priority and FIFO by an atomic sequence within one, with no floating-point score. A waiter's
own expiry is checked before it is granted, claimed or kept alive, so the bounded sweep only reclaims space. Every
write keeps the keys alive ``state_ttl_ms`` past the latest of the next probe time, the lease and any waiter's expiry.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError, log_redis_failure
from litellm.router_utils.dynamic_queue import (
    AdmitReply,
    CoordinationUnavailable,
    Gone,
    Grant,
    MatchingFailure,
    Outcome,
    PassThrough,
    PollReply,
    Probe,
    QueueFull,
    QueueSettings,
    Released,
    Succeeded,
    Waiting,
    WaitRequest,
)

REDIS_UNAVAILABLE: Final = "redis_unavailable"
UNEXPECTED_REPLY: Final = "unexpected_reply"
_RANK_WIDTH: Final = 6

_SCRIPT: Final = """
local state, queue, drain, members, expiry = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5]
local op, waiter = ARGV[1], ARGV[3]
local now
if ARGV[2] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
else
  now = tonumber(ARGV[2])
end
local required, initial_backoff, max_backoff = tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[6])
local lease_ms, liveness_ms, max_waiting = tonumber(ARGV[7]), tonumber(ARGV[8]), tonumber(ARGV[9])
local ttl_ms, sweep_limit = tonumber(ARGV[10]), tonumber(ARGV[11])

local function num(name)
  return tonumber(redis.call('HGET', state, name) or '0')
end

local function waiter_of(member)
  return string.sub(member, string.find(member, '|', 1, true) + 22)
end

local function entry_of(w)
  local entry = redis.call('HGET', members, w)
  if not entry then
    return nil
  end
  local first = string.find(entry, '|', 1, true)
  local second = string.find(entry, '|', first + 1, true)
  return tonumber(string.sub(entry, 1, first - 1)), tonumber(string.sub(entry, first + 1, second - 1)),
    string.sub(entry, second + 1)
end

local function member_of(w)
  local _, _, member = entry_of(w)
  return member
end

local function ready_of(w)
  local _, ready = entry_of(w)
  return ready or 0
end

local function expired(w)
  local score = redis.call('ZSCORE', expiry, w)
  return score and tonumber(score) <= now
end

local function congested()
  return redis.call('HGET', state, 'mode') == 'congested'
end

local function healthy()
  return not congested() and redis.call('ZCARD', drain) == 0
end

local function clear_lease()
  redis.call('HDEL', state, 'lease_waiter', 'lease_token', 'lease_until', 'lease_claimed')
end

local function drop(w)
  local member = member_of(w)
  if member then
    redis.call('ZREM', queue, member)
    redis.call('ZREM', drain, member)
    redis.call('HDEL', members, w)
  end
  redis.call('ZREM', expiry, w)
  if redis.call('HGET', state, 'lease_waiter') == w then
    clear_lease()
  end
end

local function release_lease()
  local w = redis.call('HGET', state, 'lease_waiter')
  if w and redis.call('HGET', state, 'lease_claimed') == '0' then
    local member = member_of(w)
    if member and not expired(w) then
      redis.call('ZADD', queue, 0, member)
    else
      drop(w)
    end
  end
  clear_lease()
end

local function sweep()
  local w = redis.call('HGET', state, 'lease_waiter')
  if w and (num('lease_until') <= now or (redis.call('HGET', state, 'lease_claimed') == '0' and expired(w))) then
    release_lease()
  end
  for _, e in ipairs(redis.call('ZRANGEBYSCORE', expiry, '-inf', now, 'LIMIT', 0, sweep_limit)) do
    drop(e)
  end
end

local function first_ready(set)
  for _, member in ipairs(redis.call('ZRANGE', set, 0, max_waiting - 1)) do
    local w = waiter_of(member)
    if expired(w) then
      drop(w)
    elseif ready_of(w) <= now then
      return member, w
    end
  end
  return nil
end

local function grant_probe()
  if not congested() or redis.call('HEXISTS', state, 'lease_waiter') == 1 or now < num('next_probe') then
    return
  end
  local head, w = first_ready(queue)
  if not head then
    return
  end
  local token = redis.call('HINCRBY', state, 'tokens', 1)
  redis.call('HSET', state, 'lease_waiter', w, 'lease_token', token, 'lease_until', now + lease_ms, 'lease_claimed', 0)
  redis.call('ZREM', queue, head)
end

local function claim_probe(w)
  if redis.call('HGET', state, 'lease_waiter') ~= w or redis.call('HGET', state, 'lease_claimed') ~= '0' then
    return nil
  end
  redis.call('HSET', state, 'lease_claimed', 1, 'lease_until', now + lease_ms)
  redis.call('HDEL', members, w)
  redis.call('ZREM', expiry, w)
  return {'grant', 'probe|' .. num('generation') .. '|' .. num('lease_token')}
end

local function congest(retry_after, failed_probe)
  local backoff = initial_backoff
  if congested() then
    backoff = num('backoff')
    if failed_probe then
      backoff = math.min(backoff * 2, max_backoff)
    end
  end
  release_lease()
  redis.call('ZUNIONSTORE', queue, 2, queue, drain)
  redis.call('DEL', drain)
  redis.call('HINCRBY', state, 'generation', 1)
  local next_probe = math.max(num('next_probe'), now + math.max(backoff, retry_after))
  redis.call('HSET', state, 'mode', 'congested', 'successes', 0, 'backoff', backoff, 'next_probe', next_probe)
end

local function recover()
  redis.call('HINCRBY', state, 'generation', 1)
  redis.call('HDEL', state, 'mode', 'successes', 'backoff', 'next_probe')
  clear_lease()
  if redis.call('ZCARD', queue) > 0 then
    redis.call('RENAME', queue, drain)
  end
end

local function touch()
  local horizon = math.max(num('next_probe'), num('lease_until'), now)
  local latest = redis.call('ZRANGE', expiry, -1, -1, 'WITHSCORES')[2]
  if latest then
    horizon = math.max(horizon, tonumber(latest))
  end
  for _, key in ipairs(KEYS) do
    redis.call('PEXPIRE', key, horizon - now + ttl_ms)
  end
end

local function settle_congested(w)
  grant_probe()
  local claimed = claim_probe(w)
  touch()
  if claimed then
    return claimed
  end
  local probe_in = num('next_probe') - now
  if probe_in > 0 and redis.call('HEXISTS', state, 'lease_waiter') == 0 then
    return {'wait', 'probe', probe_in}
  end
  return {'wait'}
end

local function admitted(w, member)
  redis.call('ZREM', queue, member)
  redis.call('ZREM', drain, member)
  redis.call('HDEL', members, w)
  redis.call('ZREM', expiry, w)
  touch()
end

sweep()

if op == 'admit' then
  if healthy() then
    touch()
    return {'pass', num('generation')}
  end
  local budget = tonumber(ARGV[13])
  if budget <= 0 then
    return {'gone'}
  end
  local unclaimed = redis.call('HGET', state, 'lease_claimed') == '0' and 1 or 0
  if redis.call('ZCARD', queue) + redis.call('ZCARD', drain) + unclaimed >= max_waiting then
    touch()
    return {'full'}
  end
  local seq = redis.call('HINCRBY', state, 'seq', 1)
  local member = ARGV[12] .. '|' .. string.format('%020d', seq) .. '|' .. waiter
  local deadline = now + budget
  redis.call('ZADD', queue, 0, member)
  redis.call('HSET', members, waiter, deadline .. '|' .. (now + tonumber(ARGV[14])) .. '|' .. member)
  redis.call('ZADD', expiry, math.min(deadline, now + liveness_ms), waiter)
  if congested() then
    return settle_congested(waiter)
  end
  touch()
  return {'wait', redis.call('ZCARD', drain)}
end

if op == 'poll' then
  if expired(waiter) then
    drop(waiter)
    grant_probe()
    touch()
    return {'gone'}
  end
  local deadline, _, member = entry_of(waiter)
  if member then
    local ready = now + tonumber(ARGV[12])
    redis.call('HSET', members, waiter, deadline .. '|' .. ready .. '|' .. member)
    if ready > now and redis.call('HGET', state, 'lease_waiter') == waiter
      and redis.call('HGET', state, 'lease_claimed') == '0' then
      clear_lease()
      redis.call('ZADD', queue, 0, member)
    end
  end
  local claimed = claim_probe(waiter)
  if claimed then
    touch()
    return claimed
  end
  if not member then
    return {'gone'}
  end
  redis.call('ZADD', expiry, math.min(deadline, now + liveness_ms), waiter)
  if redis.call('ZSCORE', drain, member) then
    local _, head_waiter = first_ready(drain)
    if head_waiter == waiter then
      admitted(waiter, member)
      return {'grant', 'released|' .. num('generation')}
    end
    touch()
    return {'wait', redis.call('ZRANK', drain, member)}
  end
  if healthy() then
    admitted(waiter, member)
    return {'pass', num('generation')}
  end
  if not congested() then
    touch()
    return {'wait', redis.call('ZCARD', drain)}
  end
  return settle_congested(waiter)
end

if op == 'renew' then
  if redis.call('HGET', state, 'lease_waiter') == waiter and redis.call('HGET', state, 'lease_claimed') == '1'
    and num('lease_token') == tonumber(ARGV[12]) then
    redis.call('HSET', state, 'lease_until', now + lease_ms)
    touch()
    return {'renewed'}
  end
  return {'lost'}
end

if op == 'report' then
  local active_probe = ARGV[12] == 'probe'
    and redis.call('HGET', state, 'lease_waiter') == waiter
    and redis.call('HGET', state, 'lease_claimed') == '1'
    and num('lease_token') == tonumber(ARGV[14])
    and num('generation') == tonumber(ARGV[13])
  if ARGV[15] == 'matching' then
    congest(tonumber(ARGV[16]), active_probe)
  elseif active_probe then
    clear_lease()
    if ARGV[15] == 'success' then
      if redis.call('HINCRBY', state, 'successes', 1) >= required then
        recover()
      end
    else
      redis.call('HSET', state, 'successes', 0)
    end
    grant_probe()
  end
  touch()
  return {'ok'}
end

if op == 'abandon' then
  drop(waiter)
  grant_probe()
  touch()
  return {'ok'}
end

return redis.error_reply('unknown dynamic queue op ' .. op)
"""

_REPLY: Final = TypeAdapter(tuple[str | bytes | int, ...])


class ScriptHost(Protocol):
    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]: ...


def rank_key(priority: int) -> str:
    """Text that sorts byte-wise in the same order as ``priority`` does numerically, for any integer."""
    digits: Final = str(abs(priority))
    if priority >= 0:
        return f"p{len(digits):0{_RANK_WIDTH}d}{digits}"
    complement: Final = digits.translate(str.maketrans("0123456789", "9876543210"))
    return f"n{10**_RANK_WIDTH - 1 - len(digits):0{_RANK_WIDTH}d}{complement}"


def _keys(capacity: str) -> tuple[str, ...]:
    tag: Final = f"litellm:dynamic_queue:{{dq:{capacity}}}"
    return tuple(f"{tag}:{part}" for part in ("state", "queue", "drain", "members", "expiry"))


def _text(item: str | bytes | int) -> str:
    return item.decode() if isinstance(item, bytes) else str(item)


def _unexpected(reply: tuple[str, ...]) -> CoordinationUnavailable:
    verbose_logger.error("dynamic queue: unexpected Redis script reply %r", reply)
    return CoordinationUnavailable(reason=UNEXPECTED_REPLY)


def _settled(reply: tuple[str, ...]) -> Grant | Waiting | QueueFull | Gone | CoordinationUnavailable:
    match reply:
        case ("pass", generation):
            return PassThrough(generation=int(generation))
        case ("grant", encoded):
            return _grant(reply, encoded)
        case ("wait",):
            return Waiting()
        case ("wait", ahead):
            return Waiting(ahead=int(ahead))
        case ("wait", "probe", probe_in_ms):
            return Waiting(probe_in_ms=int(probe_in_ms))
        case ("full",):
            return QueueFull()
        case ("gone",):
            return Gone()
        case _:
            return _unexpected(reply)


def _grant(reply: tuple[str, ...], encoded: str) -> Grant | CoordinationUnavailable:
    match tuple(encoded.split("|")):
        case ("probe", generation, token):
            return Probe(generation=int(generation), token=int(token))
        case ("released", generation):
            return Released(generation=int(generation))
        case _:
            return _unexpected(reply)


def _outcome_args(outcome: Outcome) -> tuple[str, int]:
    if isinstance(outcome, MatchingFailure):
        return "matching", outcome.retry_after_ms
    return ("success" if isinstance(outcome, Succeeded) else "other"), 0


def _grant_args(grant: Grant) -> tuple[str, int, int]:
    match grant:
        case Probe(generation=generation, token=token):
            return "probe", generation, token
        case Released(generation=generation):
            return "released", generation, 0
        case PassThrough(generation=generation):
            return "pass", generation, 0


class RedisQueueBackend:
    """``clock`` stands in for Redis ``TIME`` with milliseconds of the caller's choosing, so a test can drive every
    worker's time together. Production leaves it unset."""

    def __init__(
        self,
        scripts: ScriptHost,
        settings: QueueSettings | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._run = scripts.async_register_script(_SCRIPT)
        self._settings = settings or QueueSettings()
        self._clock = clock

    @property
    def settings(self) -> QueueSettings:
        return self._settings

    async def _call(
        self, op: str, capacity: str, waiter_id: str, extra: Sequence[str | int]
    ) -> tuple[str, ...] | CoordinationUnavailable:
        from redis.exceptions import RedisError

        settings: Final = self._settings
        args: Final = (
            op,
            "" if self._clock is None else self._clock(),
            waiter_id,
            settings.required_successes,
            settings.initial_backoff_ms,
            settings.max_backoff_ms,
            settings.lease_ms,
            settings.liveness_ms,
            settings.max_waiting,
            settings.state_ttl_ms,
            settings.sweep_limit,
            *extra,
        )
        try:
            raw: Final = await self._run(keys=_keys(capacity), args=args)
        except (RedisError, RedisCircuitBreakerOpenError, OSError) as error:
            log_redis_failure(verbose_logger, logging.ERROR, f"dynamic queue {op} could not reach Redis", error)
            return CoordinationUnavailable(reason=REDIS_UNAVAILABLE)
        try:
            return tuple(_text(item) for item in _REPLY.validate_python(raw))
        except ValidationError:
            return _unexpected((type(raw).__name__,))

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        reply: Final = await self._call(
            "admit", request.capacity, request.waiter_id, (rank_key(request.priority), budget_ms, ready_in_ms)
        )
        return reply if isinstance(reply, CoordinationUnavailable) else _settled(reply)

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        reply: Final = await self._call("poll", capacity, waiter_id, (ready_in_ms,))
        if isinstance(reply, CoordinationUnavailable):
            return reply
        settled: Final = _settled(reply)
        return _unexpected(reply) if isinstance(settled, QueueFull) else settled

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        reply: Final = await self._call("renew", capacity, waiter_id, (token,))
        return reply if isinstance(reply, CoordinationUnavailable) else reply == ("renewed",)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        reply: Final = await self._call("report", capacity, waiter_id, (*_grant_args(grant), *_outcome_args(outcome)))
        return reply if isinstance(reply, CoordinationUnavailable) else None

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        reply: Final = await self._call("abandon", capacity, waiter_id, ())
        return reply if isinstance(reply, CoordinationUnavailable) else None
