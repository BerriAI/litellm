from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis

from .auto_queue_state import (
    AUTOQ_ACTIVE_LEASE_KEY_PREFIX,
    AUTOQ_CLAIM_KEY_PREFIX,
    AUTOQ_REQUEST_KEY_PREFIX,
    AutoQueueRequestState,
    active_key,
    active_lease_key,
    ceiling_key,
    claim_key,
    limit_key,
    queue_key,
    queue_score,
    request_key,
    request_state_hash,
    request_state_from_hash,
    success_key,
)

DEFAULT_ACTIVE_KEY_TTL = 600
DEFAULT_DATA_TTL = 24 * 60 * 60
DEFAULT_LEASE_TTL = 60


def current_time_ms() -> int:
    return int(time.time() * 1000)


def _decode_script_value(value: object) -> Optional[str]:
    if value in (None, "", b""):
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


@dataclass(slots=True)
class AdmitDecision:
    decision: str
    claim_token: Optional[str] = None
    request_state: Optional[AutoQueueRequestState] = None


@dataclass(slots=True)
class ReleaseTransfer:
    claimed_request_id: Optional[str]
    claim_token: Optional[str]


class DistributedAutoQueueRedis:
    """Redis-backed distributed queue state with Lua-atomic transitions."""

    def __init__(
        self,
        redis: aioredis.Redis,
        default_max_concurrent: int = 20,
        ceiling: int = 50,
        scale_up_threshold: int = 20,
        scale_down_step: int = 1,
        max_queue_depth: int = 100,
    ) -> None:
        self.redis = redis
        self.default_max_concurrent = default_max_concurrent
        self.ceiling = ceiling
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_step = scale_down_step
        self.max_queue_depth = max_queue_depth
        self._try_acquire_script = redis.register_script(_LUA_TRY_ACQUIRE)
        self._release_script = redis.register_script(_LUA_RELEASE)
        self._scale_up_script = redis.register_script(_LUA_SCALE_UP)
        self._scale_down_script = redis.register_script(_LUA_SCALE_DOWN)
        self._admit_or_enqueue_script = redis.register_script(_LUA_ADMIT_OR_ENQUEUE)
        self._release_and_claim_next_script = redis.register_script(_LUA_RELEASE_AND_CLAIM_NEXT)
        self._abandon_queued_request_script = redis.register_script(_LUA_ABANDON_QUEUED_REQUEST)

    async def try_acquire(self, model: str) -> bool:
        result = await self._try_acquire_script(
            keys=[active_key(model), limit_key(model)],
            args=[self.default_max_concurrent, DEFAULT_ACTIVE_KEY_TTL],
        )
        return bool(result)

    async def release(self, model: str) -> int:
        result = await self._release_script(
            keys=[active_key(model)],
            args=[DEFAULT_ACTIVE_KEY_TTL],
        )
        return int(result)

    async def on_success(self, model: str) -> None:
        await self._scale_up_script(
            keys=[success_key(model), limit_key(model), ceiling_key(model)],
            args=[self.scale_up_threshold, self.ceiling, self.default_max_concurrent, DEFAULT_DATA_TTL],
        )

    async def on_429(self, model: str) -> None:
        await self._scale_down_script(
            keys=[limit_key(model), success_key(model)],
            args=[self.default_max_concurrent, self.scale_down_step, DEFAULT_DATA_TTL],
        )

    async def get_model_info(self, model: str) -> dict[str, int]:
        pipe = self.redis.pipeline()
        pipe.get(active_key(model))
        pipe.get(limit_key(model))
        pipe.get(ceiling_key(model))
        pipe.zcard(queue_key(model))
        active_raw, limit_raw, ceiling_raw, queued_raw = await pipe.execute()
        return {
            "active": int(active_raw or 0),
            "limit": int(limit_raw or self.default_max_concurrent),
            "queued": int(queued_raw or 0),
            "ceiling": int(ceiling_raw or self.ceiling),
        }

    async def admit_or_enqueue(
        self,
        model: str,
        request_id: str,
        priority: int,
        deadline_at_ms: int,
        worker_id: str,
    ) -> AdmitDecision:
        now_ms = current_time_ms()
        claim_token = uuid.uuid4().hex
        result = await self._admit_or_enqueue_script(
            keys=[
                active_key(model),
                limit_key(model),
                queue_key(model),
                request_key(request_id),
                claim_key(request_id),
                active_lease_key(request_id),
            ],
            args=[
                self.default_max_concurrent,
                DEFAULT_ACTIVE_KEY_TTL,
                DEFAULT_DATA_TTL,
                DEFAULT_LEASE_TTL,
                self.max_queue_depth,
                now_ms,
                request_id,
                model,
                priority,
                deadline_at_ms,
                worker_id,
                claim_token,
            ],
        )
        decision = int(result[0])
        returned_claim_token = _decode_script_value(result[1])
        admitted = decision == 1
        rejected = decision == 2
        if admitted:
            raw_state = await self.redis.hgetall(request_key(request_id))
            state = request_state_from_hash(raw_state)
            return AdmitDecision(
                decision="admit_now",
                claim_token=returned_claim_token or claim_token,
                request_state=state,
            )
        if rejected:
            raw_state = await self.redis.hgetall(request_key(request_id))
            return AdmitDecision(decision="queue_full", request_state=request_state_from_hash(raw_state))
        raw_state = await self.redis.hgetall(request_key(request_id))
        return AdmitDecision(decision="queued", request_state=request_state_from_hash(raw_state))

    async def release_and_claim_next(
        self,
        model: str,
        request_id: str,
        *,
        terminal_state: str = "completed",
        allow_missing_active: bool = False,
        claim_next: bool = True,
    ) -> ReleaseTransfer:
        now_ms = current_time_ms()
        claim_token = uuid.uuid4().hex
        result = await self._release_and_claim_next_script(
            keys=[active_key(model), queue_key(model), request_key(request_id)],
            args=[
                DEFAULT_ACTIVE_KEY_TTL,
                DEFAULT_DATA_TTL,
                DEFAULT_LEASE_TTL,
                now_ms,
                claim_token,
                request_id,
                terminal_state,
                1 if allow_missing_active else 0,
                1 if claim_next else 0,
                AUTOQ_REQUEST_KEY_PREFIX,
                AUTOQ_CLAIM_KEY_PREFIX,
                AUTOQ_ACTIVE_LEASE_KEY_PREFIX,
            ],
        )
        claimed_request_id = _decode_script_value(result[0])
        returned_claim_token = _decode_script_value(result[1])
        return ReleaseTransfer(
            claimed_request_id=claimed_request_id,
            claim_token=returned_claim_token,
        )

    async def abandon_queued_request(
        self,
        model: str,
        request_id: str,
        *,
        terminal_state: str,
    ) -> bool:
        now_ms = current_time_ms()
        result = await self._abandon_queued_request_script(
            keys=[
                queue_key(model),
                request_key(request_id),
                claim_key(request_id),
                active_lease_key(request_id),
            ],
            args=[
                DEFAULT_DATA_TTL,
                now_ms,
                terminal_state,
                request_id,
            ],
        )
        return bool(result)


_LUA_TRY_ACQUIRE = """
local active = tonumber(redis.call('GET', KEYS[1])) or 0
local limit = tonumber(redis.call('GET', KEYS[2])) or tonumber(ARGV[1])
if active < limit then
    redis.call('SET', KEYS[1], active + 1, 'EX', tonumber(ARGV[2]))
    return 1
end
return 0
"""


_LUA_RELEASE = """
local active = tonumber(redis.call('GET', KEYS[1])) or 0
if active <= 0 then
    redis.call('SET', KEYS[1], 0, 'EX', tonumber(ARGV[1]))
    return 0
end
local new_value = active - 1
redis.call('SET', KEYS[1], new_value, 'EX', tonumber(ARGV[1]))
return new_value
"""


_LUA_SCALE_UP = """
local count = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
local threshold = tonumber(ARGV[1])
local ceiling = tonumber(redis.call('GET', KEYS[3])) or tonumber(ARGV[2])
if tonumber(count) >= threshold then
    local current = tonumber(redis.call('GET', KEYS[2])) or tonumber(ARGV[3])
    if current < ceiling then
        redis.call('SET', KEYS[2], current + 1)
        redis.call('EXPIRE', KEYS[2], tonumber(ARGV[4]))
    end
    redis.call('SET', KEYS[1], 0)
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
end
return 0
"""


_LUA_SCALE_DOWN = """
local current = tonumber(redis.call('GET', KEYS[1])) or tonumber(ARGV[1])
local step = tonumber(ARGV[2])
if current - step >= 1 then
    redis.call('SET', KEYS[1], current - step)
else
    redis.call('SET', KEYS[1], 1)
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
redis.call('SET', KEYS[2], 0)
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[3]))
return tonumber(redis.call('GET', KEYS[1]))
"""


_LUA_ADMIT_OR_ENQUEUE = """
local active = tonumber(redis.call('GET', KEYS[1])) or 0
local limit = tonumber(redis.call('GET', KEYS[2])) or tonumber(ARGV[1])
local queued = tonumber(redis.call('ZCARD', KEYS[3])) or 0
local max_queue_depth = tonumber(ARGV[5])
local now_ms = tostring(ARGV[6])
local request_id = ARGV[7]
local model = ARGV[8]
local priority = tostring(ARGV[9])
local deadline_at_ms = tostring(ARGV[10])
local worker_id = ARGV[11]
local claim_token = ARGV[12]

if active < limit and queued == 0 then
    redis.call('SET', KEYS[1], active + 1, 'EX', tonumber(ARGV[2]))
    redis.call(
        'HSET',
        KEYS[4],
        'request_id', request_id,
        'model', model,
        'priority', priority,
        'state', 'active',
        'enqueued_at_ms', now_ms,
        'deadline_at_ms', deadline_at_ms,
        'worker_id', worker_id,
        'claim_token', claim_token,
        'claimed_at_ms', now_ms,
        'started_at_ms', now_ms,
        'finished_at_ms', ''
    )
    redis.call('EXPIRE', KEYS[4], tonumber(ARGV[3]))
    redis.call('SET', KEYS[5], claim_token, 'EX', tonumber(ARGV[4]))
    redis.call('DEL', KEYS[6])
    redis.call('HSET', KEYS[6], 'worker_id', worker_id, 'claim_token', claim_token)
    redis.call('EXPIRE', KEYS[6], tonumber(ARGV[4]))
    return {1, claim_token}
end

if queued >= max_queue_depth then
    return {2, ''}
end

redis.call(
    'HSET',
    KEYS[4],
    'request_id', request_id,
    'model', model,
    'priority', priority,
    'state', 'queued',
    'enqueued_at_ms', now_ms,
    'deadline_at_ms', deadline_at_ms,
    'worker_id', worker_id,
    'claim_token', '',
    'claimed_at_ms', '',
    'started_at_ms', '',
    'finished_at_ms', ''
)
redis.call('EXPIRE', KEYS[4], tonumber(ARGV[3]))
redis.call('ZADD', KEYS[3], tonumber(priority) * 10000000000000 + tonumber(now_ms), request_id)
return {0, ''}
"""


_LUA_RELEASE_AND_CLAIM_NEXT = """
local active = tonumber(redis.call('GET', KEYS[1])) or 0
local allow_missing_active = tonumber(ARGV[8])
if active <= 0 and allow_missing_active ~= 1 then
    return {'', ''}
end
if active <= 0 then
    active = 1
end
local active_after_release = active - 1
local claim_token = ARGV[5]
local released_request_id = ARGV[6]
local terminal_state = ARGV[7]
local allow_claim_next = tonumber(ARGV[9])
local now_ms = tostring(ARGV[4])
local request_key_prefix = ARGV[10]
local claim_key_prefix = ARGV[11]
local active_lease_key_prefix = ARGV[12]
local released_claim_key = claim_key_prefix .. released_request_id
local released_active_lease_key = active_lease_key_prefix .. released_request_id

redis.call(
    'HSET',
    KEYS[3],
    'state', terminal_state,
    'claim_token', '',
    'finished_at_ms', now_ms
)
redis.call('EXPIRE', KEYS[3], tonumber(ARGV[2]))
redis.call('DEL', released_claim_key)
redis.call('DEL', released_active_lease_key)

if allow_claim_next ~= 1 then
    redis.call('SET', KEYS[1], active_after_release, 'EX', tonumber(ARGV[1]))
    return {'', ''}
end

while true do
    local popped = redis.call('ZPOPMIN', KEYS[2], 1)
    if not popped[1] then
        redis.call('SET', KEYS[1], active_after_release, 'EX', tonumber(ARGV[1]))
        return {'', ''}
    end

    local request_id = popped[1]
    local next_request_key = request_key_prefix .. request_id
    local next_claim_key = claim_key_prefix .. request_id
    local next_active_lease_key = active_lease_key_prefix .. request_id

    local raw = redis.call('HGETALL', next_request_key)
    if #raw == 0 then
        -- stale queue member, skip it and try the next one
    else
        local worker_id = ''
        local state = ''
        local deadline_at_ms = ''
        for index = 1, #raw, 2 do
            if raw[index] == 'worker_id' then
                worker_id = raw[index + 1]
            elseif raw[index] == 'state' then
                state = raw[index + 1]
            elseif raw[index] == 'deadline_at_ms' then
                deadline_at_ms = raw[index + 1]
            end
        end

        if state ~= 'queued' then
            -- terminal or already claimed state, skip it
        elseif deadline_at_ms ~= '' and tonumber(deadline_at_ms) <= tonumber(now_ms) then
            redis.call(
                'HSET',
                next_request_key,
                'state', 'timed_out',
                'claim_token', '',
                'finished_at_ms', now_ms
            )
            redis.call('EXPIRE', next_request_key, tonumber(ARGV[2]))
            redis.call('DEL', next_claim_key)
            redis.call('DEL', next_active_lease_key)
        else

            redis.call(
                'HSET',
                next_request_key,
                'state', 'claimed',
                'claim_token', claim_token,
                'claimed_at_ms', now_ms,
                'started_at_ms', '',
                'finished_at_ms', ''
            )
            redis.call('EXPIRE', next_request_key, tonumber(ARGV[2]))
            redis.call('SET', next_claim_key, claim_token, 'EX', tonumber(ARGV[3]))
            redis.call('DEL', next_active_lease_key)
            redis.call('HSET', next_active_lease_key, 'worker_id', worker_id, 'claim_token', claim_token)
            redis.call('EXPIRE', next_active_lease_key, tonumber(ARGV[3]))
            redis.call('SET', KEYS[1], active, 'EX', tonumber(ARGV[1]))
            return {request_id, claim_token}
        end
    end
end
"""


_LUA_ABANDON_QUEUED_REQUEST = """
local state = redis.call('HGET', KEYS[2], 'state')
if not state or state ~= 'queued' then
    return 0
end

redis.call(
    'HSET',
    KEYS[2],
    'state', ARGV[3],
    'claim_token', '',
    'finished_at_ms', tostring(ARGV[2])
)
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[1]))
redis.call('ZREM', KEYS[1], ARGV[4])
redis.call('DEL', KEYS[3])
redis.call('DEL', KEYS[4])
return 1
"""


AutoQueueRedis = DistributedAutoQueueRedis
