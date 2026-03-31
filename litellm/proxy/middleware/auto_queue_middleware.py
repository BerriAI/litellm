"""litellm/proxy/middleware/auto_queue_middleware.py"""
import asyncio
import heapq
import json
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from starlette.types import ASGIApp, Receive, Scope, Send

# -- Configuration ------------------------------------------------------------

DEFAULT_MAX_CONCURRENT = int(os.environ.get("AUTOQ_DEFAULT_MAX_CONCURRENT", "20"))
SCALE_UP_THRESHOLD = int(os.environ.get("AUTOQ_SCALE_UP_THRESHOLD", "20"))
SCALE_DOWN_STEP = int(os.environ.get("AUTOQ_SCALE_DOWN_STEP", "1"))
CEILING = int(os.environ.get("AUTOQ_CEILING", "50"))
DEFAULT_TIMEOUT = int(os.environ.get("AUTOQ_DEFAULT_TIMEOUT", "300"))
MAX_QUEUE_DEPTH = int(os.environ.get("AUTOQ_MAX_QUEUE_DEPTH", "100"))
REDIS_HOST = os.environ.get("AUTOQ_REDIS_HOST", os.environ.get("REDIS_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("AUTOQ_REDIS_PORT", os.environ.get("REDIS_PORT", "6379")))
REDIS_DB = int(os.environ.get("AUTOQ_REDIS_DB", "3"))
ACTIVE_KEY_TTL = 600  # seconds


# -- Lua Scripts ---------------------------------------------------------------

_LUA_ACQUIRE = """
local active = tonumber(redis.call('GET', KEYS[1])) or 0
local limit = tonumber(redis.call('GET', KEYS[2])) or tonumber(ARGV[1])
if active < limit then
    redis.call('INCR', KEYS[1])
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
    return 1
end
return 0
"""

_LUA_RELEASE = """
local active = tonumber(redis.call('GET', KEYS[1])) or 0
if active > 0 then
    redis.call('DECR', KEYS[1])
end
return math.max(0, active - 1)
"""

_LUA_SCALE_UP = """
local count = redis.call('INCR', KEYS[1])
local threshold = tonumber(ARGV[1])
local ceiling = tonumber(redis.call('GET', KEYS[3])) or tonumber(ARGV[2])
if tonumber(count) >= threshold then
    local current = tonumber(redis.call('GET', KEYS[2])) or tonumber(ARGV[3])
    if current < ceiling then
        redis.call('SET', KEYS[2], current + 1)
    end
    redis.call('SET', KEYS[1], 0)
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
redis.call('SET', KEYS[2], 0)
return tonumber(redis.call('GET', KEYS[1]))
"""


# -- Redis Helper --------------------------------------------------------------

class AutoQueueRedis:
    """Atomic Redis operations for concurrency control and auto-scaling."""

    def __init__(
        self,
        redis: aioredis.Redis,
        default_max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        ceiling: int = CEILING,
        scale_up_threshold: int = SCALE_UP_THRESHOLD,
        scale_down_step: int = SCALE_DOWN_STEP,
    ):
        self.redis = redis
        self.default_max_concurrent = default_max_concurrent
        self.ceiling = ceiling
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_step = scale_down_step
        self._acquire_script = redis.register_script(_LUA_ACQUIRE)
        self._release_script = redis.register_script(_LUA_RELEASE)
        self._scale_up_script = redis.register_script(_LUA_SCALE_UP)
        self._scale_down_script = redis.register_script(_LUA_SCALE_DOWN)

    async def try_acquire(self, model: str) -> bool:
        result = await self._acquire_script(
            keys=[f"autoq:active:{model}", f"autoq:limit:{model}"],
            args=[self.default_max_concurrent, ACTIVE_KEY_TTL],
        )
        return bool(result)

    async def release(self, model: str) -> int:
        result = await self._release_script(
            keys=[f"autoq:active:{model}"],
        )
        return int(result)

    async def on_success(self, model: str) -> None:
        await self._scale_up_script(
            keys=[
                f"autoq:success:{model}",
                f"autoq:limit:{model}",
                f"autoq:ceiling:{model}",
            ],
            args=[self.scale_up_threshold, self.ceiling, self.default_max_concurrent],
        )

    async def on_429(self, model: str) -> None:
        await self._scale_down_script(
            keys=[f"autoq:limit:{model}", f"autoq:success:{model}"],
            args=[self.default_max_concurrent, self.scale_down_step],
        )

    async def get_model_info(self, model: str) -> dict:
        pipe = self.redis.pipeline()
        pipe.get(f"autoq:active:{model}")
        pipe.get(f"autoq:limit:{model}")
        pipe.get(f"autoq:ceiling:{model}")
        active_raw, limit_raw, ceiling_raw = await pipe.execute()
        return {
            "active": int(active_raw or 0),
            "limit": int(limit_raw or self.default_max_concurrent),
            "queued": 0,  # filled by middleware, not Redis
            "ceiling": int(ceiling_raw or self.ceiling),
        }
