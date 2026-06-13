"""
Storage for in-flight budget reservations ("holds").

A hold is one request's optimistic reservation against a budget counter. Holds
are kept separate from the committed spend counter and each one expires on its
own, so a pod that dies before settling a request (OOMKill, SIGKILL) cannot
leave a reservation pinned on the shared counter. The committed spend counter is
only ever incremented by real cost; it is never decremented to "give back" a
reservation, which is what made the previous design corrupt under pod churn.

Redis backend (multi-pod): one sorted set per counter, member ``<hold_id>:<cost>``
scored by expiry. Admission prunes expired members (``ZREMRANGEBYSCORE``), adds
its own hold, and sums the live members in a single atomic script, so an
orphaned hold ages out within ``ttl_seconds``.

In-memory backend (single pod, no Redis): a per-counter dict of
``hold_id -> (cost, expiry)``. A single pod has no cross-pod orphan to recover,
so losing this dict on pod death is the correct, complete cleanup.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from litellm.caching.redis_cache import RedisCache
    from litellm.caching.dual_cache import DualCache

_HOLD_KEY_PREFIX = "budget_holds:"

# Atomic admission against the per-counter sorted set: prune expired holds, add
# this request's hold, bound the set's lifetime, then return the live total.
_PLACE_AND_TOTAL_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[4])
local members = redis.call('ZRANGE', KEYS[1], 0, -1)
local total = 0.0
for i = 1, #members do
  local pos = string.find(members[i], ':[^:]*$')
  if pos then
    local c = tonumber(string.sub(members[i], pos + 1))
    if c then total = total + c end
  end
end
return tostring(total)
"""

# Atomic resize: swap this hold's member for its resized cost in one step so a
# concurrent admission's sum never runs while the hold is momentarily absent.
_RESIZE_LUA = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[4])
"""


def _member(hold_id: str, cost: float) -> str:
    return f"{hold_id}:{cost!r}"


class BudgetHoldStore:
    def __init__(self, dual_cache: "DualCache", ttl_seconds: int) -> None:
        self._dual_cache = dual_cache
        self._ttl_seconds = ttl_seconds
        self._memory_holds: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def _redis(self) -> "RedisCache | None":
        return self._dual_cache.redis_cache

    def _zkey(self, counter_key: str) -> str:
        return f"{_HOLD_KEY_PREFIX}{counter_key}"

    async def place_and_total(
        self, counter_key: str, hold_id: str, cost: float
    ) -> float:
        """Record a hold of ``cost`` and return the sum of all live holds (this one included)."""
        redis = self._redis()
        if redis is not None:
            client: "Redis" = redis.init_async_client()  # type: ignore[assignment]
            zkey = redis.check_and_fix_namespace(key=self._zkey(counter_key))
            now = time.time()
            total = await client.eval(
                _PLACE_AND_TOTAL_LUA,
                1,
                zkey,
                now,
                now + self._ttl_seconds,
                _member(hold_id, cost),
                self._ttl_seconds,
            )
            return float(total)
        return self._memory_place_and_total(counter_key, hold_id, cost)

    async def resize(
        self, counter_key: str, hold_id: str, old_cost: float, new_cost: float
    ) -> None:
        redis = self._redis()
        if redis is not None:
            client: "Redis" = redis.init_async_client()  # type: ignore[assignment]
            zkey = redis.check_and_fix_namespace(key=self._zkey(counter_key))
            # ZREM of the last member would drop the set and its key TTL, so the
            # script re-sets EXPIRE; doing it atomically also keeps a concurrent
            # admission from summing while the resized hold is absent.
            await client.eval(
                _RESIZE_LUA,
                1,
                zkey,
                _member(hold_id, old_cost),
                time.time() + self._ttl_seconds,
                _member(hold_id, new_cost),
                self._ttl_seconds,
            )
            return
        holds = self._memory_holds.get(counter_key)
        if holds is not None and hold_id in holds:
            _, expiry = holds[hold_id]
            holds[hold_id] = (new_cost, expiry)

    async def remove(self, counter_key: str, hold_id: str, cost: float) -> None:
        redis = self._redis()
        if redis is not None:
            client: "Redis" = redis.init_async_client()  # type: ignore[assignment]
            zkey = redis.check_and_fix_namespace(key=self._zkey(counter_key))
            await client.zrem(zkey, _member(hold_id, cost))
            return
        holds = self._memory_holds.get(counter_key)
        if holds is not None:
            holds.pop(hold_id, None)
            if not holds:
                self._memory_holds.pop(counter_key, None)

    def _memory_place_and_total(
        self, counter_key: str, hold_id: str, cost: float
    ) -> float:
        holds = self._memory_holds.setdefault(counter_key, {})
        now = time.time()
        expired: List[str] = [hid for hid, (_, exp) in holds.items() if exp <= now]
        for hid in expired:
            del holds[hid]
        holds[hold_id] = (cost, now + self._ttl_seconds)
        return sum(c for c, _ in holds.values())
