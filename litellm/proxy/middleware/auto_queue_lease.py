from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import redis.asyncio as aioredis

from .auto_queue_scripts import DEFAULT_DATA_TTL, DEFAULT_LEASE_TTL
from .auto_queue_state import active_lease_key, request_key, request_state_from_hash

logger = logging.getLogger("litellm.proxy.middleware.auto_queue.lease")


def _now_ms() -> int:
    return int(time.time() * 1000)


class ActiveLeaseHeartbeat:
    """Refresh an active lease while a claimed request is executing.

    The first refresh also promotes a claimed request into the active state so
    Redis reflects that the worker has begun real downstream execution.
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        *,
        request_id: str,
        worker_id: str,
        claim_token: str,
        lease_ttl_seconds: int = DEFAULT_LEASE_TTL,
        interval_seconds: Optional[float] = None,
    ) -> None:
        self.redis = redis
        self.request_id = request_id
        self.worker_id = worker_id
        self.claim_token = claim_token
        self.lease_ttl_seconds = lease_ttl_seconds
        self.interval_seconds = interval_seconds or max(1.0, lease_ttl_seconds / 3)
        self._task: Optional[asyncio.Task[None]] = None

    async def refresh_once(self) -> bool:
        raw_state = await self.redis.hgetall(request_key(self.request_id))
        if not raw_state:
            return False

        state = request_state_from_hash(raw_state)
        if state.claim_token != self.claim_token or state.state not in {"claimed", "active"}:
            return False

        now_ms = _now_ms()
        pipe = self.redis.pipeline()

        request_updates: dict[str, str] = {}
        if state.state != "active":
            request_updates["state"] = "active"
        if state.started_at_ms is None:
            request_updates["started_at_ms"] = str(now_ms)
        if request_updates:
            pipe.hset(request_key(self.request_id), mapping=request_updates)
            pipe.expire(request_key(self.request_id), DEFAULT_DATA_TTL)

        pipe.hset(
            active_lease_key(self.request_id),
            mapping={
                "worker_id": self.worker_id,
                "claim_token": self.claim_token,
                "heartbeat_at_ms": str(now_ms),
            },
        )
        pipe.expire(active_lease_key(self.request_id), self.lease_ttl_seconds)
        await pipe.execute()
        return True

    async def start(self) -> bool:
        if self._task is not None and not self._task.done():
            return True

        refreshed = await self.refresh_once()
        if not refreshed:
            return False

        self._task = asyncio.create_task(
            self._run(),
            name=f"autoq-heartbeat:{self.request_id}",
        )
        return True

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_seconds)
                refreshed = await self.refresh_once()
                if not refreshed:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Active lease heartbeat crashed",
                extra={"request_id": self.request_id, "worker_id": self.worker_id},
            )
