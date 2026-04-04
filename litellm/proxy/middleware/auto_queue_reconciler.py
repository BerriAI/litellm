from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

import redis.asyncio as aioredis

from .auto_queue_scripts import DEFAULT_LEASE_TTL, DistributedAutoQueueRedis
from .auto_queue_state import (
    AUTOQ_REQUEST_KEY_PREFIX,
    active_lease_key,
    request_state_from_hash,
)

logger = logging.getLogger("litellm.proxy.middleware.auto_queue.reconciler")

DEFAULT_MAX_CONCURRENT = int(os.environ.get("AUTOQ_DEFAULT_MAX_CONCURRENT", "20"))
SCALE_UP_THRESHOLD = int(os.environ.get("AUTOQ_SCALE_UP_THRESHOLD", "20"))
SCALE_DOWN_STEP = int(os.environ.get("AUTOQ_SCALE_DOWN_STEP", "1"))
CEILING = int(os.environ.get("AUTOQ_CEILING", "50"))
MAX_QUEUE_DEPTH = int(os.environ.get("AUTOQ_MAX_QUEUE_DEPTH", "100"))
REDIS_HOST = os.environ.get("AUTOQ_REDIS_HOST", os.environ.get("REDIS_HOST", "localhost"))
REDIS_PORT = int(os.environ.get("AUTOQ_REDIS_PORT", os.environ.get("REDIS_PORT", "6379")))
REDIS_DB = int(os.environ.get("AUTOQ_REDIS_DB", "3"))
DEFAULT_RECONCILE_INTERVAL_SECONDS = float(
    os.environ.get("AUTOQ_RECONCILE_INTERVAL_SECONDS", str(max(DEFAULT_LEASE_TTL / 2, 5)))
)
DEFAULT_LOCK_TTL_SECONDS = int(
    os.environ.get("AUTOQ_RECONCILE_LOCK_TTL_SECONDS", str(DEFAULT_LEASE_TTL))
)
RECONCILE_LOCK_KEY = os.environ.get("AUTOQ_RECONCILE_LOCK_KEY", "autoq:reconciler:lock")


def build_auto_queue_reconciler() -> "AutoQueueReconciler":
    redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    aqr = DistributedAutoQueueRedis(
        redis=redis_client,
        default_max_concurrent=DEFAULT_MAX_CONCURRENT,
        ceiling=CEILING,
        scale_up_threshold=SCALE_UP_THRESHOLD,
        scale_down_step=SCALE_DOWN_STEP,
        max_queue_depth=MAX_QUEUE_DEPTH,
    )
    return AutoQueueReconciler(
        aqr=aqr,
        interval_seconds=DEFAULT_RECONCILE_INTERVAL_SECONDS,
        lock_ttl_seconds=DEFAULT_LOCK_TTL_SECONDS,
        close_redis_on_stop=True,
    )


class AutoQueueReconciler:
    """Recover slots held by requests whose active lease has gone stale."""

    def __init__(
        self,
        aqr: DistributedAutoQueueRedis,
        *,
        interval_seconds: float = DEFAULT_RECONCILE_INTERVAL_SECONDS,
        lock_ttl_seconds: int = DEFAULT_LOCK_TTL_SECONDS,
        lock_key: str = RECONCILE_LOCK_KEY,
        close_redis_on_stop: bool = False,
        stale_terminal_state: str = "cancelled",
    ) -> None:
        self.aqr = aqr
        self.redis = aqr.redis
        self.interval_seconds = interval_seconds
        self.lock_ttl_seconds = lock_ttl_seconds
        self.lock_key = lock_key
        self.close_redis_on_stop = close_redis_on_stop
        self.stale_terminal_state = stale_terminal_state
        self.owner_id = f"autoq-reconciler:{os.getpid()}:{uuid.uuid4().hex}"
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name=f"autoq-reconciler:{self.owner_id}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

        if self.close_redis_on_stop:
            close = getattr(self.redis, "aclose", None)
            if callable(close):
                await close()

    async def reconcile_once(self) -> int:
        lock_acquired = await self.redis.set(
            self.lock_key,
            self.owner_id,
            ex=self.lock_ttl_seconds,
            nx=True,
        )
        if not lock_acquired:
            return 0

        try:
            return await self._reconcile_once_locked()
        finally:
            lock_owner = await self.redis.get(self.lock_key)
            if lock_owner is not None and lock_owner.decode() == self.owner_id:
                await self.redis.delete(self.lock_key)

    async def _reconcile_once_locked(self) -> int:
        reconciled = 0
        async for raw_key in self.redis.scan_iter(match=f"{AUTOQ_REQUEST_KEY_PREFIX}*"):
            key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            raw_state = await self.redis.hgetall(key)
            if not raw_state:
                continue

            state = request_state_from_hash(raw_state)
            if state.state not in {"claimed", "active"}:
                continue
            if not state.request_id or not state.model or not state.claim_token:
                continue

            raw_lease = await self.redis.hgetall(active_lease_key(state.request_id))
            if raw_lease:
                lease_claim_token = raw_lease.get(b"claim_token") or raw_lease.get("claim_token")
                if isinstance(lease_claim_token, bytes):
                    lease_claim_token = lease_claim_token.decode()
                if lease_claim_token == state.claim_token:
                    continue

            await self.aqr.release_and_claim_next(
                state.model,
                state.request_id,
                terminal_state=self.stale_terminal_state,
                allow_missing_active=True,
            )
            reconciled += 1
            logger.warning(
                "Recovered stale auto-queue active lease",
                extra={"model": state.model, "request_id": state.request_id},
            )

        return reconciled

    async def _run(self) -> None:
        try:
            while True:
                await self.reconcile_once()
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Auto-queue reconciler crashed")
