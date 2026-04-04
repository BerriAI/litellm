from __future__ import annotations

"""Auto-queue middleware for proxy request backpressure.

Admission, queue state, and slot transfer orchestration live in Redis via Lua
scripts. This module keeps only per-request wake state in memory so the local
worker can react to distributed claims, disconnects, and shutdown.
"""
import asyncio
import heapq
import itertools
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from redis.exceptions import RedisError
from starlette.types import ASGIApp, Receive, Scope, Send

if TYPE_CHECKING:
    from litellm.proxy._types import AutoQueueModelStatus, AutoQueueStatusResponse

from .auto_queue_logging import (
    AUTOQ_METADATA_KEY,
    append_autoq_event,
    finalize_autoq_summary,
)
from .auto_queue_lease import ActiveLeaseHeartbeat
from .auto_queue_scripts import AutoQueueRedis, DistributedAutoQueueRedis
from .auto_queue_state import (
    AUTOQ_ACTIVE_KEY_PREFIX,
    AUTOQ_CEILING_KEY_PREFIX,
    AUTOQ_LIMIT_KEY_PREFIX,
    AUTOQ_QUEUE_KEY_PREFIX,
    active_lease_key,
    claim_key,
    queue_key,
    request_key,
    request_state_from_hash,
    request_state_hash,
)

logger = logging.getLogger("litellm.proxy.middleware.auto_queue")

def _resolve_worker_count() -> int:
    """Resolve the number of worker processes for the deployment.

    Prefer explicit web server envs in this order: WEB_CONCURRENCY, UVICORN_WORKERS, GUNICORN_WORKERS.
    Any non-integer or invalid value falls back to 1. Values less than 1 are treated as 1.
    This value is used to guard against multi-worker deployments when auto-queue is enabled.
    """
    candidates = ["WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"]
    worker_count = 1
    for name in candidates:
        val = os.environ.get(name)
        if val is None:
            continue
        try:
            v = int(val)
        except Exception:
            v = 1
        if v < 1:
            v = 1
        worker_count = v
        break
    return worker_count


# -- Configuration ------------------------------------------------------------

AUTOQ_ENABLED = os.environ.get("AUTOQ_ENABLED", "false").lower() in ("1", "true", "yes")
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
_KEY_TTL = 86400  # 24 hours for limit/success/ceiling keys


# -- In-Memory Priority Queue -------------------------------------------------

_id_counter = itertools.count()


@dataclass(order=True)
class _QueueEntry:
    priority: int
    seq: int  # tie-breaker for FIFO within same priority
    request_id: str = field(compare=False)
    wake_state: "_WakeState" = field(compare=False)


class ModelQueue:
    """Per-model local wake-state registry for requests waiting on Redis claims."""

    def __init__(self, max_depth: int = MAX_QUEUE_DEPTH):
        self._heap: List[_QueueEntry] = []
        self._entries: Dict[str, _QueueEntry] = {}  # request_id -> entry
        self._seq = 0
        self.max_depth = max_depth

    @property
    def depth(self) -> int:
        return len(self._entries)

    @property
    def is_full(self) -> bool:
        return self.depth >= self.max_depth

    def add(self, request_id: str, wake_state: "_WakeState", priority: int = 10) -> None:
        entry = _QueueEntry(
            priority=priority,
            seq=self._seq,
            request_id=request_id,
            wake_state=wake_state,
        )
        self._seq += 1
        heapq.heappush(self._heap, entry)
        self._entries[request_id] = entry

    def remove(self, request_id: str, *, reason: Optional[str] = None) -> None:
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            if reason is not None:
                entry.wake_state.wake(reason)
            entry.request_id = ""

    def wake_next(self) -> bool:
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.request_id and entry.request_id in self._entries:
                del self._entries[entry.request_id]
                entry.wake_state.wake(_QueueWakeReason.TRANSFERRED)
                return True
        return False

    def wake_all(self, reason: str = "shutdown") -> None:
        for entry in self._entries.values():
            entry.wake_state.wake(reason)
        self._entries.clear()
        self._heap.clear()

    def has_request(self, request_id: str) -> bool:
        return request_id in self._entries

    def count_waiters(self) -> int:
        return len(self._entries)

    def debug_snapshot(self) -> dict:
        return {"depth": self.depth, "max_depth": self.max_depth}

    def __repr__(self) -> str:
        return f"ModelQueue(depth={self.depth}, max_depth={self.max_depth})"


# -- Path Matching -------------------------------------------------------------

_QUEUED_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/chat/completions",
    "/completions",
    "/embeddings",
}


def _should_queue(scope: Scope) -> bool:
    if scope["type"] != "http":
        return False
    method = scope.get("method", "")
    path = scope.get("path", "")
    return method == "POST" and path in _QUEUED_PATHS


def _is_queue_status_request(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope.get("method", "") == "GET"
        and scope.get("path", "") == "/queue/status"
    )


# -- Body Buffering ------------------------------------------------------------

async def _buffer_body(receive: Receive) -> bytes:
    """Read the full ASGI request body, handling chunked transfer."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    return body


def _replay_receive(body: bytes, disconnect_event: Optional[asyncio.Event] = None) -> Receive:
    """Create a receive callable that replays the buffered body once.

    If *disconnect_event* is provided, the post-body ``receive()`` will return
    ``http.disconnect`` promptly when the event fires instead of sleeping.
    """
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # After body is consumed, wait for disconnect
        if disconnect_event is not None:
            await disconnect_event.wait()
            return {"type": "http.disconnect"}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    return receive


async def _send_json_error(send: Send, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [[b"content-type", b"application/json"]],
    })
    await send({"type": "http.response.body", "body": body})


async def _cancel_task(task: Optional[asyncio.Task[Any]]) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _get_key_config(scope: Scope) -> Tuple[int, int]:
    """Extract queue_timeout and priority from LiteLLM's key cache. Best-effort."""
    timeout = DEFAULT_TIMEOUT
    priority = 10
    try:
        from litellm.proxy._types import hash_token
        from litellm.proxy.proxy_server import user_api_key_cache

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            token = auth[7:]
            hashed = hash_token(token)
            cached = user_api_key_cache.get_cache(key=hashed)
            if cached and hasattr(cached, "metadata") and cached.metadata:
                meta = cached.metadata
                timeout = meta.get("queue_timeout_seconds", timeout)
                priority = meta.get("queue_priority", priority)
                logger.debug(
                    "Loaded auto-queue key config from cache",
                    extra={"queue_timeout": timeout, "queue_priority": priority},
                )
    except Exception:
        logger.debug("Failed to read key config from cache", exc_info=True)
    return timeout, priority


class _QueueWakeReason:
    TRANSFERRED = "transferred"
    SHUTDOWN = "shutdown"
    DISCONNECTED = "disconnected"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class _WakeState:
    def __init__(self) -> None:
        self.reason = _QueueWakeReason.UNKNOWN
        self.event = asyncio.Event()

    def wake(self, reason: str) -> None:
        self.reason = reason
        self.event.set()

    def clear(self) -> None:
        self.reason = _QueueWakeReason.UNKNOWN
        self.event.clear()

    @property
    def is_set(self) -> bool:
        return self.event.is_set()

    async def wait(self) -> None:
        await self.event.wait()

    def __repr__(self) -> str:
        return f"_WakeState(reason={self.reason!r}, is_set={self.is_set})"


async def _run_redis_operation(coro: Any, *, operation: str, model: str) -> Any:
    try:
        return await coro
    except RedisError:
        logger.exception(
            "Auto-queue Redis operation failed",
            extra={"operation": operation, "model": model},
        )
        raise


async def _send_redis_unavailable(send: Send, model: str) -> None:
    logger.warning("Auto-queue unavailable due to Redis error", extra={"model": model})
    await _send_json_error(send, 503, f"Auto-queue unavailable for model {model}")


def _extract_response_status(exc: Exception) -> Optional[int]:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_disconnected_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "ClientDisconnect"


def _is_signal_handler_supported_error(exc: Exception) -> bool:
    return isinstance(exc, (NotImplementedError, RuntimeError))


def _should_treat_as_redis_error(exc: Exception) -> bool:
    return isinstance(exc, RedisError)


def _describe_queue_state(model: str, queue: "ModelQueue") -> dict:
    return {"model": model, "queue_depth": queue.depth, "queue_full": queue.is_full}


# -- ASGI Middleware -----------------------------------------------------------

class AutoQueueMiddleware:
    """ASGI middleware providing per-model concurrency control and request queuing.

    Gated behind the ``AUTOQ_ENABLED`` environment variable (default: disabled).
    When disabled the middleware is a transparent passthrough.

    Redis owns the distributed queue, request state, and claim/release
    transitions. The middleware keeps a local wake registry only for queued
    requests currently attached to this worker.
    """

    def __init__(
        self,
        app: ASGIApp,
        aqr: Optional[AutoQueueRedis] = None,
        max_queue_depth: int = MAX_QUEUE_DEPTH,
        enabled: Optional[bool] = None,
    ):
        self.app = app
        self._aqr = aqr  # injected in tests; created lazily in production
        self._max_queue_depth = max_queue_depth
        self._queues: Dict[str, ModelQueue] = {}
        self._shutting_down = False
        self._enabled = enabled if enabled is not None else AUTOQ_ENABLED
        self._worker_id = f"autoq-worker:{os.getpid()}:{uuid.uuid4().hex}"

    def _ensure_aqr(self) -> AutoQueueRedis:
        """Lazy-init Redis connection on first use (production path)."""
        if self._aqr is None:
            logger.info(
                "Initializing auto-queue Redis client",
                extra={"redis_host": REDIS_HOST, "redis_port": REDIS_PORT, "redis_db": REDIS_DB},
            )
            r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            self._aqr = DistributedAutoQueueRedis(
                redis=r,
                default_max_concurrent=DEFAULT_MAX_CONCURRENT,
                ceiling=CEILING,
                scale_up_threshold=SCALE_UP_THRESHOLD,
                scale_down_step=SCALE_DOWN_STEP,
                max_queue_depth=self._max_queue_depth,
            )
        return self._aqr

    async def _safe_redis_call(self, operation: Any, *, model: str, send: Send) -> Any:
        try:
            return await operation
        except RedisError:
            await _send_redis_unavailable(send, model)
            return None

    def _get_queue(self, model: str) -> ModelQueue:
        if model not in self._queues:
            self._queues[model] = ModelQueue(max_depth=self._max_queue_depth)
        return self._queues[model]

    async def _get_status_models(self, aqr: AutoQueueRedis) -> List[str]:
        models = set(self._queues.keys())
        redis_client = getattr(aqr, "redis", None)
        if redis_client is None:
            return sorted(models)

        prefixes = (
            AUTOQ_ACTIVE_KEY_PREFIX,
            AUTOQ_LIMIT_KEY_PREFIX,
            AUTOQ_CEILING_KEY_PREFIX,
            AUTOQ_QUEUE_KEY_PREFIX,
        )
        for prefix in prefixes:
            async for raw_key in redis_client.scan_iter(match=f"{prefix}*"):
                key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
                if key.startswith(prefix):
                    models.add(key[len(prefix) :])
        return sorted(models)

    async def _load_request_state(self, aqr: AutoQueueRedis, request_id: str):
        redis_client = getattr(aqr, "redis", None)
        if redis_client is None:
            return None
        raw_state = await redis_client.hgetall(request_key(request_id))
        if not raw_state:
            return None
        return request_state_from_hash(raw_state)

    async def _finalize_locally_queued_request(
        self,
        aqr: AutoQueueRedis,
        *,
        model: str,
        request_id: str,
        terminal_state: str,
    ) -> bool:
        abandon_queued_request = getattr(aqr, "abandon_queued_request", None)
        if callable(abandon_queued_request):
            return bool(
                await abandon_queued_request(
                    model,
                    request_id,
                    terminal_state=terminal_state,
                )
            )

        redis_client = getattr(aqr, "redis", None)
        if redis_client is None:
            return False

        raw_state = await redis_client.hgetall(request_key(request_id))
        if not raw_state:
            return False

        state = request_state_from_hash(raw_state)
        if state.state != "queued":
            return False

        state.state = terminal_state
        state.claim_token = None
        state.finished_at_ms = int(time.time() * 1000)

        pipe = redis_client.pipeline()
        pipe.hset(request_key(request_id), mapping=request_state_hash(state))
        pipe.expire(request_key(request_id), _KEY_TTL)
        pipe.zrem(queue_key(model), request_id)
        pipe.delete(claim_key(request_id))
        pipe.delete(active_lease_key(request_id))
        await pipe.execute()
        return True

    def _wake_local_request(self, model: str, request_id: Optional[str]) -> bool:
        if not request_id:
            return False
        queue = self._queues.get(model)
        if queue is None or not queue.has_request(request_id):
            return False
        queue.remove(request_id, reason=_QueueWakeReason.TRANSFERRED)
        return True

    async def _wait_for_distributed_claim(
        self,
        aqr: AutoQueueRedis,
        *,
        model: str,
        request_id: str,
        queue: ModelQueue,
        wake_state: "_WakeState",
        timeout: float,
    ):
        poll_interval = 0.05
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.0)

        while True:
            if wake_state.is_set and wake_state.reason != _QueueWakeReason.TRANSFERRED:
                return None

            state = await self._load_request_state(aqr, request_id)
            if state is not None and state.state in {"claimed", "active"}:
                queue.remove(request_id)
                return state
            if state is not None and state.state != "queued":
                queue.remove(request_id)
                return state

            remaining = deadline - loop.time()
            if remaining <= 0:
                return None

            try:
                await asyncio.wait_for(wake_state.wait(), timeout=min(poll_interval, remaining))
            except asyncio.TimeoutError:
                continue

            if wake_state.reason == _QueueWakeReason.TRANSFERRED:
                continue
            return None

    async def _abandon_waiting_request(
        self,
        aqr: AutoQueueRedis,
        *,
        model: str,
        request_id: str,
        terminal_state: str,
        send: Send,
    ) -> None:
        state = await self._load_request_state(aqr, request_id)
        if state is None:
            return

        if state.state == "queued":
            finalized = await self._safe_redis_call(
                self._finalize_locally_queued_request(
                    aqr,
                    model=model,
                    request_id=request_id,
                    terminal_state=terminal_state,
                ),
                model=model,
                send=send,
            )
            if finalized is None:
                return
            if finalized:
                return
            state = await self._load_request_state(aqr, request_id)
            if state is None:
                return

        if state.state in {"claimed", "active"}:
            transfer = await self._safe_redis_call(
                aqr.release_and_claim_next(
                    model,
                    request_id,
                    terminal_state=terminal_state,
                    allow_missing_active=True,
                ),
                model=model,
                send=send,
            )
            if transfer is None:
                return
            self._wake_local_request(model, transfer.claimed_request_id)

    async def _cleanup_failed_claimed_request(
        self,
        aqr: AutoQueueRedis,
        *,
        model: str,
        request_id: str,
    ) -> None:
        try:
            transfer = await aqr.release_and_claim_next(
                model,
                request_id,
                terminal_state="cancelled",
                allow_missing_active=True,
            )
        except RedisError:
            logger.exception(
                "Failed to clean up claimed request after heartbeat startup failure",
                extra={"model": model, "request_id": request_id},
            )
            return
        self._wake_local_request(model, transfer.claimed_request_id)

    async def _process_request(
        self,
        scope: Scope,
        request_data: Dict[str, Any],
        model: str,
        send: Send,
        original_receive: Optional[Receive] = None,
    ) -> None:
        """Use distributed Redis state for admission and slot transfer orchestration."""
        aqr = self._ensure_aqr()
        request_id = f"{model}-{next(_id_counter)}-{time.monotonic_ns()}"
        queue = self._get_queue(model)
        logger.debug("Processing auto-queue request", extra={"model": model, "request_id": request_id})
        autoq_logging_data: Dict[str, Any] = {}

        if self._shutting_down:
            logger.info("Rejecting request during shutdown", extra={"model": model, "request_id": request_id})
            await _send_json_error(send, 503, "Server shutting down")
            return

        timeout, priority = _get_key_config(scope)
        timeout_seconds = max(float(timeout), 0.0)
        deadline_at_ms = int(time.time() * 1000) + int(timeout_seconds * 1000)
        finalize_autoq_summary(
            autoq_logging_data,
            {
                "request_id": request_id,
                "model": model,
                "worker_id": self._worker_id,
                "priority": priority,
                "timeout_seconds": timeout_seconds,
            },
        )
        append_autoq_event(
            autoq_logging_data,
            event="received",
            payload={
                "model": model,
                "request_id": request_id,
                "priority": priority,
                "timeout_seconds": timeout_seconds,
            },
        )

        decision = await self._safe_redis_call(
            aqr.admit_or_enqueue(
                model=model,
                request_id=request_id,
                priority=priority,
                deadline_at_ms=deadline_at_ms,
                worker_id=self._worker_id,
            ),
            model=model,
            send=send,
        )
        if decision is None:
            return
        request_state = decision.request_state
        finalize_autoq_summary(
            autoq_logging_data,
            {
                "decision": decision.decision,
                "queued": decision.decision == "queued",
            },
        )
        append_autoq_event(
            autoq_logging_data,
            event="decision",
            payload={
                "decision": decision.decision,
                "priority": priority,
                "timeout_seconds": timeout_seconds,
            },
        )

        if decision.decision == "queue_full":
            logger.warning(
                "Rejecting request because distributed queue is full",
                extra={"model": model, "request_id": request_id},
            )
            await _send_json_error(send, 503, f"Queue full for model {model}")
            return

        if decision.decision == "queued":
            wake_state = _WakeState()
            queue.add(request_id, wake_state, priority)
            append_autoq_event(
                autoq_logging_data,
                event="queued",
                payload={
                    "queue_depth": queue.depth,
                    "priority": priority,
                    "timeout_seconds": timeout_seconds,
                },
            )
            logger.info(
                "Queued auto-queue request in distributed Redis queue",
                extra={
                    "model": model,
                    "request_id": request_id,
                    "queue_priority": priority,
                    "queue_timeout": timeout_seconds,
                    "queue_depth": queue.depth,
                },
            )

            async def _watch_disconnect() -> None:
                if original_receive is None:
                    return
                try:
                    msg = await original_receive()
                    if msg.get("type") == "http.disconnect":
                        logger.info(
                            "Client disconnected while queued",
                            extra={"model": model, "request_id": request_id},
                        )
                        queue.remove(request_id, reason=_QueueWakeReason.DISCONNECTED)
                except Exception:
                    logger.debug("Queued disconnect watcher stopped", exc_info=True)

            disconnect_task = asyncio.create_task(_watch_disconnect())
            try:
                request_state = await self._wait_for_distributed_claim(
                    aqr,
                    model=model,
                    request_id=request_id,
                    queue=queue,
                    wake_state=wake_state,
                    timeout=timeout_seconds,
                )
            finally:
                await _cancel_task(disconnect_task)

            if request_state is None:
                if wake_state.reason == _QueueWakeReason.DISCONNECTED:
                    finalize_autoq_summary(
                        autoq_logging_data,
                        {"claim_state": "cancelled"},
                    )
                    append_autoq_event(
                        autoq_logging_data,
                        event="cancelled",
                        payload={"reason": _QueueWakeReason.DISCONNECTED},
                    )
                    await self._abandon_waiting_request(
                        aqr,
                        model=model,
                        request_id=request_id,
                        terminal_state="cancelled",
                        send=send,
                    )
                    return
                if wake_state.reason == _QueueWakeReason.SHUTDOWN:
                    finalize_autoq_summary(
                        autoq_logging_data,
                        {"claim_state": "cancelled"},
                    )
                    append_autoq_event(
                        autoq_logging_data,
                        event="cancelled",
                        payload={"reason": _QueueWakeReason.SHUTDOWN},
                    )
                    await self._abandon_waiting_request(
                        aqr,
                        model=model,
                        request_id=request_id,
                        terminal_state="cancelled",
                        send=send,
                    )
                    logger.info(
                        "Rejecting previously queued request after shutdown wake",
                        extra={"model": model, "request_id": request_id},
                    )
                    await _send_json_error(send, 503, "Server shutting down")
                    return

                logger.warning(
                    "Queued request timed out before distributed claim",
                    extra={"model": model, "request_id": request_id, "queue_timeout": timeout_seconds},
                )
                queue.remove(request_id, reason=_QueueWakeReason.TIMEOUT)
                finalize_autoq_summary(
                    autoq_logging_data,
                    {"claim_state": "timed_out"},
                )
                append_autoq_event(
                    autoq_logging_data,
                    event="timed_out",
                    payload={"timeout_seconds": timeout_seconds},
                )
                await self._abandon_waiting_request(
                    aqr,
                    model=model,
                    request_id=request_id,
                    terminal_state="timed_out",
                    send=send,
                )
                await _send_json_error(send, 504, f"Queue timeout after {timeout_seconds}s for model {model}")
                return

            if request_state.state not in {"claimed", "active"}:
                finalize_autoq_summary(
                    autoq_logging_data,
                    {"claim_state": request_state.state},
                )
                logger.warning(
                    "Rejecting request due to unexpected distributed queue state",
                    extra={"model": model, "request_id": request_id, "request_state": request_state.state},
                )
                status_code = 504 if request_state.state == "timed_out" else 503
                message = (
                    f"Queue timeout after {timeout_seconds}s for model {model}"
                    if status_code == 504
                    else f"Auto-queue unavailable for model {model}"
                )
                await _send_json_error(send, status_code, message)
                return

        queue_wait_ms = 0
        if (
            request_state is not None
            and request_state.claimed_at_ms is not None
            and request_state.enqueued_at_ms is not None
        ):
            queue_wait_ms = max(
                request_state.claimed_at_ms - request_state.enqueued_at_ms,
                0,
            )
        finalize_autoq_summary(
            autoq_logging_data,
            {
                "claim_state": request_state.state if request_state is not None else "unknown",
                "queue_wait_ms": queue_wait_ms,
            },
        )
        append_autoq_event(
            autoq_logging_data,
            event="claim_acquired",
            payload={
                "claim_state": request_state.state if request_state is not None else "unknown",
                "queue_wait_ms": queue_wait_ms,
            },
        )

        heartbeat: Optional[ActiveLeaseHeartbeat] = None
        redis_client = getattr(aqr, "redis", None)
        if redis_client is not None and request_state is not None and request_state.claim_token:
            heartbeat = ActiveLeaseHeartbeat(
                redis_client,
                request_id=request_id,
                worker_id=self._worker_id,
                claim_token=request_state.claim_token,
            )
            heartbeat_started = await self._safe_redis_call(
                heartbeat.start(),
                model=model,
                send=send,
            )
            if heartbeat_started is None:
                await self._cleanup_failed_claimed_request(
                    aqr,
                    model=model,
                    request_id=request_id,
                )
                return
            if heartbeat_started is False:
                await self._cleanup_failed_claimed_request(
                    aqr,
                    model=model,
                    request_id=request_id,
                )
                logger.warning(
                    "Rejecting request because active lease heartbeat could not start",
                    extra={"model": model, "request_id": request_id},
                )
                await _send_json_error(send, 503, f"Auto-queue unavailable for model {model}")
                return

        logger.debug("Forwarding request with acquired auto-queue slot", extra={"model": model, "request_id": request_id})
        append_autoq_event(
            autoq_logging_data,
            event="forwarded",
            payload={
                "claim_state": request_state.state if request_state is not None else "unknown",
                "queued": decision.decision == "queued",
            },
        )
        autoq_metadata = autoq_logging_data.get(AUTOQ_METADATA_KEY)
        if isinstance(autoq_metadata, dict):
            scope.setdefault("state", {})[AUTOQ_METADATA_KEY] = autoq_metadata
        body = json.dumps(request_data).encode()
        scope["parsed_body"] = (tuple(request_data.keys()), request_data)
        response_status = 0
        disconnect_event = asyncio.Event()

        async def _watch_client_disconnect() -> None:
            if original_receive is None:
                return
            try:
                msg = await original_receive()
                if msg.get("type") == "http.disconnect":
                    logger.info(
                        "Client disconnected during active request",
                        extra={"model": model, "request_id": request_id},
                    )
                    disconnect_event.set()
            except Exception:
                logger.debug("Active disconnect watcher stopped", exc_info=True)

        client_disconnect_task = asyncio.create_task(_watch_client_disconnect())

        async def send_wrapper(message: dict) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, _replay_receive(body, disconnect_event), send_wrapper)
        except Exception as exc:
            status_code = _extract_response_status(exc)
            if status_code is not None:
                response_status = status_code
            raise
        finally:
            await _cancel_task(client_disconnect_task)
            if heartbeat is not None:
                await heartbeat.stop()

            transfer = await self._safe_redis_call(
                aqr.release_and_claim_next(model, request_id),
                model=model,
                send=send,
            )
            if transfer is None:
                return

            if response_status == 429:
                scale_down = await self._safe_redis_call(aqr.on_429(model), model=model, send=send)
                if scale_down is None:
                    return
                logger.info("Scaled down auto-queue limit after 429", extra={"model": model, "request_id": request_id})
            elif 200 <= response_status < 400:
                scale_up = await self._safe_redis_call(aqr.on_success(model), model=model, send=send)
                if scale_up is None:
                    return
                logger.debug("Recorded auto-queue success", extra={"model": model, "request_id": request_id})

            if transfer.claimed_request_id is not None:
                woke = self._wake_local_request(model, transfer.claimed_request_id)
                logger.debug(
                    "Transferred slot using distributed claim result",
                    extra={
                        "model": model,
                        "request_id": request_id,
                        "claimed_request_id": transfer.claimed_request_id,
                        "woke_waiter": woke,
                        "queue_depth": queue.depth,
                    },
                )
            else:
                logger.debug(
                    "Released auto-queue slot without waiter transfer",
                    extra={"model": model, "request_id": request_id, "queue_depth": queue.depth},
                )

            logger.debug(
                "Completed auto-queue request",
                extra={"model": model, "request_id": request_id, "response_status": response_status},
            )

    async def _drain_shutdown_queue(self) -> None:
        for model, q in self._queues.items():
            if q.depth > 0:
                logger.info("Waking queued requests for shutdown", extra={"model": model, "queue_depth": q.depth})
            q.wake_all()

    async def _shutdown_async(self) -> None:
        self._shutting_down = True
        await self._drain_shutdown_queue()

    def shutdown(self) -> None:
        """Trigger graceful shutdown -- reject new requests, wake all queued."""
        logger.info("Auto-queue middleware entering shutdown")
        self._shutting_down = True
        for q in self._queues.values():
            q.wake_all()

    def register_signal_handlers(self) -> None:
        """Register SIGTERM handler. Call once after event loop is running."""
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGTERM, self.shutdown)
        except Exception as exc:
            if _is_signal_handler_supported_error(exc):
                logger.warning("Signal handlers are not available for auto-queue middleware", exc_info=True)
                return
            raise

    async def _handle_status(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Respond to GET /queue/status with model queue info."""
        models_info: Dict[str, "AutoQueueModelStatus"] = {}
        if self._enabled:
            aqr = self._ensure_aqr()
            for model in await self._get_status_models(aqr):
                info = await aqr.get_model_info(model)
                info["local_waiters"] = self._queues.get(model).depth if model in self._queues else 0
                models_info[model] = info
        response_payload: "AutoQueueStatusResponse" = {"models": models_info}
        body = json.dumps(response_payload).encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._enabled:
            await self.app(scope, receive, send)
            return

        if _is_queue_status_request(scope):
            await self._handle_status(scope, receive, send)
            return

        if not _should_queue(scope):
            await self.app(scope, receive, send)
            return

        body = await _buffer_body(receive)
        try:
            data = json.loads(body)
            model = data.get("model", "unknown")
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self.app(scope, _replay_receive(body), send)
            return

        try:
            await self._process_request(
                scope,
                data,
                model,
                send,
                original_receive=receive,
            )
        except RedisError:
            await _send_redis_unavailable(send, model)
        except Exception as exc:
            if _is_disconnected_error(exc):
                logger.info("Client disconnected before response completed", extra={"model": model})
                return
            raise
