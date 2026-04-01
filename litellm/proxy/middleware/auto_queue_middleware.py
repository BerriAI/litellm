"""litellm/proxy/middleware/auto_queue_middleware.py"""
import asyncio
import heapq
import itertools
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis
from starlette.types import ASGIApp, Receive, Scope, Send

try:
    from litellm.proxy._types import hash_token
    from litellm.proxy.proxy_server import user_api_key_cache

    _HAS_LITELLM_INTERNALS = True
except Exception:  # noqa: BLE001 – optional at import time
    _HAS_LITELLM_INTERNALS = False

logger = logging.getLogger("litellm.proxy.middleware.auto_queue")

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
    local new = redis.call('DECR', KEYS[1])
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
    return math.max(0, new)
end
return 0
"""

_LUA_RELEASE_AND_TRANSFER = """
-- Atomically release a slot and, if waiters exist, immediately re-acquire
-- for the next waiter so no new arrival can steal it.
local active = tonumber(redis.call('GET', KEYS[1])) or 0
if active <= 0 then
    return 0
end
local has_waiters = tonumber(ARGV[1])
if has_waiters == 1 then
    -- Transfer: keep active count the same (release + acquire cancel out)
    return 1
end
-- No waiters: release normally
local new = redis.call('DECR', KEYS[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 0
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
        self._release_transfer_script = redis.register_script(_LUA_RELEASE_AND_TRANSFER)
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
            args=[ACTIVE_KEY_TTL],
        )
        return int(result)

    async def release_and_transfer(self, model: str, has_waiters: bool) -> bool:
        """Atomically release a slot and re-acquire for a waiter if one exists.

        Returns True if a slot was transferred (caller should wake next waiter).
        Returns False if the slot was simply released.
        """
        result = await self._release_transfer_script(
            keys=[f"autoq:active:{model}"],
            args=[1 if has_waiters else 0, ACTIVE_KEY_TTL],
        )
        return bool(result)

    async def on_success(self, model: str) -> None:
        await self._scale_up_script(
            keys=[
                f"autoq:success:{model}",
                f"autoq:limit:{model}",
                f"autoq:ceiling:{model}",
            ],
            args=[self.scale_up_threshold, self.ceiling, self.default_max_concurrent, _KEY_TTL],
        )

    async def on_429(self, model: str) -> None:
        await self._scale_down_script(
            keys=[f"autoq:limit:{model}", f"autoq:success:{model}"],
            args=[self.default_max_concurrent, self.scale_down_step, _KEY_TTL],
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


# -- In-Memory Priority Queue -------------------------------------------------

_id_counter = itertools.count()


@dataclass(order=True)
class _QueueEntry:
    priority: int
    seq: int  # tie-breaker for FIFO within same priority
    request_id: str = field(compare=False)
    event: asyncio.Event = field(compare=False)


class ModelQueue:
    """Per-model priority queue of waiting requests."""

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

    def add(self, request_id: str, event: asyncio.Event, priority: int = 10) -> None:
        entry = _QueueEntry(priority=priority, seq=self._seq, request_id=request_id, event=event)
        self._seq += 1
        heapq.heappush(self._heap, entry)
        self._entries[request_id] = entry

    def remove(self, request_id: str) -> None:
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            # Lazy deletion -- mark as removed, skip during wake_next
            entry.request_id = ""

    def wake_next(self) -> bool:
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.request_id and entry.request_id in self._entries:
                del self._entries[entry.request_id]
                entry.event.set()
                return True
        return False

    def wake_all(self) -> None:
        for entry in self._entries.values():
            entry.event.set()
        self._entries.clear()
        self._heap.clear()


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


def _get_key_config(scope: Scope) -> Tuple[int, int]:
    """Extract queue_timeout and priority from LiteLLM's key cache. Best-effort."""
    timeout = DEFAULT_TIMEOUT
    priority = 10
    if not _HAS_LITELLM_INTERNALS:
        return timeout, priority
    try:
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
    except Exception:
        logger.debug("Failed to read key config from cache", exc_info=True)
    return timeout, priority


# -- ASGI Middleware -----------------------------------------------------------

class AutoQueueMiddleware:
    """ASGI middleware providing per-model concurrency control and request queuing.

    Gated behind the ``AUTOQ_ENABLED`` environment variable (default: disabled).
    When disabled the middleware is a transparent passthrough.
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

    def _ensure_aqr(self) -> AutoQueueRedis:
        """Lazy-init Redis connection on first use (production path)."""
        if self._aqr is None:
            r = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
            self._aqr = AutoQueueRedis(redis=r)
        return self._aqr

    def _get_queue(self, model: str) -> ModelQueue:
        if model not in self._queues:
            self._queues[model] = ModelQueue(max_depth=self._max_queue_depth)
        return self._queues[model]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._enabled:
            await self.app(scope, receive, send)
            return

        if not _should_queue(scope):
            await self.app(scope, receive, send)
            return

        # Buffer body and extract model
        body = await _buffer_body(receive)
        try:
            data = json.loads(body)
            model = data.get("model", "unknown")
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON -- pass through without queuing
            await self.app(scope, _replay_receive(body), send)
            return

        await self._process_request(scope, body, model, send, original_receive=receive)

    async def _process_request(
        self, scope: Scope, body: bytes, model: str, send: Send,
        original_receive: Optional[Receive] = None,
    ) -> None:
        """Acquire slot (or queue), forward request, release slot, auto-scale."""
        aqr = self._ensure_aqr()
        request_id = f"{model}-{next(_id_counter)}-{time.monotonic_ns()}"
        queue = self._get_queue(model)

        # Try to acquire a slot
        acquired = await aqr.try_acquire(model)
        if not acquired:
            # Check shutdown
            if self._shutting_down:
                await _send_json_error(send, 503, "Server shutting down")
                return

            # Check queue depth
            if queue.is_full:
                await _send_json_error(send, 503, f"Queue full for model {model}")
                return

            # Read per-key timeout/priority (best-effort from LiteLLM cache)
            timeout, priority = _get_key_config(scope)

            # Enqueue and wait with disconnect detection
            event = asyncio.Event()
            disconnected = False
            queue.add(request_id, event, priority)

            async def _watch_disconnect() -> None:
                nonlocal disconnected
                if original_receive is None:
                    return
                try:
                    msg = await original_receive()
                    if msg.get("type") == "http.disconnect":
                        disconnected = True
                        queue.remove(request_id)
                        event.set()
                except Exception:
                    pass

            disconnect_task = asyncio.create_task(_watch_disconnect())

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                queue.remove(request_id)
                disconnect_task.cancel()
                await _send_json_error(
                    send, 504, f"Queue timeout after {timeout}s for model {model}"
                )
                return

            disconnect_task.cancel()

            # If woken by disconnect, exit silently
            if disconnected:
                return

            # Slot was pre-acquired for us by release_and_transfer()

        # Slot acquired -- forward request and intercept response status
        response_status = 0
        disconnect_event = asyncio.Event()

        async def _watch_client_disconnect() -> None:
            """Detect client disconnect during active request processing."""
            if original_receive is None:
                return
            try:
                msg = await original_receive()
                if msg.get("type") == "http.disconnect":
                    disconnect_event.set()
            except Exception:
                pass

        client_disconnect_task = asyncio.create_task(_watch_client_disconnect())

        async def send_wrapper(message: dict) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, _replay_receive(body, disconnect_event), send_wrapper)
        finally:
            client_disconnect_task.cancel()

            # Atomically release slot and transfer to next waiter if one exists
            transferred = await aqr.release_and_transfer(model, has_waiters=queue.depth > 0)

            # Auto-scale based on response
            if response_status == 429:
                await aqr.on_429(model)
            elif 200 <= response_status < 400:
                await aqr.on_success(model)

            if transferred:
                queue.wake_next()

    def shutdown(self) -> None:
        """Trigger graceful shutdown -- reject new requests, wake all queued."""
        self._shutting_down = True
        for q in self._queues.values():
            q.wake_all()

    def register_signal_handlers(self) -> None:
        """Register SIGTERM handler. Call once after event loop is running."""
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGTERM, self.shutdown)

    async def _handle_status(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Respond to GET /queue/status with model queue info."""
        models_info: Dict[str, Any] = {}
        if self._aqr:
            for model, q in self._queues.items():
                info = await self._aqr.get_model_info(model)
                info["queued"] = q.depth
                models_info[model] = info
        body = json.dumps({"models": models_info}).encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"application/json"]],
        })
        await send({"type": "http.response.body", "body": body})
