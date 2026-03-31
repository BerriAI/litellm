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


# -- In-Memory Priority Queue -------------------------------------------------

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


def _replay_receive(body: bytes) -> Receive:
    """Create a receive callable that replays the buffered body once."""
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # After body is consumed, wait for disconnect
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
    try:
        from litellm.proxy.proxy_server import user_api_key_cache
        from litellm.proxy._types import hash_token

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
        pass  # Fall back to defaults if LiteLLM internals aren't available
    return timeout, priority


# -- ASGI Middleware -----------------------------------------------------------

class AutoQueueMiddleware:
    """ASGI middleware providing per-model concurrency control and request queuing."""

    def __init__(
        self,
        app: ASGIApp,
        aqr: Optional[AutoQueueRedis] = None,
        max_queue_depth: int = MAX_QUEUE_DEPTH,
    ):
        self.app = app
        self._aqr = aqr  # injected in tests; created lazily in production
        self._max_queue_depth = max_queue_depth
        self._queues: Dict[str, ModelQueue] = {}
        self._shutting_down = False

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
        # Handle /queue/status
        if scope["type"] == "http" and scope.get("path") == "/queue/status" and scope.get("method") == "GET":
            await self._handle_status(scope, receive, send)
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

        # Pass through with replayed body (queuing added in Task 4)
        await self._process_request(scope, body, model, send, original_receive=receive)

    async def _process_request(
        self, scope: Scope, body: bytes, model: str, send: Send,
        original_receive: Optional[Receive] = None,
    ) -> None:
        """Acquire slot (or queue), forward request, release slot, auto-scale."""
        aqr = self._ensure_aqr()
        request_id = f"{model}-{time.monotonic_ns()}"
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

            async def _watch_disconnect():
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

            # Slot was pre-acquired for us by release_and_wake_next()

        # Slot acquired -- forward request and intercept response status
        response_status = 0

        async def send_wrapper(message: dict) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, _replay_receive(body), send_wrapper)
        finally:
            # Release slot and wake next queued request
            await aqr.release(model)

            # Auto-scale based on response
            if response_status == 429:
                await aqr.on_429(model)
            elif 200 <= response_status < 400:
                await aqr.on_success(model)

            # Try to acquire for next waiter, then wake them
            if queue.depth > 0:
                re_acquired = await aqr.try_acquire(model)
                if re_acquired:
                    queue.wake_next()
                # If re_acquired fails, waiter stays queued until another release

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
