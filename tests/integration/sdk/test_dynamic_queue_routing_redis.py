import asyncio
import json
import os
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Final

import pytest
import pytest_asyncio
from integration._support.client import eventually
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from redis import Redis

import litellm
from litellm.router_utils.dynamic_queue_redis import _keys
from litellm.router_utils.dynamic_queue_router import (
    QUEUE_FULL_CODE,
    DynamicQueueFullError,
    DynamicQueueTimeoutError,
)

CHAT_PATH: Final = "/v1/chat/completions"
MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _rate_limited(retry_after_s: int | None) -> Reply:
    return Reply(
        status=429,
        body=json.dumps({"error": {"message": "rate limited", "type": "rate_limit_error"}}).encode(),
        headers={} if retry_after_s is None else {"retry-after": str(retry_after_s)},
    )


def _unavailable() -> Reply:
    return Reply(
        status=503,
        body=json.dumps({"error": {"message": "temporarily unavailable", "type": "server_error"}}).encode(),
    )


@dataclass(frozen=True, slots=True)
class Upstream:
    wire: Wire
    replies: SimpleQueue[Reply]

    def chat_requests(self) -> int:
        """The chat completion requests the upstream received so far, read once."""
        return sum(1 for request in self.wire.drain() if request.target.startswith(CHAT_PATH))


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    replies: Final[SimpleQueue[Reply]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        try:
            return replies.get_nowait()
        except Empty:
            return chat_reply(f"chatcmpl-{uuid.uuid4()}", "gpt-x", "served", stream=False)

    with wire_server(respond) as wire:
        yield Upstream(wire, replies)


class RouterFactory:
    """Builds one proxy worker's router per call, each on its own named Redis connection, and closes them all."""

    def __init__(self, cleanup: AsyncExitStack, upstream: Upstream) -> None:
        self._cleanup = cleanup
        self._upstream = upstream

    def __call__(self, capacity: str, max_parallel_requests: int | None = None, **queue: object) -> litellm.Router:
        slots: Final = {} if max_parallel_requests is None else {"max_parallel_requests": max_parallel_requests}
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "group",
                    "litellm_params": {
                        "model": "openai/gpt-x",
                        "api_base": f"{self._upstream.wire.url}/v1",
                        "api_key": "integration-key",
                        "max_retries": 0,
                        **slots,
                    },
                    "model_info": {"id": capacity},
                }
            ],
            redis_host=os.environ["REDIS_HOST"],
            redis_port=int(os.environ["REDIS_PORT"]),
            cache_kwargs={"client_name": f"dynamic-queue-routing-test-{uuid.uuid4()}"},
            default_priority=0,
            num_retries=0,
            cooldown_time=1,
            dynamic_queue={"enabled": True, "max_waiting": 8, **queue},
        )
        cache: Final = router.health().store.redis_cache
        assert cache is not None
        self._cleanup.push_async_callback(cache.disconnect)
        client: Final = cache.init_async_client()
        self._cleanup.push_async_callback(client.aclose)
        self._cleanup.callback(router.retire)
        return router


@pytest_asyncio.fixture(loop_scope="function")
async def routers(upstream: Upstream) -> AsyncIterator[RouterFactory]:
    async with AsyncExitStack() as cleanup:
        yield RouterFactory(cleanup, upstream)


def _queued(capacity: str) -> int:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as redis:
        count: Final = redis.zcard(_keys(f"deployment:{capacity}")[1])
    assert isinstance(count, int)
    return count


@pytest.fixture
def capacity() -> Iterator[str]:
    capacity_id: Final = f"dynamic-queue-{uuid.uuid4()}"
    try:
        yield capacity_id
    finally:
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as client:
            _ = client.delete(*_keys(f"deployment:{capacity_id}"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "held_back"), [(429, True), (503, False)])
async def test_a_provider_429_holds_back_another_workers_request_and_an_overload_503_does_not(
    upstream: Upstream, routers: RouterFactory, capacity: str, status: int, held_back: bool
) -> None:
    first: Final = routers(capacity)
    second: Final = routers(capacity, queue_timeout=0.5)
    upstream.replies.put(_rate_limited(retry_after_s=30) if status == 429 else _unavailable())

    with pytest.raises(litellm.ServiceUnavailableError if status == 503 else litellm.RateLimitError) as failed:
        _ = await first.acompletion(model="group", messages=MESSAGES)
    assert failed.value.status_code == status

    if held_back:
        with pytest.raises(DynamicQueueTimeoutError):
            _ = await second.acompletion(model="group", messages=MESSAGES)
        assert upstream.chat_requests() == 1
    else:
        served = await second.acompletion(model="group", messages=MESSAGES)
        assert "served" in str(served)
        assert upstream.chat_requests() == 2


@pytest.mark.asyncio
async def test_a_congested_capacity_serves_a_waiting_request_once_its_probe_succeeds(
    upstream: Upstream, routers: RouterFactory, capacity: str
) -> None:
    first: Final = routers(capacity)
    second: Final = routers(capacity, queue_timeout=30)
    upstream.replies.put(_rate_limited(retry_after_s=None))
    with pytest.raises(litellm.RateLimitError):
        _ = await first.acompletion(model="group", messages=MESSAGES)

    served: Final = await second.acompletion(model="group", messages=MESSAGES)

    assert "served" in str(served)
    assert upstream.chat_requests() == 2


@pytest.mark.asyncio
async def test_a_rate_limit_litellm_raises_itself_for_a_full_deployment_does_not_congest_it(
    upstream: Upstream, routers: RouterFactory, capacity: str
) -> None:
    first: Final = routers(capacity, max_parallel_requests=1)
    second: Final = routers(capacity, queue_timeout=0.5)
    released: Final = threading.Event()
    upstream.replies.put(
        Reply(chunks=(chat_reply("chatcmpl-held", "gpt-x", "held", stream=False).body,), gate_before_first=released)
    )
    holder: Final = asyncio.create_task(first.acompletion(model="group", messages=MESSAGES))
    try:
        _ = await asyncio.to_thread(eventually, lambda: upstream.wire.received.qsize(), lambda count: count >= 1)
        with pytest.raises(litellm.RateLimitError):
            _ = await first.acompletion(model="group", messages=MESSAGES)
        released.set()
        assert "held" in str(await holder)

        served: Final = await second.acompletion(model="group", messages=MESSAGES)
    finally:
        released.set()
        _ = holder.cancel()
        _ = await asyncio.gather(holder, return_exceptions=True)

    assert "served" in str(served)


@pytest.mark.asyncio
async def test_a_full_queue_refuses_with_503_queue_full_and_a_cancelled_waiter_leaves_it(
    upstream: Upstream, routers: RouterFactory, capacity: str
) -> None:
    first: Final = routers(capacity, max_waiting=1)
    waiting_worker: Final = routers(capacity, max_waiting=1, queue_timeout=30)
    refused_worker: Final = routers(capacity, max_waiting=1, queue_timeout=30)
    upstream.replies.put(_rate_limited(retry_after_s=30))
    with pytest.raises(litellm.RateLimitError):
        _ = await first.acompletion(model="group", messages=MESSAGES)
    waiter: Final = asyncio.create_task(waiting_worker.acompletion(model="group", messages=MESSAGES))
    try:
        _ = await asyncio.to_thread(eventually, lambda: _queued(capacity), lambda count: count == 1)

        with pytest.raises(DynamicQueueFullError) as full:
            _ = await refused_worker.acompletion(model="group", messages=MESSAGES)
    finally:
        _ = waiter.cancel()
        _ = await asyncio.gather(waiter, return_exceptions=True)

    assert (full.value.status_code, full.value.code) == (503, QUEUE_FULL_CODE)
    assert upstream.chat_requests() == 1
    assert await asyncio.to_thread(eventually, lambda: _queued(capacity), lambda count: count == 0) == 0
