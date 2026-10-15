import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.messages.data import MESSAGES_EVENTS, MESSAGES_RESPONSE, REQUEST, call as route_call
import redis

from tests.test_litellm_rust.support.caching import (
    CallerCache,
    FailingDecisionCache,
    HeldWriteCache,
    assert_cache_hit_keeps_budget_spend_and_counts_usage,
    redis_cache,
    settle_cache_writes,
    use_cache,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging, request_body
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.results import cache_key, stored_cache_key
from tests.test_litellm_rust.support.routing import CALL_ID, assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])
STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


def body(response: object) -> dict[str, object]:
    return {name: value for name, value in _OBJECT.validate_python(response).items() if name != "_hidden_params"}


async def collect(stream: object) -> bytes:
    assert isinstance(stream, AsyncIterator)
    chunks: Final = [chunk async for chunk in stream]
    await settle_cache_writes()
    assert all(isinstance(chunk, bytes) for chunk in chunks)
    return b"".join(TypeAdapter(list[bytes]).validate_python(chunks))


async def call_python(server: RecordingServer, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> object:
    with monkeypatch.context() as patched:
        patched.setenv("LITELLM_RUST", "0")
        response: Final = await litellm.anthropic.messages.acreate(
            **REQUEST, api_key="test-key", api_base=server.base_url, litellm_call_id=CALL_ID, **kwargs
        )
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    return response


async def call(server: RecordingServer, **kwargs: object) -> object:
    response: Final = await route_call(server, **kwargs)
    await settle_cache_writes()
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_a_repeated_native_call_is_answered_from_the_python_cache(
    server: RecordingServer, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch)

    first: Final = await call(server, asynchronous=asynchronous)
    second: Final = await call(server, asynchronous=asynchronous)

    assert_served_natively(server, first)
    assert cache_key(first) is None
    assert isinstance(cache_key(second), str)
    assert body(second) == body(first) == MESSAGES_RESPONSE


@pytest.mark.asyncio
async def test_a_completed_native_stream_is_replayed_from_the_python_cache(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch)
    server.default_response = STREAM

    first_stream: Final = await call(server, stream=True)
    assert cache_key(first_stream) is None
    first: Final = await collect(first_stream)
    second_stream: Final = await call(server, stream=True)
    assert isinstance(cache_key(second_stream), str)
    second: Final = await collect(second_stream)

    assert len(server.requests) == 1
    assert first == b"".join(STREAM.payloads())
    assert second == first


@pytest.mark.asyncio
async def test_a_stream_closed_early_is_not_stored(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    use_cache(monkeypatch)
    server.default_response = STREAM
    server.expected_requests = 2

    stream: Final = await call(server, stream=True)
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()
    await collect(await call(server, stream=True))

    assert len(server.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ("python", "native"))
@pytest.mark.parametrize("stream", (False, True), ids=("response", "stream"))
async def test_native_and_python_calls_share_python_cache_entries(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    writer: Literal["python", "native"],
    stream: bool,
) -> None:
    use_cache(monkeypatch)
    if stream:
        server.default_response = STREAM

    async def native() -> object:
        return await call(server, stream=stream)

    async def python() -> object:
        return await call_python(server, monkeypatch, stream=stream)

    first: Final = await (python() if writer == "python" else native())
    first_payload: Final = await collect(first) if stream else body(first)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    second: Final = await (native() if writer == "python" else python())
    second_payload: Final = await collect(second) if stream else body(second)

    assert len(server.requests) == 1
    assert second_payload == first_payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    (
        ({"cache": {"no-store": True}}, {}),
        ({}, {"cache": {"no-cache": True}}),
        ({}, {"caching": False}),
        ({}, {"messages": [{"role": "user", "content": "Goodbye"}]}),
        ({}, {"system": "Be brief"}),
        ({"cache": {"namespace": "first"}}, {"cache": {"namespace": "second"}}),
        ({"metadata": {"redis_namespace": "first"}}, {"metadata": {"redis_namespace": "second"}}),
    ),
    ids=(
        "no-store",
        "no-cache",
        "caching-off-read",
        "other-messages",
        "other-param",
        "cache-namespace",
        "metadata-namespace",
    ),
)
async def test_cache_controls_and_keys_keep_the_second_call_on_the_provider(
    server: RecordingServer, first: dict[str, object], second: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch)
    server.expected_requests = 2

    await call(server, **first)
    response: Final = await call(server, **second)

    assert_served_natively(server, count=2)
    assert cache_key(response) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
async def test_caching_off_skips_the_read_but_still_writes(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Literal["python", "native"]
) -> None:
    use_cache(monkeypatch)
    server.expected_requests = 2

    async def send(**kwargs: object) -> object:
        if engine == "python":
            return await call_python(server, monkeypatch, **kwargs)
        return await call(server, **kwargs)

    await send(caching=False)
    await send()
    await send(caching=False)

    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_each_namespace_keeps_its_own_entry(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> None:
    use_cache(monkeypatch)
    server.expected_requests = 2

    for namespace in ("first", "second", "first", "second"):
        await call(server, cache={"namespace": namespace})

    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_the_key_is_built_from_the_call_arguments_not_the_wire_body(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Rewrite(RecordingLogger):
        temperature = 0.1

        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = self.temperature
            super().log_pre_api_call(model, messages, kwargs)

    use_cache(monkeypatch)
    logger: Final = Rewrite()

    await call(server, callbacks=[logger])
    logger.temperature = 0.8
    hit: Final = await call(server, callbacks=[logger])

    assert len(server.requests) == 1
    assert _OBJECT.validate_python(server.requests[0].body)["temperature"] == 0.1
    assert isinstance(cache_key(hit), str)


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded", (["embedding"], []))
async def test_supported_call_types_gate_native_reads_and_writes(
    server: RecordingServer, excluded: list[Literal["embedding"]], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = use_cache(monkeypatch)
    cache.supported_call_types = excluded
    server.expected_requests = 3

    await call(server)
    await call(server)
    cache.supported_call_types = ["anthropic_messages"]
    await call(server)
    hit: Final = await call(server)

    assert len(server.requests) == 3
    assert isinstance(cache_key(hit), str)


@pytest.mark.asyncio
async def test_default_off_mode_caches_only_calls_that_opt_in(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = use_cache(monkeypatch)
    cache.mode = CacheMode.default_off
    server.expected_requests = 4

    await call(server)
    await call(server)
    await call(server, cache={"use-cache": True})
    hit: Final = await call(server, cache={"use-cache": True})
    await call(server)

    assert len(server.requests) == 4
    assert isinstance(cache_key(hit), str)


@pytest.mark.asyncio
async def test_a_semantic_python_cache_answers_a_similar_prompt(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch, Cache(type=LiteLLMCacheType.REDIS_SEMANTIC, _backend=InMemoryCache()))

    first: Final = await call(server, messages=[{"role": "user", "content": "hello"}])
    second: Final = await call(server, messages=[{"role": "user", "content": "hi"}])

    assert len(server.requests) == 1
    assert cache_key(first) is None
    assert isinstance(cache_key(second), str)
    assert body(second) == body(first)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("response", "stream"))
async def test_a_native_hit_is_logged_as_a_free_cache_hit(
    server: RecordingServer, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch)
    if stream:
        server.default_response = STREAM
    recorder: Final = RecordingLogger()

    for _ in range(2):
        response = await call(server, stream=stream, callbacks=[recorder])  # rebind-ok: one response per call
        if stream:
            await collect(response)
    await drain_logging()

    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    miss, hit = (_OBJECT.validate_python(event.kwargs) for event in successes)
    assert miss["cache_hit"] is not True
    assert TypeAdapter(float).validate_python(miss["response_cost"]) > 0
    assert hit["cache_hit"] is True
    assert hit["response_cost"] == 0


@pytest.mark.asyncio
async def test_python_cache_lookups_run_in_the_callers_task(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = CallerCache()
    use_cache(monkeypatch, Cache(_backend=backend))

    await call(server)
    hit: Final = await call(server)

    assert len(server.requests) == 1
    assert isinstance(cache_key(hit), str)


@pytest.mark.asyncio
async def test_cancelling_a_pending_lookup_sends_and_stores_nothing(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    backend: Final = CallerCache(entered, release)
    use_cache(monkeypatch, Cache(_backend=backend))
    server.expected_requests = 0

    task: Final = asyncio.create_task(call(server))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0)

    assert server.requests == []
    assert backend.storage.cache_dict == {}


Engine = Literal["python", "native"]
EMPTY: Final = {**MESSAGES_RESPONSE, "content": []}


async def send(engine: Engine, server: RecordingServer, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> object:
    if engine == "python":
        return await call_python(server, monkeypatch, **kwargs)
    return await call(server, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
@pytest.mark.parametrize("stream", (False, True), ids=("response", "stream"))
async def test_a_cache_hit_keeps_budget_spend_and_counts_usage(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Engine, stream: bool
) -> None:
    use_cache(monkeypatch)
    if stream:
        server.default_response = STREAM

    async def options(extra: Mapping[str, object]) -> object:
        response: Final = await send(engine, server, monkeypatch, stream=stream, **extra)
        if stream:
            await collect(response)
        return response

    usage: Final = TypeAdapter(dict[str, int]).validate_python(MESSAGES_RESPONSE["usage"])
    await assert_cache_hit_keeps_budget_spend_and_counts_usage(
        options, server, usage["input_tokens"] + usage["output_tokens"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
async def test_a_response_without_output_is_not_stored(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    use_cache(monkeypatch)
    server.default_response = ResponseSpec(body=EMPTY)
    server.expected_requests = 2

    await send(engine, server, monkeypatch)
    await send(engine, server, monkeypatch)

    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_a_stored_response_without_output_is_a_miss(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = InMemoryCache()
    cache: Final = use_cache(monkeypatch, Cache(_backend=backend))
    await call(server)
    (key,) = backend.cache_dict
    cache.add_cache(EMPTY, cache_key=key)
    server.expected_requests = 2

    response: Final = await call(server)

    assert len(server.requests) == 2
    assert body(response) == MESSAGES_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
async def test_a_failing_cache_decision_runs_the_call_uncached(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    use_cache(monkeypatch, FailingDecisionCache())
    server.expected_requests = 2

    first: Final = await send(engine, server, monkeypatch)
    await send(engine, server, monkeypatch)

    assert body(first) == MESSAGES_RESPONSE
    assert len(server.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
async def test_the_response_returns_before_the_cache_write_lands(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Engine
) -> None:
    backend: Final = HeldWriteCache()
    use_cache(monkeypatch, Cache(_backend=backend))

    if engine == "python":
        with monkeypatch.context() as patched:
            patched.setenv("LITELLM_RUST", "0")
            first: Final = await asyncio.wait_for(
                litellm.anthropic.messages.acreate(
                    **REQUEST, api_key="test-key", api_base=server.base_url, litellm_call_id=CALL_ID
                ),
                timeout=5,
            )
    else:
        first = await asyncio.wait_for(route_call(server), timeout=5)  # rebind-ok: one branch binds it
    backend.release.set()
    await settle_cache_writes()
    hit: Final = await send(engine, server, monkeypatch)

    assert body(first) == MESSAGES_RESPONSE
    assert len(server.requests) == 1
    assert body(hit) == MESSAGES_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
async def test_a_redis_hit_fills_the_worker_memory_tier(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Engine, redis_url: str
) -> None:
    use_cache(monkeypatch, redis_cache(redis_url))

    await send(engine, server, monkeypatch)
    await send(engine, server, monkeypatch)
    redis.Redis.from_url(redis_url).flushall()
    hit: Final = await send(engine, server, monkeypatch)

    assert len(server.requests) == 1
    assert body(hit) == MESSAGES_RESPONSE


@pytest.mark.asyncio
async def test_a_hit_reports_the_key_the_entry_is_stored_under(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = InMemoryCache()
    use_cache(monkeypatch, Cache(_backend=backend))

    await call(server)
    hit: Final = await call(server)

    (stored,) = backend.cache_dict
    assert stored_cache_key(hit) == cache_key(hit) == stored
