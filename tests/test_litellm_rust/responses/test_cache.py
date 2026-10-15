import asyncio
from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.caching.caching import Cache, CacheMode
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.caching import CachingSupportedCallTypes
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.responses.data import RESPONSES_RESPONSE, RESPONSES_TEXT
from tests.test_litellm_rust.support.caching import (
    Call,
    CallerCache,
    Engine,
    FailingLookupCache,
    FakeRedis,
    HeldWriteCache,
    assert_cache_hit_keeps_budget_spend_and_counts_usage,
    native_only,
    other,
    settle_cache_writes,
    use_cache,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging, request_body
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.results import cache_key, stored_cache_key
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])
EMPTY: Final = {**RESPONSES_RESPONSE, "output": []}
OWN_CALL_TYPES: Final = ("responses",)


def output_text(response: object) -> str:
    assert isinstance(response, ResponsesAPIResponse)
    return response.output_text


@native_only
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.usefixtures("cache")
async def test_a_repeated_native_call_is_answered_from_the_python_cache(
    call: Call, server: RecordingServer, asynchronous: bool
) -> None:
    first: Final = await call(asynchronous=asynchronous)
    second: Final = await call(asynchronous=asynchronous)

    assert_served_natively(server, first)
    assert cache_key(first) is None
    assert isinstance(cache_key(second), str)
    assert output_text(second) == output_text(first) == RESPONSES_TEXT


@pytest.mark.usefixtures("cache")
async def test_native_and_python_calls_share_python_cache_entries(
    calls: Mapping[Engine, Call], engine: Engine, server: RecordingServer
) -> None:
    first: Final = await calls[engine]()
    await settle_cache_writes()
    second: Final = await calls[other(engine)]()

    assert len(server.requests) == 1
    assert output_text(second) == output_text(first) == RESPONSES_TEXT


@native_only
@pytest.mark.parametrize(
    ("first", "second"),
    (
        ({"cache": {"no-store": True}}, {}),
        ({}, {"cache": {"no-cache": True}}),
        ({}, {"caching": False}),
        ({}, {"input": "Goodbye"}),
        ({}, {"instructions": "Be brief"}),
        ({"cache": {"namespace": "first"}}, {"cache": {"namespace": "second"}}),
        ({"metadata": {"redis_namespace": "first"}}, {"metadata": {"redis_namespace": "second"}}),
    ),
    ids=(
        "no-store",
        "no-cache",
        "caching-off-read",
        "other-input",
        "other-param",
        "cache-namespace",
        "metadata-namespace",
    ),
)
@pytest.mark.usefixtures("cache")
async def test_cache_controls_and_keys_keep_the_second_call_on_the_provider(
    call: Call, server: RecordingServer, first: Mapping[str, object], second: Mapping[str, object]
) -> None:
    server.expect(2)

    await call(**first)
    response: Final = await call(**second)

    assert_served_natively(server, count=2)
    assert cache_key(response) is None


@pytest.mark.usefixtures("cache")
async def test_caching_off_skips_the_read_but_still_writes(call: Call, server: RecordingServer) -> None:
    await call(caching=False)
    hit: Final = await call()

    assert len(server.requests) == 1
    assert output_text(hit) == RESPONSES_TEXT


@native_only
@pytest.mark.usefixtures("cache")
async def test_each_namespace_keeps_its_own_entry(call: Call, server: RecordingServer) -> None:
    server.expect(2)

    await call(cache={"namespace": "first"})
    await call(cache={"namespace": "second"})
    hit: Final = await call(cache={"namespace": "first"})

    assert len(server.requests) == 2
    assert isinstance(cache_key(hit), str)


@native_only
@pytest.mark.usefixtures("cache")
async def test_the_key_is_built_from_the_call_arguments_not_the_wire_body(call: Call, server: RecordingServer) -> None:
    class Rewrite(RecordingLogger):
        temperature = 0.1

        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = self.temperature
            super().log_pre_api_call(model, messages, kwargs)

    logger: Final = Rewrite()

    await call(callbacks=[logger])
    logger.temperature = 0.8
    hit: Final = await call(callbacks=[logger])

    assert len(server.requests) == 1
    assert _OBJECT.validate_python(server.requests[0].body)["temperature"] == 0.1
    assert isinstance(cache_key(hit), str)


@native_only
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize(
    ("supported", "expected_requests"), ((["embedding"], 2), (OWN_CALL_TYPES, 1)), ids=("other-type", "own-type")
)
async def test_supported_call_types_gate_native_reads_and_writes(
    call: Call,
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    supported: Sequence[CachingSupportedCallTypes],
    expected_requests: int,
) -> None:
    use_cache(monkeypatch, Cache(supported_call_types=list(supported)))
    server.expect(expected_requests)

    await call(asynchronous=asynchronous)
    second: Final = await call(asynchronous=asynchronous)

    assert len(server.requests) == expected_requests
    assert isinstance(cache_key(second), str) == (expected_requests == 1)


@native_only
@pytest.mark.parametrize(
    ("options", "expected_requests"), (({}, 2), ({"cache": {"use-cache": True}}, 1)), ids=("plain", "opt-in")
)
async def test_default_off_mode_caches_only_calls_that_opt_in(
    call: Call,
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    options: Mapping[str, object],
    expected_requests: int,
) -> None:
    use_cache(monkeypatch, Cache(mode=CacheMode.default_off))
    server.expect(expected_requests)

    await call(**options)
    second: Final = await call(**options)

    assert len(server.requests) == expected_requests
    assert isinstance(cache_key(second), str) == (expected_requests == 1)


@native_only
@pytest.mark.usefixtures("cache")
async def test_a_native_hit_is_logged_as_a_free_cache_hit(call: Call, server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    await call(callbacks=[recorder])
    await call(callbacks=[recorder])
    await drain_logging()

    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    miss, hit = (_OBJECT.validate_python(event.kwargs) for event in successes)
    assert len(server.requests) == 1
    assert miss["cache_hit"] is not True
    assert TypeAdapter(float).validate_python(miss["response_cost"]) > 0
    assert hit["cache_hit"] is True
    assert hit["response_cost"] == 0


@native_only
async def test_python_cache_lookups_run_in_the_callers_task(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = CallerCache()
    use_cache(monkeypatch, Cache(_backend=backend))

    await call()
    hit: Final = await call()

    assert len(server.requests) == 1
    assert isinstance(cache_key(hit), str)
    assert backend.lookup_tasks == [asyncio.current_task(), asyncio.current_task()]


@native_only
async def test_cancelling_a_pending_lookup_sends_and_stores_nothing(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    backend: Final = CallerCache(entered, release)
    use_cache(monkeypatch, Cache(_backend=backend))
    server.expect(0)

    task: Final = asyncio.create_task(call())
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0)

    assert server.requests == []
    assert backend.storage.cache_dict == {}


@pytest.mark.usefixtures("cache")
async def test_a_cache_hit_keeps_budget_spend_and_counts_usage(call: Call, server: RecordingServer) -> None:
    usage: Final = TypeAdapter(dict[str, int]).validate_python(RESPONSES_RESPONSE["usage"])

    await assert_cache_hit_keeps_budget_spend_and_counts_usage(call, server, usage["total_tokens"])


@pytest.mark.usefixtures("cache")
async def test_a_response_without_output_is_not_stored(call: Call, server: RecordingServer) -> None:
    server.serve(ResponseSpec(body=EMPTY))
    server.expect(2)

    await call()
    await call()

    assert len(server.requests) == 2


@native_only
async def test_a_stored_response_without_output_is_a_miss(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = InMemoryCache()
    cache: Final = use_cache(monkeypatch, Cache(_backend=backend))
    await call()
    (key,) = backend.cache_dict
    cache.add_cache(EMPTY, cache_key=key)
    server.expect(2)

    response: Final = await call()

    assert len(server.requests) == 2
    assert output_text(response) == RESPONSES_TEXT


async def test_a_failing_cache_lookup_runs_the_call_uncached(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch, FailingLookupCache())
    server.expect(2)

    first: Final = await call()
    await call()

    assert output_text(first) == RESPONSES_TEXT
    assert len(server.requests) == 2


async def test_the_response_returns_before_the_cache_write_lands(
    calls: Mapping[Engine, Call], engine: Engine, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = HeldWriteCache()
    use_cache(monkeypatch, Cache(_backend=backend))

    first: Final = await asyncio.wait_for(calls[engine](), timeout=5)
    backend.release.set()
    await settle_cache_writes()
    hit: Final = await calls[engine]()

    assert output_text(first) == RESPONSES_TEXT
    assert len(server.requests) == 1
    assert output_text(hit) == RESPONSES_TEXT


async def test_a_redis_hit_fills_the_worker_memory_tier(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    use_cache(monkeypatch, fake_redis.cache())

    await call()
    await call()
    fake_redis.client.flushall()
    hit: Final = await call()

    assert len(server.requests) == 1
    assert output_text(hit) == RESPONSES_TEXT


async def test_a_stored_response_lands_in_the_worker_memory_tier(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    use_cache(monkeypatch, fake_redis.cache())

    first: Final = await call()
    fake_redis.client.flushall()
    hit: Final = await call()

    assert len(server.requests) == 1
    assert output_text(hit) == output_text(first)


async def test_a_hit_reports_the_key_the_entry_is_stored_under(
    call: Call, server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend: Final = InMemoryCache()
    use_cache(monkeypatch, Cache(_backend=backend))

    await call()
    hit: Final = await call()

    (stored,) = backend.cache_dict
    assert len(server.requests) == 1
    assert stored_cache_key(hit) == stored
