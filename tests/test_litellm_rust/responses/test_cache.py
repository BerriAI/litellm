import asyncio
from collections.abc import Mapping
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.responses.data import REQUEST, RESPONSES_RESPONSE, RESPONSES_TEXT, call as route_call
from tests.test_litellm_rust.support.caching import (
    CallerCache,
    assert_cache_hit_keeps_budget_spend_and_counts_usage,
    settle_cache_writes,
    use_cache,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging, request_body
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.results import cache_key
from tests.test_litellm_rust.support.routing import CALL_ID, assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


def output_text(response: object) -> str:
    assert isinstance(response, ResponsesAPIResponse)
    return response.output_text


async def call_python(server: RecordingServer, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> object:
    with monkeypatch.context() as patched:
        patched.setenv("LITELLM_RUST", "0")
        response: Final = await litellm.aresponses(
            **{**REQUEST, **kwargs}, api_key="test-key", api_base=server.base_url, litellm_call_id=CALL_ID
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
    assert output_text(second) == output_text(first) == RESPONSES_TEXT


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ("python", "native"))
async def test_native_and_python_calls_share_python_cache_entries(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, writer: Literal["python", "native"]
) -> None:
    use_cache(monkeypatch)

    first: Final = await (call_python(server, monkeypatch) if writer == "python" else call(server))
    second: Final = await (call(server) if writer == "python" else call_python(server, monkeypatch))

    assert len(server.requests) == 1
    assert output_text(second) == output_text(first) == RESPONSES_TEXT


@pytest.mark.asyncio
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
async def test_a_native_hit_is_logged_as_a_free_cache_hit(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch)
    recorder: Final = RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    await call(server)
    await call(server)
    await drain_logging()

    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    miss, hit = (TypeAdapter(dict[str, object]).validate_python(event.kwargs) for event in successes)
    assert miss["cache_hit"] is not True
    assert TypeAdapter(float).validate_python(miss["response_cost"]) > 0
    assert hit["cache_hit"] is True
    assert hit["response_cost"] == 0


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
    assert TypeAdapter(dict[str, object]).validate_python(server.requests[0].body)["temperature"] == 0.1
    assert isinstance(cache_key(hit), str)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("excluded", (["embedding"], []))
async def test_supported_call_types_gate_native_reads_and_writes(
    server: RecordingServer, asynchronous: bool, excluded: list[Literal["embedding"]], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache: Final = use_cache(monkeypatch)
    cache.supported_call_types = excluded
    server.expected_requests = 3

    await call(server, asynchronous=asynchronous)
    await call(server, asynchronous=asynchronous)
    cache.supported_call_types = ["aresponses" if asynchronous else "responses"]
    await call(server, asynchronous=asynchronous)
    hit: Final = await call(server, asynchronous=asynchronous)

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
async def test_python_cache_lookups_run_in_the_callers_task(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch, Cache(_backend=CallerCache()))

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


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ("python", "native"))
async def test_a_cache_hit_keeps_budget_spend_and_counts_usage(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, engine: Literal["python", "native"]
) -> None:
    use_cache(monkeypatch)

    async def send(options: Mapping[str, object]) -> object:
        if engine == "python":
            return await call_python(server, monkeypatch, **options)
        return await call(server, **options)

    usage: Final = TypeAdapter(dict[str, int]).validate_python(RESPONSES_RESPONSE["usage"])
    await assert_cache_hit_keeps_budget_spend_and_counts_usage(send, server, usage["total_tokens"])
