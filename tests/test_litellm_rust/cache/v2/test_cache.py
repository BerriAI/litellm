import asyncio
from collections.abc import AsyncIterator
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

import litellm
from litellm import _v2
from litellm._v2.cache import NativeBackend
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.cache import (
    cache_key,
    invoke,
    payload,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.test_inference import RESPONSES_MODEL, RESPONSES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "messages", "responses"))
@pytest.mark.parametrize("backend", ("memory", "redis"))
async def test_v2_cache_skips_provider_and_reports_one_success_per_call(
    recording_server: RecordingServer,
    route: Literal["chat", "messages", "responses"],
    backend: Literal["memory", "redis"],
    redis_url: str,
) -> None:
    recording_server.expected_requests = 2
    litellm.cache = _v2.Cache.memory() if backend == "memory" else _v2.Cache.redis(redis_url, namespace="headers")
    recorder: Final = RecordingLogger()
    first: Final = await invoke(route, recording_server, {"callbacks": [recorder]})
    await recorder.wait_for_async("async_log_success_event")
    second: Final = await invoke(route, recording_server, {"callbacks": [recorder]})
    assert payload(first) == payload(second)
    assert cache_key(first) is None
    key: Final = cache_key(second)
    assert isinstance(key, str)
    assert key == get_hidden_params_dict(second)["cache_key"]
    assert len(recording_server.requests) == 1
    await drain_logging()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    assert len(successes) == 2
    cached_log: Final = TypeAdapter(dict[str, object]).validate_python(successes[-1].kwargs)
    assert cached_log["cache_hit"] is True
    assert cached_log["response_cost"] == 0
    await litellm.cache.delete_cache_keys([key])
    refreshed: Final = await invoke(route, recording_server, {"callbacks": [recorder]})
    assert cache_key(refreshed) is None
    assert len(recording_server.requests) == 2
    assert len(await recorder.wait_for_async("async_log_success_event", count=3)) == 3
    await litellm.cache.disconnect()


@pytest.mark.asyncio
async def test_v2_global_cache_leaves_legacy_only_calls_usable() -> None:
    litellm.cache = _v2.Cache.memory()
    response: Final = await litellm.aembedding(
        model="openai/cache-test-embedding",
        input=["hello"],
        api_key="test-key",
        mock_response=[0.25, 0.75],
    )
    assert response.model_dump(include={"data"}) == {
        "data": [{"embedding": [0.25, 0.75], "index": 0, "object": "embedding"}]
    }


@pytest.mark.parametrize("route", ("chat", "responses"))
def test_v2_cache_works_through_python_inference(
    recording_server: RecordingServer, route: Literal["chat", "messages", "responses"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    litellm.cache = _v2.Cache.memory()
    common: Final = {"api_key": "test-key", "api_base": recording_server.base_url}
    if route == "responses":
        recording_server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        parameters: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        first: Final = litellm.responses(**parameters)
        second: Final = litellm.responses(**parameters)
        assert payload(first) == payload(second)
    else:
        recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
        arguments: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
        initial: Final = litellm.completion(**arguments)
        cached: Final = litellm.completion(**arguments)
        assert isinstance(initial, ModelResponse) and isinstance(cached, ModelResponse)
        assert (
            initial.choices[0].message.content
            == cached.choices[0].message.content
            == MESSAGES_RESPONSE["content"][0]["text"]
        )
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
async def test_v2_facade_and_backend_share_storage_and_management() -> None:
    cache: Final = _v2.Cache.memory()
    await cache.async_add_cache({"answer": 7}, cache_key="shared")
    assert cache.get_cache(cache_key="shared") == {"answer": 7}
    assert await cache.ping() is True
    await cache.delete_cache_keys(["shared"])
    assert await cache.async_get_cache(cache_key="shared") is None
    cache.add_cache({"answer": 8}, cache_key="flush")
    backend: Final = cache.cache
    assert isinstance(backend, NativeBackend)
    backend.flush_cache()
    assert cache.get_cache(cache_key="flush") is None
    await cache.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ("s-maxage", "s-max-age"))
async def test_v2_native_cache_accepts_existing_freshness_aliases(
    recording_server: RecordingServer, control: str
) -> None:
    litellm.cache = _v2.Cache.memory()
    first: Final = await invoke("responses", recording_server, {})
    second: Final = await invoke("responses", recording_server, {"cache": {control: 600}})
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
async def test_v2_cache_does_not_force_native_responses_streaming(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    litellm.cache = _v2.Cache.memory()
    recording_server.default_response = ResponseSpec(
        body=None,
        events=(
            ("response.created", {"type": "response.created", "sequence_number": 0, "response": RESPONSES_RESPONSE}),
            (
                "response.completed",
                {"type": "response.completed", "sequence_number": 1, "response": RESPONSES_RESPONSE},
            ),
        ),
    )
    response: Final = await litellm.aresponses(
        model=RESPONSES_MODEL,
        input="hello",
        stream=True,
        caching=False,
        api_key="test-key",
        api_base=recording_server.base_url,
    )
    assert isinstance(response, AsyncIterator)
    chunks: Final = [chunk async for chunk in response]
    assert chunks[-1].type == "response.completed"
    assert chunks[-1].response.output[0].content[0].text == "native response"


@pytest.mark.asyncio
async def test_v2_cache_works_through_python_messages(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    litellm.cache = _v2.Cache.memory()
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    parameters: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
    }
    first: Final = await litellm.anthropic_messages(**parameters)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    second: Final = await litellm.anthropic_messages(**parameters)
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_v2_redis_flush_only_removes_its_namespace(
    redis_url: str, recording_server: RecordingServer, asynchronous: bool
) -> None:
    own: Final = _v2.Cache.redis(redis_url, namespace="flush-own")
    other: Final = _v2.Cache.redis(redis_url, namespace="flush-other")
    litellm.cache = own
    recording_server.expected_requests = 2
    await own.async_add_cache({"answer": "own"}, cache_key="shared")
    await other.async_add_cache({"answer": "other"}, cache_key="shared")
    await invoke("responses", recording_server, {})
    hit: Final = await invoke("responses", recording_server, {})
    assert isinstance(cache_key(hit), str)
    assert await own.async_get_cache(cache_key="shared") == {"answer": "own"}
    backend: Final = own.cache
    assert isinstance(backend, NativeBackend)
    if asynchronous:
        await backend.async_flush_cache()
    else:
        backend.flush_cache()
    assert await own.async_get_cache(cache_key="shared") is None
    assert await other.async_get_cache(cache_key="shared") == {"answer": "other"}
    refreshed: Final = await invoke("responses", recording_server, {})
    assert cache_key(refreshed) is None
    assert len(recording_server.requests) == 2
    await own.disconnect()
    await other.disconnect()
