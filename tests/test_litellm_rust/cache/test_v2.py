import asyncio
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, Literal

import pytest
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm import _v2
from litellm._v2.cache import NativeBackend
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION, LiteLLMChatCompletionsRequest
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import call_hook
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, LiteLLMMessagesRequest
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES, LiteLLMResponsesRequest
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.test_inference import RESPONSES_MODEL, RESPONSES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


def payload(value: object) -> object:
    if isinstance(value, ModelResponse):
        return value.model_dump_json(exclude=MappingProxyType({"id": True, "created": True}))
    if isinstance(value, dict):
        fields: Final = TypeAdapter(dict[str, object]).validate_python(value)
        return {name: field for name, field in fields.items() if name != "_hidden_params"}
    return value.model_dump_json() if isinstance(value, BaseModel) else value


def cache_key(response: object) -> object:
    hidden: Final = get_hidden_params_dict(response)
    headers: Final = TypeAdapter(dict[str, object]).validate_python(hidden.get("additional_headers", {}))
    return headers.get("x-litellm-cache-key")


async def invoke(
    route: Literal["chat", "messages", "responses"],
    server: RecordingServer,
    options: Mapping[str, object],
    native: bool = True,
) -> object:
    common: Final = {"api_key": "test-key", "api_base": server.base_url, **options}
    if route == "responses":
        server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
        arguments: Final = {"model": RESPONSES_MODEL, "input": "hello", **common}
        if not native:
            return await litellm.aresponses(**arguments)
        request: Final = LiteLLMResponsesRequest(
            RESPONSES_MODEL, "hello", None, "test-key", server.base_url, "openai", None, arguments
        )
        return await runtime.arun(
            RouteContext(Route.RESPONSES),
            binding=NATIVE_ARESPONSES,
            native=lambda hook: call_hook(hook, request, (), arguments),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),),
        )
    server.default_response = (
        ResponseSpec(body=None, events=MESSAGES_EVENTS)
        if options.get("stream")
        else ResponseSpec(body=MESSAGES_RESPONSE)
    )
    parameters: Final = {"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 32, **common}
    if route == "chat":
        if not native:
            return await litellm.acompletion(**parameters)
        chat: Final = LiteLLMChatCompletionsRequest(
            MESSAGES_MODEL, list(MESSAGES), None, "test-key", server.base_url, None, None, parameters
        )
        return await runtime.arun(
            RouteContext(Route.CHAT_COMPLETIONS),
            binding=NATIVE_ACOMPLETION,
            native=lambda hook: call_hook(hook, chat, (), parameters),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),),
        )
    if not native:
        return await litellm.anthropic_messages(**parameters)
    messages: Final = LiteLLMMessagesRequest(
        MESSAGES_MODEL, list(MESSAGES), 32, None, "test-key", server.base_url, "anthropic", parameters
    )
    return await runtime.arun(
        RouteContext(Route.MESSAGES),
        binding=NATIVE_AMESSAGES,
        native=lambda hook: call_hook(hook, messages, (), parameters),
        python=runtime.NO_PYTHON,
        rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
    )


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
@pytest.mark.parametrize(
    ("route", "stream"), (("chat", False), ("messages", False), ("responses", False), ("messages", True))
)
@pytest.mark.parametrize("native", (False, True), ids=("python", "rust"))
async def test_cache_hit_keeps_model_budget_spend_but_accounts_for_usage(
    recording_server: RecordingServer,
    route: Literal["chat", "messages", "responses"],
    stream: bool,
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1" if native else "0")
    litellm.cache = _v2.Cache.memory()
    counters: Final = litellm.DualCache()
    budget: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(counters)
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(
        InternalUsageCache(counters), model_group_resolver=lambda model: model
    )
    recorder: Final = RecordingLogger()
    key_hash: Final = "a" * 64
    metadata: Final = {
        "user_api_key": key_hash,
        "model_group": "cached-model",
        "user_api_key_model_max_budget": {"cached-model": {"max_budget": 1, "budget_duration": "1h"}},
    }
    options: Final = {
        "callbacks": [budget, limiter, recorder],
        "metadata": metadata,
        "stream": stream,
    }
    spend_key: Final = model_budget_spend_cache_key(Litellm_EntityType.KEY, key_hash, "cached-model", "1h")
    token_key: Final = limiter.create_rate_limit_keys("api_key", key_hash, "tokens")
    first: Final = await invoke(route, recording_server, options, native=native)
    if stream:
        await collect(first)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await drain_logging()
    first_events: Final = await recorder.wait_for_async("async_log_success_event")
    first_log: Final = TypeAdapter(dict[str, object]).validate_python(first_events[0].kwargs)
    first_payload: Final = TypeAdapter(dict[str, object]).validate_python(first_log["standard_logging_object"])
    expected_cost: Final = TypeAdapter(float).validate_python(first_log["response_cost"])
    usage: Final = RESPONSES_RESPONSE["usage"] if route == "responses" else MESSAGES_RESPONSE["usage"]
    expected_tokens: Final = usage["input_tokens"] + usage["output_tokens"]
    assert expected_cost > 0
    assert counters.get_cache(spend_key) == pytest.approx(expected_cost)
    assert counters.get_cache(token_key) == first_payload["total_tokens"] == expected_tokens

    second: Final = await invoke(route, recording_server, options, native=native)
    if stream:
        await collect(second)
    await drain_logging()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    cached_log: Final = TypeAdapter(dict[str, object]).validate_python(successes[-1].kwargs)
    cached_payload: Final = TypeAdapter(dict[str, object]).validate_python(cached_log["standard_logging_object"])
    assert len(recording_server.requests) == 1
    assert len(successes) == 2
    assert cached_log["cache_hit"] is True
    assert cached_log["response_cost"] == cached_payload["response_cost"] == 0
    assert cached_payload["cache_hit"] is True
    assert cached_payload["id"] != first_payload["id"]
    assert cached_payload["custom_llm_provider"] == first_payload["custom_llm_provider"]
    assert cached_payload["custom_llm_provider"] == ("openai" if route == "responses" else "anthropic"), {
        "miss_provider": first_log.get("custom_llm_provider"),
        "hit_provider": cached_log.get("custom_llm_provider"),
    }
    cached_hidden: Final = TypeAdapter(dict[str, object]).validate_python(cached_payload["hidden_params"])
    assert cached_hidden["response_cost"] == 0
    breakdown: Final = TypeAdapter(dict[str, object]).validate_python(cached_payload["cost_breakdown"])
    assert breakdown["total_cost"] == 0
    assert cached_payload["total_tokens"] == first_payload["total_tokens"]
    assert counters.get_cache(spend_key) == pytest.approx(expected_cost)
    assert counters.get_cache(token_key) == 2 * expected_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("native", (False, True), ids=("python", "rust"))
async def test_cache_hit_releases_parallel_slot_and_preserves_request_rate_limit(
    recording_server: RecordingServer, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1" if native else "0")
    litellm.cache = _v2.Cache.memory()
    counters: Final = litellm.DualCache()
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(
        InternalUsageCache(counters), model_group_resolver=lambda model: model
    )
    key_hash: Final = "b" * 64
    identity: Final = UserAPIKeyAuth(api_key=key_hash, rpm_limit=2, tpm_limit=1000, max_parallel_requests=1)
    request_key: Final = limiter.create_rate_limit_keys("api_key", key_hash, "requests")
    token_key: Final = limiter.create_rate_limit_keys("api_key", key_hash, "tokens")
    parallel_key: Final = limiter.create_rate_limit_keys("api_key", key_hash, "max_parallel_requests")
    expected_tokens: Final = MESSAGES_RESPONSE["usage"]["input_tokens"] + MESSAGES_RESPONSE["usage"]["output_tokens"]
    recorder: Final = RecordingLogger()

    async def request(call_id: str, successes: int) -> object:
        data: Final = {
            "model": MESSAGES_MODEL,
            "messages": list(MESSAGES),
            "litellm_call_id": call_id,
            "max_tokens": 32,
            "metadata": {"user_api_key": key_hash},
        }
        await limiter.async_pre_call_hook(identity, counters, data, "acompletion")
        assert len(TypeAdapter(dict[str, float]).validate_python(counters.get_cache(parallel_key))) == 1
        response: Final = await invoke(
            "chat", recording_server, {**data, "callbacks": [limiter, recorder]}, native=native
        )
        await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
        await recorder.wait_for_async("async_log_success_event", count=successes)
        return response

    await asyncio.create_task(request("cache-miss", 1))
    assert counters.get_cache(parallel_key) == {}
    assert counters.get_cache(request_key) == 1
    assert counters.get_cache(token_key) == expected_tokens
    await asyncio.create_task(request("cache-hit", 2))
    assert len(recording_server.requests) == 1
    assert counters.get_cache(parallel_key) == {}
    assert counters.get_cache(request_key) == 2
    assert counters.get_cache(token_key) == 2 * expected_tokens
    with pytest.raises(litellm.RateLimitError):
        await asyncio.create_task(request("over-rpm-limit", 3))
    assert len(recording_server.requests) == 1
    assert counters.get_cache(parallel_key) == {}
    assert counters.get_cache(token_key) == 2 * expected_tokens


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


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "messages", "responses"))
async def test_v2_cache_controls_and_credentials_isolate_requests(
    recording_server: RecordingServer, route: Literal["chat", "messages", "responses"]
) -> None:
    recording_server.expected_requests = 4
    litellm.cache = _v2.Cache.memory()
    await invoke(route, recording_server, {"cache": {"no-store": True}})
    await invoke(route, recording_server, {})
    await invoke(route, recording_server, {})
    assert len(recording_server.requests) == 2
    await invoke(route, recording_server, {"cache": {"no-cache": True}})
    await invoke(route, recording_server, {"api_key": "another-key"})
    assert len(recording_server.requests) == 4


async def collect(stream: object) -> bytes:
    assert isinstance(stream, AsyncIterator)
    return b"".join([chunk_bytes(chunk) async for chunk in stream])


def chunk_bytes(value: object) -> bytes:
    assert isinstance(value, bytes)
    return value


@pytest.mark.asyncio
async def test_v2_messages_replays_a_completed_stream(recording_server: RecordingServer) -> None:
    recording_server.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
    litellm.cache = _v2.Cache.memory()
    recorder: Final = RecordingLogger()
    parameters: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
        "stream": True,
        "callbacks": [recorder],
    }
    first_stream: Final = await litellm.anthropic_messages(**parameters)
    assert cache_key(first_stream) is None
    first: Final = await collect(first_stream)
    await recorder.wait_for_async("async_log_success_event")
    second_stream: Final = await litellm.anthropic_messages(**parameters)
    assert isinstance(cache_key(second_stream), str)
    assert cache_key(second_stream) == get_hidden_params_dict(second_stream)["cache_key"]
    second: Final = await collect(second_stream)
    assert payload(first) == payload(second)
    assert first == b"".join(recording_server.default_response.payloads())
    assert len(recording_server.requests) == 1
    await drain_logging()
    successes: Final = await recorder.wait_for_async("async_log_success_event", count=2)
    cached_log: Final = TypeAdapter(dict[str, object]).validate_python(successes[-1].kwargs)
    assert cached_log["cache_hit"] is True
    assert cached_log["response_cost"] == 0


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
async def test_rust_messages_fallback_honors_a_legacy_cache(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setenv("LITELLM_RUST", "1")
    litellm.cache = Cache()
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
