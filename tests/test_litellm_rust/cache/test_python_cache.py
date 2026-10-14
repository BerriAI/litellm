import asyncio
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

import litellm
from litellm import _v2
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.dispatch import call_hook
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.caching import CachingSupportedCallTypes
from tests.test_litellm_rust.support.cache import cache_key, collect, invoke, payload
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.test_inference import RESPONSES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "stream", "legacy"),
    (
        ("chat", False, False),
        ("messages", False, False),
        ("responses", False, False),
        ("messages", True, False),
        ("messages", False, True),
        ("messages", True, True),
    ),
)
@pytest.mark.parametrize("native", (False, True), ids=("python", "rust"))
async def test_cache_hit_keeps_model_budget_spend_but_accounts_for_usage(
    recording_server: RecordingServer,
    route: Literal["chat", "messages", "responses"],
    stream: bool,
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1" if native else "0")
    litellm.cache = Cache() if legacy else _v2.Cache.memory()
    counters: Final = litellm.DualCache()
    budget: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(counters)
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
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
    assert cached_payload["total_tokens"] == first_payload["total_tokens"]
    assert counters.get_cache(spend_key) == pytest.approx(expected_cost)
    assert counters.get_cache(token_key) == 2 * expected_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("native", (False, True), ids=("python", "rust"))
@pytest.mark.parametrize("backend", ("disabled", "memory", "redis"))
async def test_response_cache_backend_does_not_control_coordination(
    recording_server: RecordingServer,
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["disabled", "memory", "redis"],
    redis_url: str,
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1" if native else "0")
    litellm.cache = (
        None
        if backend == "disabled"
        else _v2.Cache.memory()
        if backend == "memory"
        else _v2.Cache.redis(redis_url, namespace="independent-coordination")
    )
    recording_server.expected_requests = 2 if backend == "disabled" else 1
    counters: Final = litellm.DualCache()
    budget: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(counters)
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        InternalUsageCache(counters), model_group_resolver=lambda model: model
    )
    key_hash: Final = "b" * 64
    identity: Final = UserAPIKeyAuth(api_key=key_hash, rpm_limit=2, tpm_limit=1000, max_parallel_requests=1)
    spend_key: Final = model_budget_spend_cache_key(Litellm_EntityType.KEY, key_hash, "cached-model", "1h")
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
            "metadata": {
                "user_api_key": key_hash,
                "model_group": "cached-model",
                "user_api_key_model_max_budget": {"cached-model": {"max_budget": 1, "budget_duration": "1h"}},
            },
        }
        await limiter.async_pre_call_hook(identity, counters, data, "acompletion")
        assert len(TypeAdapter(dict[str, float]).validate_python(counters.get_cache(parallel_key))) == 1
        response: Final = await invoke(
            "chat", recording_server, {**data, "callbacks": [budget, limiter, recorder]}, native=native
        )
        await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
        await recorder.wait_for_async("async_log_success_event", count=successes)
        return response

    await asyncio.create_task(request("cache-miss", 1))
    assert counters.get_cache(parallel_key) == {}
    assert counters.get_cache(request_key) == 1
    assert counters.get_cache(token_key) == expected_tokens
    first_events: Final = await recorder.wait_for_async("async_log_success_event")
    first_cost: Final = TypeAdapter(float).validate_python(first_events[0].kwargs["response_cost"])
    assert first_cost > 0
    assert counters.get_cache(spend_key) == pytest.approx(first_cost)
    await asyncio.create_task(request("cache-hit", 2))
    expected_spend: Final = first_cost * recording_server.expected_requests
    assert counters.get_cache(spend_key) == pytest.approx(expected_spend)
    assert len(recording_server.requests) == recording_server.expected_requests
    assert counters.get_cache(parallel_key) == {}
    assert counters.get_cache(request_key) == 2
    assert counters.get_cache(token_key) == 2 * expected_tokens
    with pytest.raises(litellm.RateLimitError):
        await asyncio.create_task(request("over-rpm-limit", 3))
    assert len(recording_server.requests) == recording_server.expected_requests
    assert counters.get_cache(parallel_key) == {}
    assert counters.get_cache(token_key) == 2 * expected_tokens

    assert counters.get_cache(spend_key) == pytest.approx(expected_spend)
    if litellm.cache is not None:
        await litellm.cache.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "legacy"), (("chat", False), ("messages", False), ("responses", False), ("messages", True))
)
async def test_cache_controls_and_backend_credential_key_semantics(
    recording_server: RecordingServer,
    route: Literal["chat", "messages", "responses"],
    legacy: bool,
) -> None:
    recording_server.expected_requests = 3 if legacy else 4
    litellm.cache = Cache() if legacy else _v2.Cache.memory()
    await invoke(route, recording_server, {"cache": {"no-store": True}})
    await invoke(route, recording_server, {})
    await invoke(route, recording_server, {})
    assert len(recording_server.requests) == 2
    await invoke(route, recording_server, {"cache": {"no-cache": True}})
    await invoke(route, recording_server, {"api_key": "another-key"})
    assert len(recording_server.requests) == recording_server.expected_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", (False, True))
async def test_v2_messages_replays_a_completed_stream(recording_server: RecordingServer, legacy: bool) -> None:
    recording_server.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
    litellm.cache = Cache() if legacy else _v2.Cache.memory()
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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "redis"))
async def test_rust_messages_uses_a_legacy_cache_without_python_inference(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["memory", "redis"],
    redis_url: str,
) -> None:
    from litellm.caching.caching import Cache

    monkeypatch.setenv("LITELLM_RUST", "1")
    litellm.cache = Cache() if backend == "memory" else Cache(type="redis", url=redis_url, namespace="rust-host")
    logger: Final = RecordingLogger()
    litellm.callbacks = [logger]
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    parameters: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
    }
    first: Final = await invoke("messages", recording_server, parameters)
    second: Final = await invoke("messages", recording_server, parameters)
    assert cache_key(second)
    assert cache_key(first) is None
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1

    await logger.wait_for_async("async_log_success_event", count=2)
    assert logger.names.count("async_log_success_event") == 2
    assert "log_failure_event" not in logger.names
    assert "async_log_failure_event" not in logger.names


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("chat", "messages", "responses"))
@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("excluded", (None, [], ["embedding"]))
async def test_v2_cache_honors_supported_call_types_for_reads_and_writes(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    route: Literal["chat", "messages", "responses"],
    native: bool,
    excluded: list[CachingSupportedCallTypes] | None,
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1" if native else "0")
    litellm.cache = _v2.Cache.memory()
    call_type: Final[CachingSupportedCallTypes] = (
        "acompletion" if route == "chat" else "anthropic_messages" if route == "messages" else "aresponses"
    )
    recording_server.expected_requests = 4
    litellm.cache.supported_call_types = excluded
    await invoke(route, recording_server, {}, native=native)
    await invoke(route, recording_server, {}, native=native)
    assert len(recording_server.requests) == 2
    litellm.cache.supported_call_types = [call_type]
    await invoke(route, recording_server, {}, native=native)
    assert len(recording_server.requests) == 3
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await invoke(route, recording_server, {}, native=native)
    assert len(recording_server.requests) == 3
    litellm.cache.supported_call_types = excluded
    await invoke(route, recording_server, {}, native=native)
    assert len(recording_server.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize(
    ("route", "legacy"), (("chat", False), ("messages", False), ("responses", False), ("messages", True))
)
async def test_v2_default_off_requires_opt_in_even_for_existing_entries(
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    route: Literal["chat", "messages", "responses"],
    native: bool,
    legacy: bool,
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1" if native else "0")
    litellm.cache = Cache() if legacy else _v2.Cache.memory()
    litellm.cache.mode = CacheMode.default_off
    recording_server.expected_requests = 4
    await invoke(route, recording_server, {}, native=native)
    await invoke(route, recording_server, {}, native=native)
    assert len(recording_server.requests) == 2
    await invoke(route, recording_server, {"cache": {"use-cache": True}}, native=native)
    assert len(recording_server.requests) == 3
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await invoke(route, recording_server, {"cache": {"use-cache": True}}, native=native)
    assert len(recording_server.requests) == 3
    await invoke(route, recording_server, {}, native=native)
    assert len(recording_server.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "legacy"), (("chat", False), ("messages", False), ("responses", False), ("messages", True))
)
async def test_cache_lookup_uses_backend_request_callback_semantics(
    recording_server: RecordingServer,
    route: Literal["chat", "messages", "responses"],
    legacy: bool,
) -> None:
    from tests.test_litellm_rust.support.requests import request_body

    class Rewrite(RecordingLogger):
        temperature = 0.1

        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = self.temperature
            super().log_pre_api_call(model, messages, kwargs)

    logger: Final = Rewrite()
    litellm.cache = Cache() if legacy else _v2.Cache.memory()
    recording_server.expected_requests = 1 if legacy else 2
    await invoke(route, recording_server, {"callbacks": [logger]})
    first_hit: Final = await invoke(route, recording_server, {"callbacks": [logger]})
    logger.temperature = 0.8
    await invoke(route, recording_server, {"callbacks": [logger]})
    second_hit: Final = await invoke(route, recording_server, {"callbacks": [logger]})
    assert logger.names.count("log_pre_api_call") == 4
    assert len(recording_server.requests) == recording_server.expected_requests
    assert recording_server.requests[0].body["temperature"] == 0.1
    if not legacy:
        assert recording_server.requests[1].body["temperature"] == 0.8
    assert isinstance(cache_key(first_hit), str)
    assert isinstance(cache_key(second_hit), str)
    if not legacy:
        assert cache_key(first_hit) != cache_key(second_hit)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_lookup", (False, True))
async def test_python_cache_operations_stay_in_the_rust_callers_task(
    recording_server: RecordingServer,
    cancel_lookup: bool,
) -> None:
    from litellm.caching.base_cache import BaseCache
    from litellm.caching.in_memory_cache import InMemoryCache

    caller: Final = asyncio.current_task()
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    storage: Final = InMemoryCache()

    class CallerCache(BaseCache):
        async def async_set_cache_pipeline(
            self, cache_list: list[tuple[str, object]], ttl: float | None = None
        ) -> None:
            await storage.async_set_cache_pipeline(cache_list, ttl=ttl)

        async def async_get_cache(self, key: str, **kwargs: object) -> object:
            if cancel_lookup:
                entered.set()
                await release.wait()
            else:
                assert asyncio.current_task() is caller
            return storage.get_cache(key, **kwargs)

        async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
            assert asyncio.current_task() is caller
            await asyncio.sleep(0)
            storage.set_cache(key, value, **kwargs)

    litellm.cache = Cache(_backend=CallerCache())
    if cancel_lookup:
        recording_server.expected_requests = 0
        task: Final = asyncio.create_task(invoke("messages", recording_server, {}))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.sleep(0)
        assert len(recording_server.requests) == 0
        assert storage.cache_dict == {}
        return
    first: Final = await invoke("messages", recording_server, {})
    second: Final = await invoke("messages", recording_server, {})
    assert payload(first) == payload(second)
    assert cache_key(second)
    assert len(recording_server.requests) == 1


def test_sync_rust_messages_calls_python_cache(recording_server: RecordingServer) -> None:
    from litellm.rust_bridge.messages.entrypoints import NATIVE_MESSAGES

    litellm.cache = Cache()
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    arguments: Final = {
        "model": MESSAGES_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 32,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
    }
    request: Final = NativeCall(
        args=(),
        kwargs=arguments,
        base={
            "model": MESSAGES_MODEL,
            "messages": list(MESSAGES),
            "max_tokens": 32,
            "stream": None,
            "api_key": "test-key",
            "api_base": recording_server.base_url,
            "custom_llm_provider": "anthropic",
            **arguments,
        },
    )

    def call() -> object:
        return runtime.run(
            RouteContext(Route.MESSAGES),
            binding=NATIVE_MESSAGES,
            native=lambda hook: hook(request),
            python=runtime.NO_PYTHON,
            rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
        )

    first: Final = call()
    second: Final = call()
    assert payload(first) == payload(second)
    assert cache_key(second)
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace_source", ("cache", "metadata"))
async def test_rust_messages_legacy_cache_honors_request_namespaces(
    recording_server: RecordingServer, namespace_source: str
) -> None:
    litellm.cache = Cache()
    recording_server.expected_requests = 2
    first_options: Final = (
        {"cache": {"namespace": "first"}} if namespace_source == "cache" else {"metadata": {"redis_namespace": "first"}}
    )
    second_options: Final = (
        {"cache": {"namespace": "second"}}
        if namespace_source == "cache"
        else {"metadata": {"redis_namespace": "second"}}
    )
    await invoke("messages", recording_server, first_options)
    second: Final = await invoke("messages", recording_server, second_options)
    first_hit: Final = await invoke("messages", recording_server, first_options)
    second_hit: Final = await invoke("messages", recording_server, second_options)
    assert cache_key(second) is None
    assert cache_key(first_hit) is not None
    assert cache_key(second_hit) is not None
    assert len(recording_server.requests) == 2


@pytest.mark.asyncio
async def test_rust_messages_legacy_semantic_cache_preserves_python_scope(
    recording_server: RecordingServer,
) -> None:
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.types.caching import LiteLLMCacheType

    litellm.cache = Cache(type=LiteLLMCacheType.REDIS_SEMANTIC, _backend=InMemoryCache())
    first_options: Final = {"messages": [{"role": "user", "content": "hello"}]}
    second_options: Final = {"messages": [{"role": "user", "content": "hi"}]}
    first: Final = await invoke("messages", recording_server, first_options)
    second: Final = await invoke("messages", recording_server, second_options)
    assert cache_key(first) is None
    assert cache_key(second) is not None
    assert payload(first) == payload(second)
    assert len(recording_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rust_first", (False, True), ids=("python_to_rust", "rust_to_python"))
@pytest.mark.parametrize("stream", (False, True), ids=("response", "stream"))
async def test_legacy_cache_keeps_public_messages_responses_compatible(
    recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, rust_first: bool, stream: bool
) -> None:
    litellm.cache = Cache()
    monkeypatch.setenv("LITELLM_RUST", "0")
    options: Final = {"litellm_params": {"preset_cache_key": "shared-messages"}, "stream": stream}
    first: Final = await invoke("messages", recording_server, options, native=rust_first)
    first_payload: Final = await collect(first) if stream else payload(first)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    second: Final = await invoke("messages", recording_server, options, native=not rust_first)
    second_payload: Final = await collect(second) if stream else payload(second)
    assert second_payload == first_payload
    assert len(recording_server.requests) == 1
