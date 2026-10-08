import asyncio
import io
import json
from typing import Final

import litellm
import pytest
import respx
from litellm import CustomLogger, Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import RoutingStrategy
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import ImageResponse, RerankResponse
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_successes_for_current_minute,
)
from tests.unit.proxy.conftest import httpx_transport

pytestmark = pytest.mark.usefixtures(httpx_transport.__name__)


class _RouterLoggingCapture(CustomLogger):
    def __init__(self, expected_model_group: str | None = None) -> None:
        super().__init__()
        self.expected_model_group = expected_model_group
        self.success_events: asyncio.Queue[tuple[object | None, object | None]] = asyncio.Queue()

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if self.expected_model_group is not None:
            if not isinstance(standard_logging_object, dict):
                return
            model_group: Final[object] = standard_logging_object.get("model_group")
            if model_group != self.expected_model_group:
                return
        self.success_events.put_nowait((kwargs.get("client"), standard_logging_object))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment", "route_url"),
    [
        (
            {"model_name": "whisper", "litellm_params": {"model": "whisper-1", "api_key": "sk-fake"}},
            "https://api.openai.com/v1/audio/transcriptions",
        ),
        (
            {
                "model_name": "whisper",
                "litellm_params": {
                    "model": "azure/whisper",
                    "api_base": "https://azure.test",
                    "api_key": "sk-fake",
                    "api_version": "2025-02-01-preview",
                },
            },
            "https://azure.test/openai/deployments/whisper/audio/transcriptions?api-version=2025-02-01-preview",
        ),
    ],
)
async def test_router_transcription_dispatches_to_each_deployment(
    deployment: dict[str, object],
    route_url: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    capture: Final = _RouterLoggingCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = Router(model_list=[deployment])
    route: Final = respx_mock.post(route_url).respond(200, json={"text": "hello"})

    response: Final = await router.atranscription(
        model="whisper", file=("speech.wav", io.BytesIO(b"offline audio"), "audio/wav")
    )
    public_event: Final = await capture.success_events.get()
    standard_logging_object: Final = public_event[1]
    internal_response: Final = await router._atranscription(
        model="whisper", file=("speech.wav", io.BytesIO(b"offline audio"), "audio/wav")
    )
    upstream_requests: Final = tuple(call.request for call in route.calls)

    assert route.call_count == 2
    assert tuple(str(request.url) for request in upstream_requests) == (route_url, route_url)
    assert all(
        request.headers.get("content-type", "").startswith("multipart/form-data")
        and b'filename="speech.wav"' in request.content
        and b"offline audio" in request.content
        for request in upstream_requests
    )
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object.get("model_group") == "whisper"
    assert response.text == "hello"
    assert internal_response.text == "hello"


@pytest.mark.asyncio
async def test_router_speech_returns_binary_content_and_logs_model_group(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _RouterLoggingCapture(expected_model_group="tts")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {"model": "openai/tts-1", "api_key": "sk-fake"},
            }
        ]
    )
    route: Final = respx_mock.post("https://api.openai.com/v1/audio/speech").respond(200, content=b"audio")

    response: Final = await router.aspeech(model="tts", input="hello", voice="alloy")
    _, standard_logging_object = await capture.success_events.get()

    assert route.called
    assert isinstance(response, HttpxBinaryResponseContent)
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object["model_group"] == "tts"


@pytest.mark.asyncio
async def test_router_rerank_returns_valid_response_from_public_and_underlying_calls(
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "cohere-rerank",
                "litellm_params": {"model": "cohere/rerank-english-v3.0", "api_key": "sk-fake"},
            }
        ]
    )
    route: Final = respx_mock.post("https://api.cohere.com/v2/rerank").respond(
        200,
        json={
            "id": "rerank-1",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {"api_version": {"version": "2"}},
        },
    )

    public_response: Final = await router.arerank(
        model="cohere-rerank",
        query="hello",
        documents=["hello", "world"],
        top_n=1,
    )
    underlying_response: Final = await router._arerank(
        model="cohere-rerank",
        query="hello",
        documents=["hello", "world"],
        top_n=1,
    )

    assert route.call_count == 2
    public_validated: Final = RerankResponse.model_validate(public_response)
    assert public_validated.id == "rerank-1"
    assert public_validated.results[0]["relevance_score"] == 0.9
    underlying_validated: Final = RerankResponse.model_validate(underlying_response)
    assert underlying_validated.id == "rerank-1"
    assert underlying_validated.results[0]["relevance_score"] == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("omni-moderation-latest", "omni-moderation-latest"),
        ("openai/omni-moderation-latest", "omni-moderation-latest"),
        (None, None),
    ],
)
async def test_router_moderation_sends_resolved_model_and_input(
    model: str | None,
    expected_model: str | None,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-fake"},
            },
            {
                "model_name": "*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-fake"},
            },
        ]
    )
    route: Final = respx_mock.post("https://api.openai.com/v1/moderations").respond(
        200,
        json={
            "id": "modr-1",
            "model": "omni-moderation-latest",
            "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
        },
    )

    response: Final = await router.amoderation(model=model, input="hello")

    assert route.called
    request_body: Final = json.loads(route.calls[0].request.content)
    expected_body: Final = {"input": "hello"} if expected_model is None else {"input": "hello", "model": expected_model}
    assert request_body == expected_body
    assert response.id == "modr-1"
    assert response.model == "omni-moderation-latest"
    assert response.results[0].flagged is False


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_router_image_generation_returns_valid_image_response(
    sync_mode: bool,
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-image-1",
                "litellm_params": {"model": "openai/gpt-image-1", "api_key": "sk-fake"},
            }
        ]
    )
    route: Final = respx_mock.post("https://api.openai.com/v1/images/generations").respond(
        200,
        json={"created": 1700000000, "data": [{"url": "https://images.test/result.png"}]},
    )

    if sync_mode:
        response: Final = router._image_generation(model="gpt-image-1", prompt="a cat")
    else:
        response = await router._aimage_generation(model="gpt-image-1", prompt="a cat")

    assert route.called
    assert response.data[0].url == "https://images.test/result.png"
    ImageResponse.model_validate(response)


@pytest.mark.asyncio
async def test_router_stream_counts_request_before_headers_and_tokens_once_on_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _RouterLoggingCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": "lit-3058-stream"},
            }
        ]
    )
    stream: Final = await router.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong pong pong",
        stream=True,
        stream_options={"include_usage": True},
    )
    headers: Final = stream._hidden_params["additional_headers"]

    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert await router.get_model_group_usage("gpt-5-mini") == (0, 1)

    chunks: Final = [chunk async for chunk in stream]
    total_tokens: Final = chunks[-1].usage.total_tokens
    await capture.success_events.get()

    assert total_tokens > 0
    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)


def test_router_validate_fallbacks_rejects_malformed_entries() -> None:
    router: Final = Router(model_list=[])
    router.validate_fallbacks([{"primary": "fallback"}])
    with pytest.raises(ValueError, match="must have exactly one key"):
        router.validate_fallbacks([{"primary": "fallback", "other": "fallback"}])


@pytest.mark.parametrize(
    ("strategy", "expected_selector"),
    [
        (RoutingStrategy.LEAST_BUSY, "LeastBusyLoggingHandler"),
        (RoutingStrategy.LATENCY_BASED, "LowestLatencyLoggingHandler"),
        (RoutingStrategy.COST_BASED, "LowestCostLoggingHandler"),
        (RoutingStrategy.USAGE_BASED_ROUTING_V2, "LowestTPMLoggingHandler_v2"),
        (RoutingStrategy.USAGE_BASED_ROUTING, "LowestTPMLoggingHandler"),
        (RoutingStrategy.PROVIDER_BUDGET_LIMITING, None),
    ],
)
def test_router_routing_strategy_init_selects_expected_object(
    strategy: RoutingStrategy,
    expected_selector: str | None,
) -> None:
    router: Final = Router(model_list=[])

    router.routing_strategy_init(routing_strategy=strategy, routing_strategy_args={})

    normalized: Final = strategy.value
    attr_name: Final = router._DEFAULT_SELECTOR_ATTR_BY_STRATEGY.get(normalized)
    selector: Final = getattr(router, attr_name, None) if attr_name is not None else None
    assert selector is None if expected_selector is None else type(selector).__name__ == expected_selector


@pytest.mark.parametrize(
    "strategy",
    ["simple-shuffle", *(strategy.value for strategy in RoutingStrategy)],
)
def test_router_routing_strategy_init_accepts_valid_strings(strategy: str) -> None:
    router: Final = Router(model_list=[])
    router.routing_strategy_init(routing_strategy=strategy, routing_strategy_args={})
    assert router._normalize_strategy(strategy) == strategy


def test_router_track_deployment_metrics_updates_observable_usage() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake"},
                "model_info": {"id": "metrics-deployment"},
            }
        ]
    )
    deployment: Final = router.model_list[0]

    router._track_deployment_metrics(deployment=deployment, parent_otel_span=None)

    assert router.cache.get_cache(key="metrics-deployment", local_only=True) == 1


class _GatedIncrementCache(DualCache):
    def __init__(self) -> None:
        super().__init__(in_memory_cache=InMemoryCache())
        self.first_increment_started = asyncio.Event()
        self.release_first_increment = asyncio.Event()
        self.deployment_success_incremented = asyncio.Event()
        self.increment_calls = 0

    def increment_cache(self, key: str, value: int, local_only: bool = False, **kwargs: object) -> int:
        result: Final = super().increment_cache(key=key, value=value, local_only=local_only, **kwargs)
        if key.endswith(":successes"):
            self.deployment_success_incremented.set()
        return result

    async def async_increment_cache_pipeline(
        self,
        increment_list: list[RedisPipelineIncrementOperation],
        local_only: bool = False,
        parent_otel_span: object = None,
        **kwargs: object,
    ) -> list[float] | None:
        self.increment_calls += 1
        if self.increment_calls == 1:
            self.first_increment_started.set()
            await self.release_first_increment.wait()
        return await super().async_increment_cache_pipeline(
            increment_list=increment_list,
            local_only=local_only,
            parent_otel_span=parent_otel_span,
            **kwargs,
        )


class _UnavailableIncrementCache(DualCache):
    def __init__(self) -> None:
        super().__init__(in_memory_cache=InMemoryCache())
        self.first_increment_started = asyncio.Event()
        self.release_first_increment = asyncio.Event()
        self.deployment_success_incremented = asyncio.Event()
        self.increment_calls = 0

    def increment_cache(self, key: str, value: int, local_only: bool = False, **kwargs: object) -> int:
        result: Final = super().increment_cache(key=key, value=value, local_only=local_only, **kwargs)
        if key.endswith(":successes"):
            self.deployment_success_incremented.set()
        return result

    async def async_increment_cache_pipeline(
        self,
        increment_list: list[RedisPipelineIncrementOperation],
        local_only: bool = False,
        parent_otel_span: object = None,
        **kwargs: object,
    ) -> list[float] | None:
        self.increment_calls += 1
        if self.increment_calls == 1:
            self.first_increment_started.set()
            await self.release_first_increment.wait()
        raise RuntimeError("cache unavailable")


@pytest.mark.asyncio
async def test_router_success_callback_during_pre_header_increment_does_not_double_count() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": "lit-3058-race"},
            }
        ]
    )
    cache: Final = _GatedIncrementCache()
    router.cache = cache
    request: Final = asyncio.create_task(
        router.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="pong",
        )
    )

    await cache.first_increment_started.wait()
    await cache.deployment_success_incremented.wait()
    assert get_deployment_successes_for_current_minute(router, "lit-3058-race") == 1
    assert cache.increment_calls == 1
    cache.release_first_increment.set()
    response: Final = await request

    assert await router.get_model_group_usage("gpt-5-mini") == (response.usage.total_tokens, 1)


@pytest.mark.asyncio
async def test_router_failed_pre_header_increment_clears_counted_tokens_stamp() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": "lit-3058-fail"},
            }
        ]
    )
    cache: Final = _UnavailableIncrementCache()
    router.cache = cache
    metadata: Final = {}
    request: Final = asyncio.create_task(
        router.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="pong",
            metadata=metadata,
        )
    )

    await cache.first_increment_started.wait()
    await cache.deployment_success_incremented.wait()
    assert metadata[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY] == 30
    assert get_deployment_successes_for_current_minute(router, "lit-3058-fail") == 1
    assert cache.increment_calls == 1
    cache.release_first_increment.set()
    response: Final = await request

    assert response.usage.total_tokens == 30
    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in metadata
    assert response._hidden_params["additional_headers"]["x-ratelimit-remaining-requests"] == 100
    assert await router.get_model_group_usage("gpt-5-mini") == (None, None)


@pytest.mark.asyncio
async def test_router_assistants_endpoint_factory_invokes_provider(
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = Router(model_list=[])
    route: Final = respx_mock.post("https://api.openai.com/v1/assistants").respond(
        200,
        json={
            "id": "asst-1",
            "object": "assistant",
            "created_at": 1700000000,
            "name": "offline",
            "description": None,
            "model": "gpt-4o-mini",
            "instructions": "hello",
            "tools": [],
            "metadata": {},
        },
    )

    response: Final = await router._pass_through_assistants_endpoint_factory(
        original_function=litellm.acreate_assistants,
        custom_llm_provider="openai",
        model="gpt-4o-mini",
        api_key="sk-fake",
        name="offline",
    )

    assert route.called
    assert response.id == "asst-1"


@pytest.mark.asyncio
async def test_router_factory_function_returns_invokable_assistants_wrapper(
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = Router(model_list=[])
    route: Final = respx_mock.post("https://api.openai.com/v1/assistants").respond(
        200,
        json={
            "id": "asst-2",
            "object": "assistant",
            "created_at": 1700000000,
            "name": "offline",
            "description": None,
            "model": "gpt-4o-mini",
            "instructions": "hello",
            "tools": [],
            "metadata": {},
        },
    )
    wrapper: Final = router.factory_function(litellm.acreate_assistants, call_type="assistants")

    response: Final = await wrapper(
        custom_llm_provider="openai",
        model="gpt-4o-mini",
        api_key="sk-fake",
        name="offline",
    )

    assert route.called
    assert response.id == "asst-2"


@pytest.mark.asyncio
async def test_router_moderation_endpoint_factory_invokes_default_model(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    router: Final = Router(model_list=[])
    route: Final = respx_mock.post("https://api.openai.com/v1/moderations").respond(
        200,
        json={
            "id": "modr-2",
            "model": "omni-moderation-latest",
            "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
        },
    )

    response: Final = await router._pass_through_moderation_endpoint_factory(
        original_function=litellm.amoderation,
        custom_llm_provider="openai",
        input="hello",
        model=None,
        api_key="sk-fake",
    )

    assert route.called
    assert response.id == "modr-2"


def test_router_clientside_credential_requires_model_group_metadata() -> None:
    router: Final = Router(model_list=[])
    deployment: Final = {
        "model_name": "gpt-4.1",
        "litellm_params": {"model": "gpt-4.1", "api_key": "test-key"},
        "model_info": {"id": "original-id"},
    }
    kwargs_without_metadata: Final = {"api_key": "client-key", "api_base": "https://api.openai.com/v1"}
    kwargs_with_empty_metadata: Final = {**kwargs_without_metadata, "metadata": {}}

    with pytest.raises(TypeError):
        router._handle_clientside_credential(
            deployment=deployment,
            kwargs=kwargs_without_metadata,
            function_name="acompletion",
        )
    with pytest.raises(TypeError):
        router._handle_clientside_credential(
            deployment=deployment,
            kwargs=kwargs_with_empty_metadata,
            function_name="acompletion",
        )
