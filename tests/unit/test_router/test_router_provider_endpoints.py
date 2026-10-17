import asyncio
import contextlib
import io
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Final

import httpx
import litellm
import pytest
import respx
from litellm import CustomLogger, Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_successes_for_current_minute,
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import RoutingStrategy
from litellm.types.utils import ImageResponse, ModelResponse, RerankResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI

EVENT_TIMEOUT_SECONDS: Final = 5
ROUTING_SELECTIONS: Final = 40
ROUTING_MESSAGES: Final = ({"role": "user", "content": "route this request"},)
EXPENSIVE_COSTS: Final = {"input_cost_per_token": 1.0, "output_cost_per_token": 1.0}
CHEAP_COSTS: Final = {"input_cost_per_token": 1e-9, "output_cost_per_token": 1e-9}
CHAT_RESPONSE: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("request_call_type", [None, "aimage_edit", "image_edit"])
async def test_router_image_edit_accepts_call_type_metadata_without_duplicate_dispatch(
    respx_mock: respx.MockRouter, request_call_type: str | None
) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": "image-edit", "litellm_params": {"model": "openai/gpt-image-2", "api_key": "sk-fake"}}
        ],
        num_retries=0,
    )
    image: Final = b"image-edit-reference"
    encoded: Final = "aW1hZ2UtZWRpdC1yZXN1bHQ="
    route: Final = respx_mock.post("https://api.openai.com/v1/images/edits").respond(
        200, json={"created": 1, "data": [{"b64_json": encoded}]}
    )
    metadata: Final = {} if request_call_type is None else {"call_type": request_call_type}

    response: Final = await router.aimage_edit(
        model="image-edit", image=io.BytesIO(image), prompt="make it blue", **metadata
    )

    assert route.call_count == 1
    assert image in route.calls[0].request.content
    assert b"make it blue" in route.calls[0].request.content
    assert response.data is not None
    assert response.data[0].b64_json == encoded


@pytest.fixture(autouse=True)
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def router_minute_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned: Final = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("litellm.router.get_utc_datetime", lambda: pinned)


class _RouterLoggingCapture(CustomLogger):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id: Final = model_id
        self.success_events: asyncio.Queue[tuple[object | None, object | None]] = asyncio.Queue()

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if not isinstance(standard_logging_object, dict) or standard_logging_object.get("model_id") != self.model_id:
            return
        self.success_events.put_nowait((kwargs.get("client"), standard_logging_object))

    async def next_event(self) -> tuple[object | None, object | None]:
        return await asyncio.wait_for(self.success_events.get(), timeout=EVENT_TIMEOUT_SECONDS)


def _rpm_tpm_router(model_id: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-fake", "tpm": 1000, "rpm": 100},
                "model_info": {"id": model_id},
            }
        ]
    )


def _ratelimit_headers(response: ModelResponse | CustomStreamWrapper) -> dict[str, int]:
    return {k: v for k, v in response._hidden_params["additional_headers"].items() if k.startswith("x-ratelimit-")}


def _openai_router_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="sk-fake")


def _azure_router_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(api_key="sk-fake", azure_endpoint="https://azure.test", api_version="2025-02-01-preview")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("litellm_params", "build_client", "route_url"),
    [
        (
            {"model": "whisper-1", "api_key": "sk-fake"},
            _openai_router_client,
            "https://api.openai.com/v1/audio/transcriptions",
        ),
        (
            {
                "model": "azure/whisper",
                "api_base": "https://azure.test",
                "api_key": "sk-fake",
                "api_version": "2025-02-01-preview",
            },
            _azure_router_client,
            "https://azure.test/openai/deployments/whisper/audio/transcriptions?api-version=2025-02-01-preview",
        ),
    ],
    ids=["openai", "azure"],
)
async def test_router_transcription_reuses_router_level_client_for_each_deployment(
    litellm_params: dict[str, str],
    build_client: Callable[[], AsyncOpenAI],
    route_url: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    model_id: Final = f"whisper-{uuid.uuid4().hex}"
    capture: Final = _RouterLoggingCapture(model_id)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = Router(
        model_list=[{"model_name": "whisper", "litellm_params": dict(litellm_params), "model_info": {"id": model_id}}]
    )
    router_level_client: Final = build_client()
    router.cache.set_cache(key=f"{model_id}_async_client", value=router_level_client, local_only=True)
    route: Final = respx_mock.post(route_url).respond(200, json={"text": "hello"})

    response: Final = await router.atranscription(
        model="whisper", file=("speech.wav", io.BytesIO(b"offline audio"), "audio/wav")
    )
    public_client, public_logging = await capture.next_event()
    internal_response: Final = await router._atranscription(
        model="whisper", file=("speech.wav", io.BytesIO(b"offline audio"), "audio/wav")
    )
    internal_client, _ = await capture.next_event()
    upstream_requests: Final = tuple(call.request for call in route.calls)

    assert public_client == str(router_level_client)
    assert internal_client == str(router_level_client)
    assert tuple(str(request.url) for request in upstream_requests) == (route_url, route_url)
    assert all(
        request.headers.get("content-type", "").startswith("multipart/form-data")
        and b'filename="speech.wav"' in request.content
        and b"offline audio" in request.content
        for request in upstream_requests
    )
    assert isinstance(public_logging, dict)
    assert public_logging.get("model_group") == "whisper"
    assert response.text == "hello"
    assert internal_response.text == "hello"


@pytest.mark.asyncio
async def test_router_speech_returns_binary_content_and_logs_model_group(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id: Final = f"tts-{uuid.uuid4().hex}"
    capture: Final = _RouterLoggingCapture(model_id)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {"model": "openai/tts-1", "api_key": "sk-fake"},
                "model_info": {"id": model_id},
            }
        ]
    )
    route: Final = respx_mock.post("https://api.openai.com/v1/audio/speech").respond(200, content=b"audio")

    response: Final = await router.aspeech(model="tts", input="hello", voice="alloy")
    _, standard_logging_object = await capture.next_event()

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {"model": "tts-1", "input": "hello", "voice": "alloy"}
    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.content == b"audio"
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
    request_bodies: Final = tuple(json.loads(call.request.content) for call in route.calls)
    assert all(
        body["model"] == "rerank-english-v3.0"
        and body["query"] == "hello"
        and body["documents"] == ["hello", "world"]
        and body["top_n"] == 1
        for body in request_bodies
    )
    public_validated: Final = RerankResponse.model_validate(public_response)
    assert public_validated.id == "rerank-1"
    assert public_validated.results[0]["relevance_score"] == 0.9
    underlying_validated: Final = RerankResponse.model_validate(underlying_response)
    assert underlying_validated.id == "rerank-1"
    assert underlying_validated.results[0]["relevance_score"] == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_model", "expected_api_key"),
    [
        ("omni-moderation-latest", "omni-moderation-latest", "sk-catch-all"),
        ("openai/omni-moderation-latest", "omni-moderation-latest", "sk-openai-wildcard"),
        (None, None, "sk-env"),
    ],
)
async def test_router_moderation_routes_through_wildcard_deployments(
    model: str | None,
    expected_model: str | None,
    expected_api_key: str,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-openai-wildcard"},
            },
            {
                "model_name": "*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-catch-all"},
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

    assert route.call_count == 1
    upstream_request: Final = route.calls[0].request
    expected_body: Final = {"input": "hello"} if expected_model is None else {"input": "hello", "model": expected_model}
    assert json.loads(upstream_request.content) == expected_body
    assert upstream_request.headers["authorization"] == f"Bearer {expected_api_key}"
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

    response: Final = (
        router._image_generation(model="gpt-image-1", prompt="a cat")
        if sync_mode
        else await router._aimage_generation(model="gpt-image-1", prompt="a cat")
    )

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["model"] == "gpt-image-1"
    assert request_body["prompt"] == "a cat"
    validated: Final = ImageResponse.model_validate(response)
    assert validated.data[0].url == "https://images.test/result.png"


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_router_acompletion_headers_read_post_increment_counter_and_count_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _RouterLoggingCapture("lit-3058-async")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = _rpm_tpm_router("lit-3058-async")

    response: Final = await router.acompletion(
        model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}], mock_response="pong"
    )
    total_tokens: Final = response.usage.total_tokens
    headers: Final = _ratelimit_headers(response)

    assert total_tokens > 0
    assert headers["x-ratelimit-remaining-tokens"] == 1000 - total_tokens
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)

    await capture.next_event()

    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_router_stream_counts_request_before_headers_and_tokens_once_on_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture: Final = _RouterLoggingCapture("lit-3058-stream")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    router: Final = _rpm_tpm_router("lit-3058-stream")
    stream: Final = await router.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="pong pong pong",
        stream=True,
        stream_options={"include_usage": True},
    )
    headers: Final = _ratelimit_headers(stream)

    assert headers["x-ratelimit-remaining-tokens"] == 1000
    assert headers["x-ratelimit-remaining-requests"] == 99
    assert await router.get_model_group_usage("gpt-5-mini") == (0, 1)

    chunks: Final = [chunk async for chunk in stream]
    total_tokens: Final = chunks[-1].usage.total_tokens
    await capture.next_event()

    assert total_tokens > 0
    assert await router.get_model_group_usage("gpt-5-mini") == (total_tokens, 1)


def test_router_validate_fallbacks_accepts_well_formed_and_rejects_malformed_entries() -> None:
    router: Final = Router(model_list=[])

    assert router.validate_fallbacks([{"gpt-5.5": ["gpt-5-mini"]}, {"gpt-5-mini": ["gpt-5.5"]}]) is None
    with pytest.raises(ValueError, match="must have exactly one key"):
        router.validate_fallbacks([{"primary": "fallback", "other": "fallback"}])
    with pytest.raises(ValueError, match="is not a dictionary"):
        router.validate_fallbacks(["primary"])


def _routing_deployment(deployment_id: str, extra_params: dict[str, float]) -> dict[str, object]:
    return {
        "model_name": "gpt",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-fake",
            "api_base": f"https://{deployment_id}.test/v1",
            **extra_params,
        },
        "model_info": {"id": deployment_id},
    }


def _routing_router(
    strategy: RoutingStrategy | str,
    a_params: dict[str, float] | None = None,
    b_params: dict[str, float] | None = None,
) -> Router:
    return Router(
        model_list=[_routing_deployment("a", a_params or {}), _routing_deployment("b", b_params or {})],
        routing_strategy=strategy,
        disable_cooldowns=True,
        num_retries=0,
    )


async def _selected_deployment_ids(router: Router) -> frozenset[str]:
    deployments: Final = [
        await router.async_get_available_deployment(model="gpt", messages=list(ROUTING_MESSAGES), request_kwargs={})
        for _ in range(ROUTING_SELECTIONS)
    ]
    return frozenset(deployment["model_info"]["id"] for deployment in deployments)


def _enum_and_string(strategy: RoutingStrategy) -> list[RoutingStrategy | str]:
    return [strategy, strategy.value]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", _enum_and_string(RoutingStrategy.COST_BASED))
async def test_router_cost_based_routing_selects_cheapest_deployment(strategy: RoutingStrategy | str) -> None:
    router: Final = _routing_router(strategy, EXPENSIVE_COSTS, CHEAP_COSTS)

    assert await _selected_deployment_ids(router) == frozenset({"b"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    [*_enum_and_string(RoutingStrategy.USAGE_BASED_ROUTING), *_enum_and_string(RoutingStrategy.USAGE_BASED_ROUTING_V2)],
)
async def test_router_usage_based_routing_skips_deployment_over_tpm_limit(strategy: RoutingStrategy | str) -> None:
    router: Final = _routing_router(strategy, {"tpm": 1}, {"tpm": 1_000_000})

    assert await _selected_deployment_ids(router) == frozenset({"b"})


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", _enum_and_string(RoutingStrategy.LATENCY_BASED))
async def test_router_latency_based_routing_avoids_deployment_that_timed_out(
    strategy: RoutingStrategy | str,
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = _routing_router(strategy)
    timed_out: Final = respx_mock.post("https://a.test/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("upstream timed out")
    )
    respx_mock.post("https://b.test/v1/chat/completions").respond(200, json=CHAT_RESPONSE)

    for _ in range(5):
        if timed_out.called:
            break
        with contextlib.suppress(litellm.Timeout):
            await router.acompletion(model="gpt", messages=list(ROUTING_MESSAGES), max_retries=0)

    assert timed_out.called
    assert await _selected_deployment_ids(router) == frozenset({"b"})


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", _enum_and_string(RoutingStrategy.LEAST_BUSY))
async def test_router_least_busy_routing_avoids_deployment_with_request_in_flight(
    strategy: RoutingStrategy | str,
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = _routing_router(strategy)
    upstream_hosts: Final = asyncio.Queue[str]()
    release_upstream: Final = asyncio.Event()

    async def hold_request(request: httpx.Request) -> httpx.Response:
        upstream_hosts.put_nowait(request.url.host.split(".")[0])
        await release_upstream.wait()
        return httpx.Response(200, json=CHAT_RESPONSE)

    respx_mock.post(url__regex=r"https://[ab]\.test/v1/chat/completions").mock(side_effect=hold_request)
    in_flight: Final = asyncio.create_task(router.acompletion(model="gpt", messages=list(ROUTING_MESSAGES)))
    try:
        busy_id: Final = await asyncio.wait_for(upstream_hosts.get(), timeout=EVENT_TIMEOUT_SECONDS)
        selected: Final = await _selected_deployment_ids(router)
    finally:
        release_upstream.set()
        await in_flight

    assert selected == frozenset({"a", "b"} - {busy_id})


@pytest.mark.asyncio
async def test_router_simple_shuffle_ignores_cost_and_spreads_across_deployments() -> None:
    router: Final = _routing_router("simple-shuffle", EXPENSIVE_COSTS, CHEAP_COSTS)

    assert await _selected_deployment_ids(router) == frozenset({"a", "b"})


@pytest.mark.parametrize("strategy", _enum_and_string(RoutingStrategy.PROVIDER_BUDGET_LIMITING))
def test_router_routing_strategy_init_accepts_provider_budget_strategy(strategy: RoutingStrategy | str) -> None:
    router: Final = _routing_router(strategy)

    router.routing_strategy_init(routing_strategy=strategy, routing_strategy_args={})

    assert router.get_settings()["routing_strategy"] == "provider-budget-routing"


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
    router._track_deployment_metrics(deployment=deployment, parent_otel_span=None)

    assert router.cache.get_cache(key="metrics-deployment", local_only=True) == 2


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
@pytest.mark.usefixtures("router_minute_pinned")
async def test_router_success_callback_during_pre_header_increment_does_not_double_count() -> None:
    router: Final = _rpm_tpm_router("lit-3058-race")
    cache: Final = _GatedIncrementCache()
    router.cache = cache
    request: Final = asyncio.create_task(
        router.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="pong",
        )
    )

    await asyncio.wait_for(cache.first_increment_started.wait(), timeout=EVENT_TIMEOUT_SECONDS)
    await asyncio.wait_for(cache.deployment_success_incremented.wait(), timeout=EVENT_TIMEOUT_SECONDS)
    assert get_deployment_successes_for_current_minute(router, "lit-3058-race") == 1
    assert cache.increment_calls == 1
    cache.release_first_increment.set()
    response: Final = await request

    assert await router.get_model_group_usage("gpt-5-mini") == (response.usage.total_tokens, 1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("router_minute_pinned")
async def test_router_failed_pre_header_increment_clears_counted_tokens_stamp() -> None:
    router: Final = _rpm_tpm_router("lit-3058-fail")
    cache: Final = _UnavailableIncrementCache()
    router.cache = cache
    metadata: Final[dict[str, object]] = {}
    request: Final = asyncio.create_task(
        router.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="pong",
            metadata=metadata,
        )
    )

    await asyncio.wait_for(cache.first_increment_started.wait(), timeout=EVENT_TIMEOUT_SECONDS)
    await asyncio.wait_for(cache.deployment_success_incremented.wait(), timeout=EVENT_TIMEOUT_SECONDS)
    assert metadata[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY] == 30
    assert get_deployment_successes_for_current_minute(router, "lit-3058-fail") == 1
    assert cache.increment_calls == 1
    cache.release_first_increment.set()
    response: Final = await request

    assert response.usage.total_tokens == 30
    assert ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY not in metadata
    assert _ratelimit_headers(response)["x-ratelimit-remaining-requests"] == 100
    assert await router.get_model_group_usage("gpt-5-mini") == (None, None)


ASSISTANT_RESPONSE: Final = {
    "object": "assistant",
    "created_at": 1700000000,
    "name": "offline",
    "description": None,
    "model": "gpt-4o-mini",
    "instructions": "hello",
    "tools": [],
    "metadata": {},
}


@pytest.mark.asyncio
async def test_router_assistants_endpoint_factory_invokes_provider(
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = Router(model_list=[])
    route: Final = respx_mock.post("https://api.openai.com/v1/assistants").respond(
        200, json={**ASSISTANT_RESPONSE, "id": "asst-1"}
    )

    response: Final = await router._pass_through_assistants_endpoint_factory(
        original_function=litellm.acreate_assistants,
        custom_llm_provider="openai",
        model="gpt-4o-mini",
        api_key="sk-fake",
        name="offline",
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {"model": "gpt-4o-mini", "name": "offline"}
    assert response.id == "asst-1"


@pytest.mark.asyncio
async def test_router_factory_function_returns_invokable_assistants_wrapper(
    respx_mock: respx.MockRouter,
) -> None:
    router: Final = Router(model_list=[])
    route: Final = respx_mock.post("https://api.openai.com/v1/assistants").respond(
        200, json={**ASSISTANT_RESPONSE, "id": "asst-2"}
    )
    wrapper: Final = router.factory_function(litellm.acreate_assistants, call_type="assistants")

    response: Final = await wrapper(
        custom_llm_provider="openai",
        model="gpt-4o-mini",
        api_key="sk-fake",
        name="offline",
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {"model": "gpt-4o-mini", "name": "offline"}
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

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {"input": "hello"}
    assert response.id == "modr-2"
