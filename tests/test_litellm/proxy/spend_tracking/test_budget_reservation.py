import json
import math
from typing import Final

import pytest

import litellm
from litellm.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import (
    count_request_input_tokens,
    estimate_request_max_cost,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import token_counter as rust_token_counter
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

TOKEN_COUNTING_ROUTES: Final = (
    "/responses/input_tokens",
    "/v1/responses/input_tokens",
    "/openai/v1/responses/input_tokens",
    "/utils/token_counter",
    "/v1/messages/count_tokens",
    "/v1beta/models/gemini-3.8-flash:countTokens",
    "/models/gemini-3.8-flash:countTokens",
    "/bedrock/v1/messages/count-tokens",
    "/bedrock/model/us.anthropic.claude-sonnet-4-6/count-tokens",
    "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
    "/vertex-ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
)


def _budgeted_token() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", token="hashed-token", max_budget=100.0, spend=0.0)


async def _reserve(route: str) -> dict | None:
    return await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "hello"},
        route=route,
        llm_router=None,
        valid_token=_budgeted_token(),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", TOKEN_COUNTING_ROUTES)
async def test_token_counting_routes_are_exempt_from_budget_reservation(route):
    assert await _reserve(route) is None


@pytest.mark.asyncio
async def test_non_exempt_llm_route_still_reserves_budget():
    reservation: Final = await _reserve("/v1/responses")

    assert reservation is not None
    assert reservation["reserved_cost"] > 0


@pytest.mark.asyncio
async def test_reservation_carries_the_admission_input_token_count():
    reservation: Final = await _reserve("/v1/responses")
    expected: Final = litellm.token_counter(model="gpt-4o", text="hello")

    assert reservation is not None
    assert expected > 0
    assert reservation["input_tokens"] == expected


ANTHROPIC_MESSAGES: Final = [{"role": "user", "content": "hello!!!"}]
COUNT_TOKENS_REQUESTS: Final[tuple[tuple[str, dict[str, object]], ...]] = (
    ("/v1/messages/count_tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
    ("/v1beta/models/gemini-3.8-flash:countTokens", {"contents": [{"role": "user", "parts": [{"text": "hello!!!"}]}]}),
    (
        "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
        {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES},
    ),
    ("/bedrock/v1/messages/count-tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
)
TINY_BUDGET_KEY_TOKEN: Final = "hashed-count-tokens-key"


@pytest.fixture
def spend_counter_cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return cache


async def _reserve_for_tiny_budget_key(route: str, request_body: dict[str, object]) -> dict[str, object] | None:
    return await reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=None,
        valid_token=UserAPIKeyAuth(token=TINY_BUDGET_KEY_TOKEN, max_budget=0.01, spend=0.0),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "request_body"), COUNT_TOKENS_REQUESTS)
async def test_repeated_token_counting_never_touches_a_tiny_budget(
    spend_counter_cache: DualCache, route: str, request_body: dict[str, object]
):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"

    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) is None

    completion: Final = await _reserve_for_tiny_budget_key(
        "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 16, "messages": ANTHROPIC_MESSAGES}
    )
    assert completion is not None
    reserved_cost: Final = completion["reserved_cost"]
    assert isinstance(reserved_cost, float)
    assert reserved_cost > 0
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reserved_cost)


BEDROCK_SONNET: Final = "us.anthropic.claude-sonnet-4-6"
CONVERSE_BODY: Final = {
    "messages": [{"role": "user", "content": [{"text": "Reply with one word: pong"}]}],
    "inferenceConfig": {"maxTokens": 5},
}
INVOKE_BODY: Final = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 5,
    "messages": [{"role": "user", "content": "Reply with one word: pong"}],
}


def test_bedrock_converse_body_reserves_the_prompt_not_the_context_window():
    converse_cost: Final = estimate_request_max_cost(
        request_body=CONVERSE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/converse",
        llm_router=None,
        input_token_counts={},
    )
    invoke_cost: Final = estimate_request_max_cost(
        request_body=INVOKE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/invoke",
        llm_router=None,
        input_token_counts={},
    )
    assert converse_cost is not None and invoke_cost is not None
    assert invoke_cost < converse_cost < 2 * invoke_cost


def _tiered_deployment(input_cost_per_token: float) -> Deployment:
    return Deployment(
        model_name="tiered-group",
        litellm_params=LiteLLM_Params(model="dashscope/qwen3-max", api_key="sk-fake"),
        model_info=ModelInfo(
            id="tiered-deployment",
            max_output_tokens=1000,
            tiered_pricing=[
                {
                    "input_cost_per_token": input_cost_per_token,
                    "output_cost_per_token": input_cost_per_token,
                    "range": [0, 128000],
                }
            ],
        ),
    )


TIERED_BODY: Final = {"model": "tiered-group", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


def test_repeated_estimates_reuse_cached_model_cost_info() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    first: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    hits_before: Final = router.cached_deployment_model_info.cache_info().hits

    second: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)

    assert second == first
    assert router.cached_deployment_model_info.cache_info().hits == hits_before + 1


def test_deployment_pricing_update_invalidates_cached_estimate() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    before: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert before is not None

    router.upsert_deployment(_tiered_deployment(1e-03))

    after: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert after is not None
    assert math.isclose(after, before * 1000)


ANTHROPIC_TOKENIZER_MODEL: Final = "claude-sonnet-4-5-20250929"
RUST_COUNTED_BODY: Final = {"model": ANTHROPIC_TOKENIZER_MODEL, "max_tokens": 16, "messages": ANTHROPIC_MESSAGES}
RUST_INPUT_TOKENS: Final = 4_321


class _FakeDeclined(Exception):
    pass


class _FakeUpstream(Exception):
    pass


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    bodies: Final[list[bytes]] = []

    def __init__(self, tokenizer_json: str) -> None:
        pass

    async def acount_request(self, body: bytes) -> object:
        self.bodies.append(body)
        return {"model": ANTHROPIC_TOKENIZER_MODEL, "input_tokens": RUST_INPUT_TOKENS}


class _DecliningCounter:
    def __init__(self, tokenizer_json: str) -> None:
        pass

    async def acount_request(self, body: bytes) -> object:
        raise _FakeDeclined("unsupported content block")


@pytest.fixture
def rust_counter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    rust_token_counter._anthropic_counter.cache_clear()
    configuration.reset_rust_configuration()
    _RecordingCounter.bodies.clear()
    yield
    rust_token_counter.TOKEN_COUNTER.reset()
    rust_token_counter._anthropic_counter.cache_clear()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "request_body"),
    (
        ("/v1/messages", RUST_COUNTED_BODY),
        ("/v1/chat/completions", {"model": ANTHROPIC_TOKENIZER_MODEL, "messages": ANTHROPIC_MESSAGES}),
        ("/v1/completions", {"model": ANTHROPIC_TOKENIZER_MODEL, "prompt": "hi"}),
        ("/v1/responses", {"model": ANTHROPIC_TOKENIZER_MODEL, "input": "hi"}),
        ("/v1/embeddings", {"model": ANTHROPIC_TOKENIZER_MODEL, "input": ["hi"]}),
        ("/v1/rerank", {"model": ANTHROPIC_TOKENIZER_MODEL, "query": "hi", "documents": ["a"]}),
    ),
)
async def test_rust_count_replaces_python_tokenizing_on_every_llm_route(
    rust_counter: None, route: str, request_body: dict
) -> None:
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(_RecordingCounter)
    raw_body: Final = json.dumps(request_body).encode()

    counts: Final = await count_request_input_tokens(
        request_body=request_body, route=route, llm_router=None, raw_body=raw_body
    )

    assert dict(counts) == {ANTHROPIC_TOKENIZER_MODEL: RUST_INPUT_TOKENS}
    assert _RecordingCounter.bodies == [raw_body]


@pytest.mark.asyncio
async def test_rust_decline_falls_back_to_python_count(rust_counter: None) -> None:
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(_DecliningCounter)
    python_counts: Final = await count_request_input_tokens(
        request_body=RUST_COUNTED_BODY, route="/v1/messages", llm_router=None
    )

    counts: Final = await count_request_input_tokens(
        request_body=RUST_COUNTED_BODY,
        route="/v1/messages",
        llm_router=None,
        raw_body=json.dumps(RUST_COUNTED_BODY).encode(),
    )

    assert dict(counts) == dict(python_counts)
    assert counts[ANTHROPIC_TOKENIZER_MODEL] != RUST_INPUT_TOKENS


@pytest.mark.asyncio
async def test_disabled_rust_never_sees_the_raw_body(rust_counter: None) -> None:
    litellm.rust(False)
    rust_token_counter.TOKEN_COUNTER.override(_RecordingCounter)

    counts: Final = await count_request_input_tokens(
        request_body=RUST_COUNTED_BODY,
        route="/v1/messages",
        llm_router=None,
        raw_body=json.dumps(RUST_COUNTED_BODY).encode(),
    )

    assert _RecordingCounter.bodies == []
    assert counts[ANTHROPIC_TOKENIZER_MODEL] != RUST_INPUT_TOKENS


@pytest.mark.asyncio
async def test_non_anthropic_tokenizer_models_stay_in_python(rust_counter: None) -> None:
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(_RecordingCounter)
    body: Final = {"model": "gpt-4o", "messages": ANTHROPIC_MESSAGES}

    counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None, raw_body=json.dumps(body).encode()
    )

    assert _RecordingCounter.bodies == []
    assert counts["gpt-4o"] != RUST_INPUT_TOKENS
