from __future__ import annotations

import json
import math
from types import MappingProxyType
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
CL100K_MODEL: Final = "gpt-4"
O200K_MODEL: Final = "gpt-4o"
RUST_COUNTED_BODY: Final = {"model": ANTHROPIC_TOKENIZER_MODEL, "max_tokens": 16, "messages": ANTHROPIC_MESSAGES}
RUST_INPUT_TOKENS: Final = 4_321
RUST_INPUT_TOKENS_BY_TOKENIZER: Final = MappingProxyType(
    {"anthropic": RUST_INPUT_TOKENS, "cl100k_base": 1_234, "o200k_base": 2_345}
)


class _FakeDeclined(Exception):
    pass


class _FakeUpstream(Exception):
    pass


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    """Stands in for one native counter; records `(tokenizer, body)` on the shared factory."""

    def __init__(self, factory: _RecordingFactory, tokenizer: rust_token_counter.RustTokenizer) -> None:
        self.factory = factory
        self.tokenizer = tokenizer

    async def acount_request(self, body: bytes) -> object:
        self.factory.calls.append((self.tokenizer, body))
        return {"model": "", "input_tokens": RUST_INPUT_TOKENS_BY_TOKENIZER[self.tokenizer]}


class _RecordingFactory:
    """Stands in for the native `TokenCounter` class: called with tokenizer JSON, or `from_*_ranks`."""

    def __init__(self) -> None:
        self.calls: list[tuple[rust_token_counter.RustTokenizer, bytes]] = []

    def __call__(self, tokenizer_json: str) -> _RecordingCounter:
        return _RecordingCounter(self, "anthropic")

    def from_cl100k_ranks(self, rank_file: str) -> _RecordingCounter:
        return _RecordingCounter(self, "cl100k_base")

    def from_o200k_ranks(self, rank_file: str) -> _RecordingCounter:
        return _RecordingCounter(self, "o200k_base")


class _DecliningCounter:
    async def acount_request(self, body: bytes) -> object:
        raise _FakeDeclined("unsupported content block")


class _DecliningFactory:
    def __call__(self, tokenizer_json: str) -> _DecliningCounter:
        return _DecliningCounter()

    def from_cl100k_ranks(self, rank_file: str) -> _DecliningCounter:
        return _DecliningCounter()

    def from_o200k_ranks(self, rank_file: str) -> _DecliningCounter:
        return _DecliningCounter()


@pytest.fixture
def rust_counter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    rust_token_counter._counter.cache_clear()
    configuration.reset_rust_configuration()
    yield
    rust_token_counter.TOKEN_COUNTER.reset()
    rust_token_counter._counter.cache_clear()
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
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(factory)
    raw_body: Final = json.dumps(request_body).encode()

    counts: Final = await count_request_input_tokens(
        request_body=request_body, route=route, llm_router=None, raw_body=raw_body
    )

    assert dict(counts) == {ANTHROPIC_TOKENIZER_MODEL: RUST_INPUT_TOKENS}
    assert factory.calls == [("anthropic", raw_body)]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (CL100K_MODEL, "azure/gpt-35-turbo", "gemini/gemini-2.5-pro", "my-router-alias"))
async def test_tiktoken_cl100k_models_are_counted_by_rust(rust_counter: None, model: str) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(factory)
    body: Final = {"model": model, "messages": ANTHROPIC_MESSAGES}
    raw_body: Final = json.dumps(body).encode()

    counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None, raw_body=raw_body
    )

    assert dict(counts) == {model: RUST_INPUT_TOKENS_BY_TOKENIZER["cl100k_base"]}
    assert factory.calls == [("cl100k_base", raw_body)]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (O200K_MODEL, "gpt-5", "o3", "gpt-4.1", "chatgpt-4o-latest"))
async def test_tiktoken_o200k_models_are_counted_by_rust(rust_counter: None, model: str) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(factory)
    body: Final = {"model": model, "messages": ANTHROPIC_MESSAGES}
    raw_body: Final = json.dumps(body).encode()

    counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None, raw_body=raw_body
    )

    assert dict(counts) == {model: RUST_INPUT_TOKENS_BY_TOKENIZER["o200k_base"]}
    assert factory.calls == [("o200k_base", raw_body)]


@pytest.mark.asyncio
async def test_multi_model_request_counts_once_per_tokenizer_and_python_for_the_rest(rust_counter: None) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(factory)
    models: Final = (
        CL100K_MODEL,
        ANTHROPIC_TOKENIZER_MODEL,
        "gemini/gemini-2.5-pro",
        O200K_MODEL,
        "gpt-5",
        "replicate/meta/llama-2-70b-chat",
    )
    body: Final = {"model": list(models), "messages": ANTHROPIC_MESSAGES}
    raw_body: Final = json.dumps(body).encode()
    python_counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None
    )

    counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None, raw_body=raw_body
    )

    assert factory.calls == [("cl100k_base", raw_body), ("anthropic", raw_body), ("o200k_base", raw_body)]
    assert dict(counts) == {
        CL100K_MODEL: RUST_INPUT_TOKENS_BY_TOKENIZER["cl100k_base"],
        "gemini/gemini-2.5-pro": RUST_INPUT_TOKENS_BY_TOKENIZER["cl100k_base"],
        ANTHROPIC_TOKENIZER_MODEL: RUST_INPUT_TOKENS,
        O200K_MODEL: RUST_INPUT_TOKENS_BY_TOKENIZER["o200k_base"],
        "gpt-5": RUST_INPUT_TOKENS_BY_TOKENIZER["o200k_base"],
        "replicate/meta/llama-2-70b-chat": python_counts["replicate/meta/llama-2-70b-chat"],
    }
    assert counts["replicate/meta/llama-2-70b-chat"] not in RUST_INPUT_TOKENS_BY_TOKENIZER.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (ANTHROPIC_TOKENIZER_MODEL, CL100K_MODEL, O200K_MODEL))
async def test_rust_decline_falls_back_to_python_count(rust_counter: None, model: str) -> None:
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(_DecliningFactory())
    body: Final = {**RUST_COUNTED_BODY, "model": model}
    python_counts: Final = await count_request_input_tokens(request_body=body, route="/v1/messages", llm_router=None)

    counts: Final = await count_request_input_tokens(
        request_body=body,
        route="/v1/messages",
        llm_router=None,
        raw_body=json.dumps(body).encode(),
    )

    assert dict(counts) == dict(python_counts)
    assert counts[model] not in RUST_INPUT_TOKENS_BY_TOKENIZER.values()


@pytest.mark.asyncio
async def test_disabled_rust_never_sees_the_raw_body(rust_counter: None) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(False)
    rust_token_counter.TOKEN_COUNTER.override(factory)
    body: Final = {"model": [ANTHROPIC_TOKENIZER_MODEL, CL100K_MODEL, O200K_MODEL], "messages": ANTHROPIC_MESSAGES}

    counts: Final = await count_request_input_tokens(
        request_body=body,
        route="/v1/chat/completions",
        llm_router=None,
        raw_body=json.dumps(body).encode(),
    )

    assert factory.calls == []
    assert set(counts) == {ANTHROPIC_TOKENIZER_MODEL, CL100K_MODEL, O200K_MODEL}
    assert not set(counts.values()) & set(RUST_INPUT_TOKENS_BY_TOKENIZER.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("replicate/meta/llama-2-70b-chat", "meta-llama/Llama-3-8b", "text-davinci-003"))
async def test_models_without_a_rust_tokenizer_stay_in_python(
    rust_counter: None, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setattr(
        litellm, "open_ai_chat_completion_models", litellm.open_ai_chat_completion_models | {"text-davinci-003"}
    )
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    rust_token_counter.TOKEN_COUNTER.override(factory)
    body: Final = {"model": model, "messages": ANTHROPIC_MESSAGES}
    python_counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None
    )

    counts: Final = await count_request_input_tokens(
        request_body=body, route="/v1/chat/completions", llm_router=None, raw_body=json.dumps(body).encode()
    )

    assert factory.calls == []
    assert dict(counts) == dict(python_counts)
    assert counts[model] not in RUST_INPUT_TOKENS_BY_TOKENIZER.values()
