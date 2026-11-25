from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.redact_messages import should_redact_message_logging
from litellm.proxy import proxy_server
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.common_utils.semantic_text_index import EmbeddingFailed, SemanticTextIndex, Vector, router_embedder
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
)
from litellm.router import Router
from tests.unit.proxy.proxy_auth_state import ProxyAuthState

QUERY: Final = "weather query"
TEXTS: Final = (
    "valid one",
    "OVERSIZED first",
    "valid two",
    "valid three",
    "OVERSIZED second",
    "valid four",
    "valid five",
)


class ContextWindowEmbedder:
    def __init__(self, dimensions: int = 2) -> None:
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs
        self.vector: Final = (1.0,) * dimensions

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        if any("OVERSIZED" in text for text in texts):
            raise litellm.ContextWindowExceededError(
                message="input is too long", model="text-embedding-3-large", llm_provider="openai"
            )
        return tuple(self.vector for _ in texts)


class RateLimitedEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        raise litellm.RateLimitError(message="rate limited", model="text-embedding-3-large", llm_provider="openai")


class RateLimitedAfterSplitEmbedder:
    def __init__(self, rate_limited: bool = True) -> None:
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs
        self.rate_limited = rate_limited

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        if any("OVERSIZED" in text for text in texts):
            raise litellm.ContextWindowExceededError(
                message="input is too long", model="text-embedding-3-large", llm_provider="openai"
            )
        if self.rate_limited and any("RATE LIMITED" in text for text in texts):
            raise litellm.RateLimitError(message="rate limited", model="text-embedding-3-large", llm_provider="openai")
        return tuple((1.0, 1.0) for _ in texts)


class ConcurrencyTrackingEmbedder:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        if any("OVERSIZED" in text for text in texts):
            raise litellm.ContextWindowExceededError(
                message="input is too long", model="text-embedding-3-large", llm_provider="openai"
            )
        return tuple((1.0, 1.0) for _ in texts)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FixedDimensionEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.vector: Final = (1.0,) * dimensions

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self.vector for _ in texts)


class TestSemanticTextIndex:
    @pytest.mark.asyncio
    async def test_omits_multiple_oversized_texts_and_preserves_score_order(self) -> None:
        embedder = ContextWindowEmbedder()
        scores: Final = await SemanticTextIndex().scores(QUERY, TEXTS, embedder, "emb")

        assert not isinstance(scores, EmbeddingFailed)
        assert len(scores) == len(TEXTS)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] is None
        assert scores[2] == pytest.approx(1.0)
        assert scores[3] == pytest.approx(1.0)
        assert scores[4] is None
        assert scores[5] == pytest.approx(1.0)
        assert scores[6] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_repeat_search_only_embeds_the_query(self) -> None:
        index = SemanticTextIndex()
        embedder = ContextWindowEmbedder()
        first: Final = await index.scores(QUERY, TEXTS, embedder, "emb")
        assert not isinstance(first, EmbeddingFailed)
        first_call_count: Final = len(embedder.calls)

        second: Final = await index.scores(QUERY, TEXTS, embedder, "emb")

        assert not isinstance(second, EmbeddingFailed)
        assert embedder.calls[first_call_count:] == [(QUERY,)]
        assert second == first

    @pytest.mark.asyncio
    async def test_oversized_query_returns_embedding_failed(self) -> None:
        result: Final = await SemanticTextIndex().scores(
            "OVERSIZED query", ("valid text",), ContextWindowEmbedder(), "emb"
        )

        assert isinstance(result, EmbeddingFailed)

    @pytest.mark.asyncio
    async def test_non_context_provider_error_fails_after_one_embed_call(self) -> None:
        embedder = RateLimitedEmbedder()

        result: Final = await SemanticTextIndex().scores(QUERY, ("valid text",), embedder, "emb")

        assert isinstance(result, EmbeddingFailed)
        assert len(embedder.calls) == 1

    @pytest.mark.asyncio
    async def test_oversized_text_is_retried_once_the_exclusion_expires(self) -> None:
        clock = FakeClock()
        index = SemanticTextIndex(too_long_ttl_seconds=60.0, clock=clock)
        embedder = ContextWindowEmbedder()
        texts: Final = ("valid one", "OVERSIZED first")
        _ = await index.scores(QUERY, texts, embedder, "emb")
        clock.now = 59.0
        before_expiry: Final = len(embedder.calls)
        _ = await index.scores(QUERY, texts, embedder, "emb")
        assert embedder.calls[before_expiry:] == [(QUERY,)]

        clock.now = 61.0
        after_expiry: Final = len(embedder.calls)
        scores: Final = await index.scores(QUERY, texts, embedder, "emb")

        assert scores == (pytest.approx(1.0), None)
        assert embedder.calls[after_expiry] == (QUERY, "OVERSIZED first")

    @pytest.mark.asyncio
    async def test_splitting_sends_one_embedding_request_at_a_time(self) -> None:
        embedder = ConcurrencyTrackingEmbedder()

        scores: Final = await SemanticTextIndex().scores(QUERY, TEXTS, embedder, "emb")

        assert not isinstance(scores, EmbeddingFailed)
        assert embedder.max_in_flight == 1

    @pytest.mark.asyncio
    async def test_non_context_provider_error_while_splitting_fails_the_search(self) -> None:
        result: Final = await SemanticTextIndex().scores(
            QUERY, ("RATE LIMITED text", "OVERSIZED text", "valid text"), RateLimitedAfterSplitEmbedder(), "emb"
        )

        assert isinstance(result, EmbeddingFailed)
        assert "rate limited" in result.reason

    @pytest.mark.asyncio
    async def test_failed_split_keeps_vectors_and_too_long_texts_for_the_next_search(self) -> None:
        index = SemanticTextIndex()
        embedder = RateLimitedAfterSplitEmbedder()
        texts: Final = ("valid text", "OVERSIZED text", "RATE LIMITED text")

        first: Final = await index.scores(QUERY, texts, embedder, "emb")
        assert isinstance(first, EmbeddingFailed)
        assert "rate limited" in first.reason

        embedder.rate_limited = False
        before: Final = len(embedder.calls)
        second: Final = await index.scores(QUERY, texts, embedder, "emb")

        assert second == (pytest.approx(1.0), None, pytest.approx(1.0))
        assert embedder.calls[before:] == [(QUERY, "RATE LIMITED text")]

    @pytest.mark.asyncio
    async def test_mixed_dimension_reembedding_omits_oversized_text(self) -> None:
        index = SemanticTextIndex()
        seeded: Final = await index.scores(QUERY, ("cached text", "OVERSIZED text"), FixedDimensionEmbedder(3), "emb")
        assert not isinstance(seeded, EmbeddingFailed)
        embedder = ContextWindowEmbedder(2)

        scores: Final = await index.scores(QUERY, ("cached text", "OVERSIZED text"), embedder, "emb")

        assert not isinstance(scores, EmbeddingFailed)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] is None
        assert (QUERY, "cached text", "OVERSIZED text") in embedder.calls
        assert ("OVERSIZED text",) in embedder.calls


CALLER: Final = UserAPIKeyAuth(api_key="hashed-caller-key", team_id="team-1", user_id="user-1")


def _pass_through_key_limits() -> MagicMock:
    limits = MagicMock()
    limits.pre_call_hook = AsyncMock(side_effect=lambda user_api_key_dict, data, call_type: data)
    return limits


async def _embedding_response(
    model: str, input: Sequence[str], metadata: Mapping[str, object]
) -> litellm.EmbeddingResponse:
    return litellm.EmbeddingResponse(
        model=model, data=[{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i in range(len(input))]
    )


def _embedding_router() -> MagicMock:
    router = MagicMock()
    router.aembedding = AsyncMock(side_effect=_embedding_response)
    return router


def _priced_embedding_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "emb",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "sk-fake",
                    "input_cost_per_token": 0.001,
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_a_caller_over_budget_is_refused_the_embedding_sub_call(proxy_auth_state: ProxyAuthState) -> None:
    router: Final = _priced_embedding_router()
    proxy_auth_state.serve(router)
    caller: Final = UserAPIKeyAuth(api_key="sk-spent", spend=2.0, max_budget=1.0, request_route="/mcp-rest/tools/call")

    scores: Final = await SemanticTextIndex().scores(
        QUERY, ("valid one",), router_embedder(router, "emb", caller, _pass_through_key_limits()), "emb"
    )

    assert scores == EmbeddingFailed(
        reason="embedding the search query failed: emb is not available to this key (denied)"
    )


@pytest.mark.asyncio
async def test_a_key_over_its_model_budget_is_refused_the_embedding_sub_call(
    proxy_auth_state: ProxyAuthState, monkeypatch: pytest.MonkeyPatch
) -> None:
    router: Final = _priced_embedding_router()
    proxy_auth_state.serve(router)
    caller: Final = UserAPIKeyAuth(
        api_key="sk-model-spent",
        model_max_budget={"emb": {"max_budget": 0.1, "budget_duration": "1d"}},
        request_route="/mcp-rest/tools/call",
    )
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    await limiter.dual_cache.async_set_cache(
        key=model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.KEY, entity_id=caller.token, budget_model="emb", budget_duration="1d"
        ),
        value=1.0,
    )
    monkeypatch.setattr(proxy_server, "model_max_budget_limiter", limiter)

    scores: Final = await SemanticTextIndex().scores(
        QUERY, ("valid one",), router_embedder(router, "emb", caller, _pass_through_key_limits()), "emb"
    )

    assert scores == EmbeddingFailed(
        reason="embedding the search query failed: emb is not available to this key (denied)"
    )


@pytest.mark.asyncio
async def test_the_callers_redaction_header_turns_on_redaction_for_the_embedding_call() -> None:
    router: Final = _embedding_router()
    headers: Final = {"X-LiteLLM-Enable-Message-Redaction": "true"}

    await router_embedder(router, "emb", CALLER, _pass_through_key_limits(), headers)((QUERY,))

    metadata: Final = router.aembedding.await_args.kwargs["metadata"]
    assert should_redact_message_logging({"litellm_params": {"metadata": metadata}}) is True


@pytest.mark.parametrize(
    ("headers", "forwarded"),
    (
        (
            {"Authorization": "Bearer sk-caller", "x-litellm-api-key": "sk-caller", "cookie": "session=1"},
            {},
        ),
        (
            {
                "authorization": "Bearer sk-caller",
                "litellm-disable-message-redaction": "true",
                "litellm-enable-message-redaction": "true",
            },
            {"headers": {"litellm-enable-message-redaction": "true"}},
        ),
    ),
    ids=("credentials-only", "credentials-and-opt-out-dropped"),
)
@pytest.mark.asyncio
async def test_only_redaction_headers_reach_the_embedding_calls_metadata(
    headers: Mapping[str, str], forwarded: Mapping[str, object]
) -> None:
    router: Final = _embedding_router()

    await router_embedder(router, "emb", CALLER, _pass_through_key_limits(), headers)((QUERY,))

    metadata: Final = router.aembedding.await_args.kwargs["metadata"]
    assert {key: metadata[key] for key in metadata if key == "headers"} == forwarded
