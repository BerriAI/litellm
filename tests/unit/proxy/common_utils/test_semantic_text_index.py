from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

import litellm
from litellm.proxy.common_utils.semantic_text_index import EmbeddingFailed, SemanticTextIndex, Vector

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
