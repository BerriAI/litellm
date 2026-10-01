from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import litellm
import pytest

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
    async def test_repeat_search_only_reembeds_the_query_and_oversized_texts(self) -> None:
        index = SemanticTextIndex()
        embedder = ContextWindowEmbedder()
        first: Final = await index.scores(QUERY, TEXTS, embedder, "emb")
        assert not isinstance(first, EmbeddingFailed)
        first_call_count: Final = len(embedder.calls)

        second: Final = await index.scores(QUERY, TEXTS, embedder, "emb")

        assert not isinstance(second, EmbeddingFailed)
        second_batches: Final = embedder.calls[first_call_count:]
        assert second_batches[0] == (QUERY, "OVERSIZED first", "OVERSIZED second")
        assert all(text in (QUERY, "OVERSIZED first", "OVERSIZED second") for batch in second_batches for text in batch)

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
