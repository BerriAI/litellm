"""Embedding-similarity ranking over short texts with a per-model vector cache, shared by agent search and MCP tool search."""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from fastapi import HTTPException
from openai import OpenAIError
from pydantic import ConfigDict

from litellm.exceptions import BudgetExceededError, ContextWindowExceededError
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

Vector: TypeAlias = tuple[float, ...]

DEFAULT_MAX_CACHED_VECTORS: Final = 5000
"""Ceiling on how many (embedding model, text) vectors one index keeps; the least recently searched are evicted first."""
DEFAULT_TOO_LONG_TTL_SECONDS: Final = 3600.0


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> Awaitable[Sequence[Vector]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingFailed:
    reason: str


@dataclass(frozen=True, slots=True)
class _InputTooLong:
    reason: str


@dataclass(frozen=True, slots=True)
class _TextVectors:
    vectors: Mapping[str, Vector]
    too_long: frozenset[str]
    failure: EmbeddingFailed | None = None


class _EmbeddingItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    embedding: tuple[float, ...]


class _EmbeddingData(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_EmbeddingItem, ...]


class _EmbeddingRequest(LiteLLMBaseModel):
    """The /embeddings-shaped request as the pre-call hooks (rate limits, budgets, guardrails) hand it back."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str
    input: tuple[str, ...]
    metadata: dict[str, object]  # mutable-ok: the router mutates the metadata dict it is handed


def cosine_similarity(left: Vector, right: Vector) -> float:
    dot: Final = sum(a * b for a, b in zip(left, right, strict=True))
    norms: Final = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def embedding_spend_metadata(user_api_key_dict: UserAPIKeyAuth) -> dict[str, object]:  # mutable-ok: router mutates it
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    return {
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(user_api_key_dict),
        "user_api_key": LiteLLMProxyRequestSetup.get_logged_api_key(user_api_key_dict),
    }


def router_embedder(
    router: Router, embedding_model: str, user_api_key_dict: UserAPIKeyAuth, proxy_logging_obj: ProxyLogging
) -> Embedder:
    """Embeds through the router after the same key rate-limit, budget and guardrail pre-call hooks /embeddings runs."""

    async def embed(texts: Sequence[str]) -> Sequence[Vector]:
        request: Final = {
            "model": embedding_model,
            "input": list(texts),
            "metadata": embedding_spend_metadata(user_api_key_dict),
        }
        processed: Final = _EmbeddingRequest.model_validate(
            await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict, data=request, call_type="aembedding"
            )
        )
        response: Final = await router.aembedding(
            model=processed.model,
            input=list(processed.input),
            metadata=processed.metadata,
        )
        return tuple(item.embedding for item in _EmbeddingData.model_validate(response.model_dump()).data)

    return embed


_CacheKey: TypeAlias = tuple[str, str]


async def _embed_all(embed: Embedder, texts: Sequence[str]) -> tuple[Vector, ...] | EmbeddingFailed | _InputTooLong:
    try:
        vectors: Final = tuple(await embed(texts))
    except HTTPException:
        raise
    except ContextWindowExceededError as exc:
        return _InputTooLong(reason=f"embedding the search query failed: {exc}")
    except (OpenAIError, ValueError, BudgetExceededError) as exc:
        return EmbeddingFailed(reason=f"embedding the search query failed: {exc}")
    if len(vectors) != len(texts):
        return EmbeddingFailed(reason=f"embedding model returned {len(vectors)} vectors for {len(texts)} inputs")
    return vectors


async def _embed_texts(embed: Embedder, texts: Sequence[str]) -> _TextVectors:
    if not texts:
        return _TextVectors(vectors=MappingProxyType({}), too_long=frozenset())
    embedded: Final = await _embed_all(embed, texts)
    if isinstance(embedded, EmbeddingFailed):
        return _TextVectors(vectors=MappingProxyType({}), too_long=frozenset(), failure=embedded)
    if not isinstance(embedded, _InputTooLong):
        return _TextVectors(vectors=MappingProxyType(dict(zip(texts, embedded, strict=True))), too_long=frozenset())
    if len(texts) == 1:
        return _TextVectors(vectors=MappingProxyType({}), too_long=frozenset({texts[0]}))
    middle: Final = len(texts) // 2
    left: Final = await _embed_texts(embed, texts[:middle])
    if left.failure is not None:
        return left
    right: Final = await _embed_texts(embed, texts[middle:])
    return _TextVectors(
        vectors=MappingProxyType({**left.vectors, **right.vectors}),
        too_long=left.too_long | right.too_long,
        failure=right.failure,
    )


async def _embed_query_with(
    embed: Embedder, query: str, texts: Sequence[str]
) -> tuple[Vector, _TextVectors] | EmbeddingFailed:
    if not texts:
        embedded_query: Final = await _embed_all(embed, (query,))
        if isinstance(embedded_query, EmbeddingFailed):
            return embedded_query
        if isinstance(embedded_query, _InputTooLong):
            return EmbeddingFailed(reason=embedded_query.reason)
        return embedded_query[0], _TextVectors(vectors=MappingProxyType({}), too_long=frozenset())

    embedded: Final = await _embed_all(embed, (query, *texts))
    if isinstance(embedded, EmbeddingFailed):
        return embedded
    if not isinstance(embedded, _InputTooLong):
        return embedded[0], _TextVectors(
            vectors=MappingProxyType(dict(zip(texts, embedded[1:], strict=True))), too_long=frozenset()
        )
    query_embedding: Final = await _embed_all(embed, (query,))
    if isinstance(query_embedding, EmbeddingFailed):
        return query_embedding
    if isinstance(query_embedding, _InputTooLong):
        return EmbeddingFailed(reason=query_embedding.reason)
    vectors: Final = await _embed_texts(embed, texts)
    return query_embedding[0], vectors


@dataclass(frozen=True, slots=True)
class _Embedded:
    query_vector: Vector
    vectors: Mapping[str, Vector]
    too_long: frozenset[str]
    failure: EmbeddingFailed | None


def _same_dimension(query_vector: Vector, vectors: Mapping[str, Vector], texts: Sequence[str]) -> bool:
    return all(len(vectors[text]) == len(query_vector) for text in texts if text in vectors)


async def _embed_query_and_texts(
    embed: Embedder, query: str, texts: Sequence[str], cached: Mapping[str, Vector], too_long: frozenset[str]
) -> _Embedded | EmbeddingFailed:
    pending: Final = (text for text in texts if text not in cached and text not in too_long)
    missing: Final = tuple(dict.fromkeys(pending))
    query_embedding: Final = await _embed_query_with(embed, query, missing)
    if isinstance(query_embedding, EmbeddingFailed):
        return query_embedding
    query_vector, new_text_vectors = query_embedding
    vectors: Final = MappingProxyType(dict(chain(cached.items(), new_text_vectors.vectors.items())))
    embedded: Final = _Embedded(
        query_vector=query_vector,
        vectors=vectors,
        too_long=new_text_vectors.too_long,
        failure=new_text_vectors.failure,
    )
    if embedded.failure is not None:
        return embedded
    if _same_dimension(query_vector, vectors, texts):
        return embedded
    unique: Final = tuple(dict.fromkeys(text for text in texts if text in vectors))
    reembedded: Final = await _embed_query_with(embed, query, unique)
    if isinstance(reembedded, EmbeddingFailed):
        return reembedded
    reembedded_query, reembedded_text_vectors = reembedded
    return _Embedded(
        query_vector=reembedded_query,
        vectors=reembedded_text_vectors.vectors,
        too_long=new_text_vectors.too_long | reembedded_text_vectors.too_long,
        failure=reembedded_text_vectors.failure,
    )


class SemanticTextIndex:
    """Caches one vector per distinct text per embedding model, so repeat searches only embed the query.

    Holds at most ``max_entries`` vectors across all models: once full, the texts no recent search touched go first."""

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_CACHED_VECTORS,
        too_long_ttl_seconds: float = DEFAULT_TOO_LONG_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries: Final = max_entries
        self._too_long_ttl_seconds: Final = too_long_ttl_seconds
        self._clock: Final = clock
        self._vectors: Mapping[_CacheKey, Vector] = MappingProxyType({})
        self._too_long: Mapping[_CacheKey, float] = MappingProxyType({})

    def _cached(self, embedding_model: str) -> Mapping[str, Vector]:
        return MappingProxyType(
            {text: vector for (model, text), vector in self._vectors.items() if model == embedding_model}
        )

    def _too_long_for(self, embedding_model: str, now: float) -> frozenset[str]:
        return frozenset(
            text
            for (model, text), seen_at in self._too_long.items()
            if model == embedding_model and now - seen_at < self._too_long_ttl_seconds
        )

    def _merged_too_long(
        self, embedding_model: str, embedded: _Embedded, texts: Sequence[str], now: float
    ) -> Mapping[_CacheKey, float]:
        fresh: Final = MappingProxyType(
            {key: seen_at for key, seen_at in self._too_long.items() if now - seen_at < self._too_long_ttl_seconds}
        )
        newly_too_long: Final = (
            (embedding_model, text)
            for text in texts
            if text in embedded.too_long and (embedding_model, text) not in fresh
        )
        ordered: Final = MappingProxyType({**fresh, **dict.fromkeys(newly_too_long, now)})
        return MappingProxyType(dict(islice(ordered.items(), max(len(ordered) - self._max_entries, 0), None)))

    def _merged(self, embedding_model: str, embedded: _Embedded, texts: Sequence[str]) -> Mapping[_CacheKey, Vector]:
        dimension: Final = len(embedded.query_vector)
        touched: Final = MappingProxyType(
            {(embedding_model, text): embedded.vectors[text] for text in texts if text in embedded.vectors}
        )
        untouched: Final = MappingProxyType(
            {
                key: vector
                for key, vector in chain(
                    self._vectors.items(),
                    (((embedding_model, text), vector) for text, vector in embedded.vectors.items()),
                )
                if key not in touched and (key[0] != embedding_model or len(vector) == dimension)
            }
        )
        ordered: Final = MappingProxyType({**untouched, **touched})
        return MappingProxyType(dict(islice(ordered.items(), max(len(ordered) - self._max_entries, 0), None)))

    async def scores(
        self, query: str, texts: Sequence[str], embed: Embedder, embedding_model: str
    ) -> tuple[float | None, ...] | EmbeddingFailed:
        """Cosine similarity for embeddable entries of `texts`, with None for omitted entries."""
        if not texts:
            return ()
        now: Final = self._clock()
        embedded: Final = await _embed_query_and_texts(
            embed, query, texts, self._cached(embedding_model), self._too_long_for(embedding_model, now)
        )
        if isinstance(embedded, EmbeddingFailed):
            return embedded
        same_dimension: Final = _same_dimension(embedded.query_vector, embedded.vectors, texts)
        if embedded.failure is not None:
            if same_dimension:
                self._vectors = self._merged(embedding_model, embedded, texts)
                self._too_long = self._merged_too_long(embedding_model, embedded, texts, now)
            return embedded.failure
        if not same_dimension:
            return EmbeddingFailed(reason=f"embedding model {embedding_model} returned vectors of mixed dimensions")
        self._vectors = self._merged(embedding_model, embedded, texts)
        self._too_long = self._merged_too_long(embedding_model, embedded, texts, now)
        return tuple(
            cosine_similarity(embedded.query_vector, embedded.vectors[text]) if text in embedded.vectors else None
            for text in texts
        )
