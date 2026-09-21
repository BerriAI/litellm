"""Embedding-similarity ranking over short texts with a per-model vector cache, shared by agent search and MCP tool search."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from fastapi import HTTPException
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict

from litellm.exceptions import BudgetExceededError

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

Vector: TypeAlias = tuple[float, ...]

DEFAULT_MAX_CACHED_VECTORS: Final = 5000
"""Ceiling on how many (embedding model, text) vectors one index keeps; the least recently searched are evicted first."""


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> Awaitable[Sequence[Vector]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingFailed:
    reason: str


class _EmbeddingItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    embedding: tuple[float, ...]


class _EmbeddingData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_EmbeddingItem, ...]


class _EmbeddingRequest(BaseModel):
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

    return {  # mutable-ok: the router mutates the metadata dict it is handed
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(user_api_key_dict),
        "user_api_key": user_api_key_dict.api_key,
    }


def router_embedder(
    router: Router, embedding_model: str, user_api_key_dict: UserAPIKeyAuth, proxy_logging_obj: ProxyLogging
) -> Embedder:
    """Embeds through the router after the same key rate-limit, budget and guardrail pre-call hooks /embeddings runs."""

    async def embed(texts: Sequence[str]) -> Sequence[Vector]:
        request: Final = {  # mutable-ok: pre_call_hook mutates the request dict in place
            "model": embedding_model,
            "input": list(texts),  # mutable-ok: Router.aembedding accepts only str | list input
            "metadata": embedding_spend_metadata(user_api_key_dict),
        }
        processed: Final = _EmbeddingRequest.model_validate(
            await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict, data=request, call_type="aembedding"
            )
        )
        response: Final = await router.aembedding(
            model=processed.model,
            input=list(processed.input),  # mutable-ok: Router.aembedding accepts only str | list input
            metadata=processed.metadata,
        )
        return tuple(item.embedding for item in _EmbeddingData.model_validate(response.model_dump()).data)

    return embed


_CacheKey: TypeAlias = tuple[str, str]


async def _embed_all(embed: Embedder, texts: Sequence[str]) -> tuple[Vector, ...] | EmbeddingFailed:
    try:
        vectors: Final = tuple(await embed(texts))
    except HTTPException:
        raise
    except (OpenAIError, ValueError, BudgetExceededError) as exc:
        return EmbeddingFailed(reason=f"embedding the search query failed: {exc}")
    if len(vectors) != len(texts):
        return EmbeddingFailed(reason=f"embedding model returned {len(vectors)} vectors for {len(texts)} inputs")
    return vectors


@dataclass(frozen=True, slots=True)
class _Embedded:
    query_vector: Vector
    vectors: Mapping[str, Vector]


def _same_dimension(query_vector: Vector, vectors: Mapping[str, Vector], texts: Sequence[str]) -> bool:
    return all(len(vectors[text]) == len(query_vector) for text in texts)


async def _embed_query_and_texts(
    embed: Embedder, query: str, texts: Sequence[str], cached: Mapping[str, Vector]
) -> _Embedded | EmbeddingFailed:
    missing: Final = tuple(dict.fromkeys(text for text in texts if text not in cached))
    embedded: Final = await _embed_all(embed, (query, *missing))
    if isinstance(embedded, EmbeddingFailed):
        return embedded
    vectors: Final = MappingProxyType(dict(chain(cached.items(), zip(missing, embedded[1:], strict=True))))
    if _same_dimension(embedded[0], vectors, texts):
        return _Embedded(query_vector=embedded[0], vectors=vectors)
    unique: Final = tuple(dict.fromkeys(texts))
    reembedded: Final = await _embed_all(embed, (query, *unique))
    if isinstance(reembedded, EmbeddingFailed):
        return reembedded
    return _Embedded(
        query_vector=reembedded[0], vectors=MappingProxyType(dict(zip(unique, reembedded[1:], strict=True)))
    )


class SemanticTextIndex:
    """Caches one vector per distinct text per embedding model, so repeat searches only embed the query.

    Holds at most ``max_entries`` vectors across all models: once full, the texts no recent search touched go first."""

    def __init__(self, max_entries: int = DEFAULT_MAX_CACHED_VECTORS) -> None:
        self._max_entries: Final = max_entries
        self._vectors: Mapping[_CacheKey, Vector] = MappingProxyType({})

    def _cached(self, embedding_model: str) -> Mapping[str, Vector]:
        return MappingProxyType(
            {text: vector for (model, text), vector in self._vectors.items() if model == embedding_model}
        )

    def _merged(self, embedding_model: str, embedded: _Embedded, texts: Sequence[str]) -> Mapping[_CacheKey, Vector]:
        dimension: Final = len(embedded.query_vector)
        touched: Final = MappingProxyType({(embedding_model, text): embedded.vectors[text] for text in texts})
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
    ) -> tuple[float, ...] | EmbeddingFailed:
        """Cosine similarity of `query` to each entry of `texts`, in the same order."""
        if not texts:
            return ()
        embedded: Final = await _embed_query_and_texts(embed, query, texts, self._cached(embedding_model))
        if isinstance(embedded, EmbeddingFailed):
            return embedded
        if not _same_dimension(embedded.query_vector, embedded.vectors, texts):
            return EmbeddingFailed(reason=f"embedding model {embedding_model} returned vectors of mixed dimensions")
        self._vectors = self._merged(embedding_model, embedded, texts)
        return tuple(cosine_similarity(embedded.query_vector, embedded.vectors[text]) for text in texts)
