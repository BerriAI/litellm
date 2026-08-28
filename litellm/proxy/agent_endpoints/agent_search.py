"""Semantic ranking over the in-memory A2A agent registry, shared by GET /v1/agents?query= and the agent_search MCP tool."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.exceptions import BudgetExceededError
from litellm.types.agents import AgentResponse

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router

DEFAULT_AGENT_SEARCH_TOP_K: Final = 5

Vector: TypeAlias = tuple[float, ...]


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> Awaitable[Sequence[Vector]]: ...


@dataclass(frozen=True, slots=True)
class AgentSearchHit:
    agent: AgentResponse
    score: float


@dataclass(frozen=True, slots=True)
class AgentSearchHits:
    hits: tuple[AgentSearchHit, ...]


@dataclass(frozen=True, slots=True)
class AgentSearchNotConfigured:
    reason: str


@dataclass(frozen=True, slots=True)
class AgentSearchEmbeddingFailed:
    reason: str


AgentSearchOutcome: TypeAlias = AgentSearchHits | AgentSearchNotConfigured | AgentSearchEmbeddingFailed


class _SearchableSkill(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()


class _SearchableCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    description: str = ""
    skills: tuple[_SearchableSkill, ...] = ()


class _EmbeddingItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    embedding: tuple[float, ...]


class _EmbeddingData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_EmbeddingItem, ...]


class AgentSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_name: str
    description: str
    skills: tuple[_SearchableSkill, ...]
    score: float


def _searchable_card(agent: AgentResponse) -> _SearchableCard:
    try:
        return _SearchableCard.model_validate(agent.agent_card_params)
    except ValidationError:
        return _SearchableCard()


def _skill_text(skill: _SearchableSkill) -> str:
    return " ".join(part for part in (skill.name, skill.description, " ".join(skill.tags)) if part)


def agent_search_text(agent: AgentResponse) -> str:
    card: Final = _searchable_card(agent)
    skill_lines: Final = tuple(_skill_text(skill) for skill in card.skills)
    return "\n".join(part for part in (agent.agent_name, card.description, *skill_lines) if part)


def agent_search_result(hit: AgentSearchHit) -> AgentSearchResult:
    card: Final = _searchable_card(hit.agent)
    return AgentSearchResult(
        agent_id=hit.agent.agent_id,
        agent_name=hit.agent.agent_name,
        description=card.description,
        skills=card.skills,
        score=hit.score,
    )


def cosine_similarity(left: Vector, right: Vector) -> float:
    dot: Final = sum(a * b for a, b in zip(left, right, strict=True))
    norms: Final = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def embedding_spend_metadata(user_api_key_dict: UserAPIKeyAuth) -> dict[str, object]:
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    return {  # mutable-ok: the router mutates the metadata dict it is handed
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(user_api_key_dict),
        "user_api_key": user_api_key_dict.api_key,
    }


def router_embedder(router: Router, embedding_model: str, user_api_key_dict: UserAPIKeyAuth) -> Embedder:
    async def embed(texts: Sequence[str]) -> Sequence[Vector]:
        batch: Final = list(texts)  # mutable-ok: Router.aembedding accepts only str | list input
        response: Final = await router.aembedding(
            model=embedding_model, input=batch, metadata=embedding_spend_metadata(user_api_key_dict)
        )
        return tuple(item.embedding for item in _EmbeddingData.model_validate(response.model_dump()).data)

    return embed


_NO_VECTORS: Final[Mapping[str, Vector]] = MappingProxyType({})


async def _embed_all(embed: Embedder, texts: Sequence[str]) -> tuple[Vector, ...] | AgentSearchEmbeddingFailed:
    try:
        vectors: Final = tuple(await embed(texts))
    except (OpenAIError, ValueError, BudgetExceededError) as exc:
        return AgentSearchEmbeddingFailed(reason=f"embedding the search query failed: {exc}")
    if len(vectors) != len(texts):
        return AgentSearchEmbeddingFailed(
            reason=f"embedding model returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    return vectors


@dataclass(frozen=True, slots=True)
class _Embedded:
    query_vector: Vector
    vectors: Mapping[str, Vector]


def _same_dimension(query_vector: Vector, vectors: Mapping[str, Vector], texts: Sequence[str]) -> bool:
    return all(len(vectors[text]) == len(query_vector) for text in texts)


async def _embed_query_and_agents(
    embed: Embedder, query: str, texts: Sequence[str], cached: Mapping[str, Vector]
) -> _Embedded | AgentSearchEmbeddingFailed:
    missing: Final = tuple(dict.fromkeys(text for text in texts if text not in cached))
    embedded: Final = await _embed_all(embed, (query, *missing))
    if isinstance(embedded, AgentSearchEmbeddingFailed):
        return embedded
    vectors: Final = MappingProxyType(dict(chain(cached.items(), zip(missing, embedded[1:], strict=True))))
    if _same_dimension(embedded[0], vectors, texts):
        return _Embedded(query_vector=embedded[0], vectors=vectors)
    unique: Final = tuple(dict.fromkeys(texts))
    reembedded: Final = await _embed_all(embed, (query, *unique))
    if isinstance(reembedded, AgentSearchEmbeddingFailed):
        return reembedded
    return _Embedded(
        query_vector=reembedded[0], vectors=MappingProxyType(dict(zip(unique, reembedded[1:], strict=True)))
    )


class AgentSearchIndex:
    """Caches one vector per distinct agent text per embedding model, so repeat searches only embed the query."""

    def __init__(self) -> None:
        self._vectors: Mapping[str, Mapping[str, Vector]] = MappingProxyType({})

    async def search(
        self, query: str, agents: Sequence[AgentResponse], top_k: int, embed: Embedder, embedding_model: str
    ) -> AgentSearchHits | AgentSearchEmbeddingFailed:
        if not agents:
            return AgentSearchHits(hits=())
        texts: Final = tuple(agent_search_text(agent) for agent in agents)
        cached: Final = self._vectors.get(embedding_model, _NO_VECTORS)
        embedded: Final = await _embed_query_and_agents(embed, query, texts, cached)
        if isinstance(embedded, AgentSearchEmbeddingFailed):
            return embedded
        if not _same_dimension(embedded.query_vector, embedded.vectors, texts):
            return AgentSearchEmbeddingFailed(
                reason=f"embedding model {embedding_model} returned vectors of mixed dimensions"
            )
        self._vectors = MappingProxyType(
            {**self._vectors, embedding_model: MappingProxyType({**cached, **embedded.vectors})}
        )
        ranked: Final = sorted(
            (
                AgentSearchHit(agent=agent, score=cosine_similarity(embedded.query_vector, embedded.vectors[text]))
                for agent, text in zip(agents, texts, strict=True)
            ),
            key=lambda hit: hit.score,
            reverse=True,
        )
        return AgentSearchHits(hits=tuple(ranked[:top_k]))


global_agent_search_index: Final = AgentSearchIndex()


async def search_agents(
    query: str,
    agents: Sequence[AgentResponse],
    top_k: int,
    router: Router | None,
    embedding_model: str | None,
    index: AgentSearchIndex,
    user_api_key_dict: UserAPIKeyAuth,
) -> AgentSearchOutcome:
    if embedding_model is None:
        return AgentSearchNotConfigured(
            reason="agent search needs litellm_settings.agent_search_embedding_model set to an embedding model from model_list"
        )
    if router is None:
        return AgentSearchNotConfigured(reason="agent search needs a model_list so the embedding model can be called")
    return await index.search(
        query, agents, top_k, router_embedder(router, embedding_model, user_api_key_dict), embedding_model
    )
