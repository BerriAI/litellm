"""Semantic ranking over the in-memory A2A agent registry, shared by GET /v1/agents?query= and the agent_search MCP tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.proxy.common_utils.semantic_text_index import (
    Embedder,
    EmbeddingFailed,
    SemanticTextIndex,
    router_embedder,
)
from litellm.types.agents import AgentResponse

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router

DEFAULT_AGENT_SEARCH_TOP_K: Final = 5


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


class AgentSearchIndex:
    """Caches one vector per distinct agent text per embedding model, so repeat searches only embed the query."""

    def __init__(self) -> None:
        self._index: Final = SemanticTextIndex()

    async def search(
        self, query: str, agents: Sequence[AgentResponse], top_k: int, embed: Embedder, embedding_model: str
    ) -> AgentSearchHits | AgentSearchEmbeddingFailed:
        texts: Final = tuple(agent_search_text(agent) for agent in agents)
        scores: Final = await self._index.scores(query, texts, embed, embedding_model)
        if isinstance(scores, EmbeddingFailed):
            return AgentSearchEmbeddingFailed(reason=scores.reason)
        ranked: Final = sorted(
            (AgentSearchHit(agent=agent, score=score) for agent, score in zip(agents, scores, strict=True)),
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
