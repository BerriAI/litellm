"""Semantic ranking over the LiteLLM-hosted skill registry, shared by GET /v1/skills?query= and the skill_search MCP tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import BaseModel, ConfigDict

from litellm.proxy.common_utils.semantic_text_index import (
    Embedder,
    EmbeddingFailed,
    SemanticTextIndex,
    router_embedder,
)

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
    from litellm.router import Router

DEFAULT_SKILL_SEARCH_TOP_K: Final = 5
MAX_SKILL_SEARCH_TOP_K: Final = 100
"""Matches the ``le=100`` bound GET /v1/skills?query= enforces via FastAPI's Query
validation, so the MCP tool can't return a larger payload than the REST endpoint allows."""


@dataclass(frozen=True, slots=True)
class SkillSearchHit:
    skill: LiteLLM_SkillsTable
    score: float


@dataclass(frozen=True, slots=True)
class SkillSearchHits:
    hits: tuple[SkillSearchHit, ...]


@dataclass(frozen=True, slots=True)
class SkillSearchNotConfigured:
    reason: str


@dataclass(frozen=True, slots=True)
class SkillSearchEmbeddingFailed:
    reason: str


SkillSearchOutcome: TypeAlias = SkillSearchHits | SkillSearchNotConfigured | SkillSearchEmbeddingFailed


class SkillSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    display_title: str | None
    description: str | None
    score: float


def skill_search_text(skill: LiteLLM_SkillsTable) -> str:
    return "\n".join(part for part in (skill.display_title, skill.description, skill.instructions) if part)


def skill_search_result(hit: SkillSearchHit) -> SkillSearchResult:
    return SkillSearchResult(
        skill_id=hit.skill.skill_id,
        display_title=hit.skill.display_title,
        description=hit.skill.description,
        score=hit.score,
    )


class SkillSearchIndex:
    """Caches one vector per distinct skill text per embedding model, so repeat searches only embed the query."""

    def __init__(self) -> None:
        self._index: Final = SemanticTextIndex()

    async def search(
        self,
        query: str,
        skills: Sequence[LiteLLM_SkillsTable],
        top_k: int,
        embed: Embedder,
        embedding_model: str,
    ) -> SkillSearchHits | SkillSearchEmbeddingFailed:
        texts: Final = tuple(skill_search_text(skill) for skill in skills)
        scores: Final = await self._index.scores(query, texts, embed, embedding_model)
        if isinstance(scores, EmbeddingFailed):
            return SkillSearchEmbeddingFailed(reason=scores.reason)
        ranked: Final = sorted(
            (SkillSearchHit(skill=skill, score=score) for skill, score in zip(skills, scores, strict=True)),
            key=lambda hit: hit.score,
            reverse=True,
        )
        return SkillSearchHits(hits=tuple(ranked[:top_k]))


global_skill_search_index: Final = SkillSearchIndex()


async def search_skills(
    query: str,
    skills: Sequence[LiteLLM_SkillsTable],
    top_k: int,
    router: Router | None,
    embedding_model: str | None,
    index: SkillSearchIndex,
    user_api_key_dict: UserAPIKeyAuth,
) -> SkillSearchOutcome:
    if embedding_model is None:
        return SkillSearchNotConfigured(
            reason="skill search needs litellm_settings.skill_search_embedding_model set to an embedding model from model_list"
        )
    if router is None:
        return SkillSearchNotConfigured(reason="skill search needs a model_list so the embedding model can be called")
    return await index.search(
        query, skills, top_k, router_embedder(router, embedding_model, user_api_key_dict), embedding_model
    )
