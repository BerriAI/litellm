"""Semantic ranking over the LiteLLM-hosted skill registry, shared by GET /v1/skills?query= and the skill_search MCP tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import BaseModel, ConfigDict

from litellm.llms.litellm_proxy.skills.constants import MAX_SKILLS_PER_SEARCH
from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
from litellm.proxy.common_utils.semantic_text_index import (
    Embedder,
    EmbeddingFailed,
    SemanticTextIndex,
    router_embedder,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

DEFAULT_SKILL_SEARCH_TOP_K: Final = 5
MAX_SKILL_SEARCH_TOP_K: Final = 100
"""Matches the ``le=100`` bound GET /v1/skills?query= enforces via FastAPI's Query
validation, so the MCP tool can't return a larger payload than the REST endpoint allows."""
MAX_SKILL_SEARCH_TEXT_CHARS: Final = 4000
"""Per-skill cap on the title + description + instructions text that gets embedded, so one
search embeds at most ``MAX_SKILLS_PER_SEARCH * MAX_SKILL_SEARCH_TEXT_CHARS`` characters no
matter how long the stored instructions are."""


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


@dataclass(frozen=True, slots=True)
class SkillSearchUnsupportedProvider:
    reason: str


SkillSearchOutcome: TypeAlias = SkillSearchHits | SkillSearchNotConfigured | SkillSearchEmbeddingFailed
HostedSkillSearchOutcome: TypeAlias = SkillSearchOutcome | SkillSearchUnsupportedProvider


class SkillSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    display_title: str | None
    description: str | None
    score: float


def skill_search_text(skill: LiteLLM_SkillsTable) -> str:
    joined: Final = "\n".join(part for part in (skill.display_title, skill.description, skill.instructions) if part)
    return joined[:MAX_SKILL_SEARCH_TEXT_CHARS]


def skill_search_result(hit: SkillSearchHit) -> SkillSearchResult:
    return SkillSearchResult(
        skill_id=hit.skill.skill_id,
        display_title=hit.skill.display_title,
        description=hit.skill.description,
        score=hit.score,
    )


class SkillSearchIndex:
    """Caches one vector per distinct skill text per embedding model, so repeat searches only embed the query."""

    def __init__(self, max_entries: int = MAX_SKILLS_PER_SEARCH) -> None:
        self._index: Final = SemanticTextIndex(max_entries=max_entries)

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
    proxy_logging_obj: ProxyLogging,
) -> SkillSearchOutcome:
    if embedding_model is None:
        return SkillSearchNotConfigured(
            reason="skill search needs litellm_settings.skill_search_embedding_model set to an embedding model from model_list"
        )
    if router is None:
        return SkillSearchNotConfigured(reason="skill search needs a model_list so the embedding model can be called")
    embed: Final = router_embedder(router, embedding_model, user_api_key_dict, proxy_logging_obj)
    return await index.search(query, skills, top_k, embed, embedding_model)


async def search_hosted_skills(
    custom_llm_provider: str | None,
    query: str,
    top_k: int,
    router: Router | None,
    embedding_model: str | None,
    index: SkillSearchIndex,
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
) -> HostedSkillSearchOutcome:
    """GET /v1/skills?query= for the skills LiteLLM hosts itself: only ``litellm_proxy`` has a registry to rank."""
    if custom_llm_provider != LlmProviders.LITELLM_PROXY.value:
        return SkillSearchUnsupportedProvider(reason="query is only supported for custom_llm_provider=litellm_proxy")
    skills: Final = await LiteLLMSkillsHandler.list_skills_for_search(user_api_key_dict=user_api_key_dict)
    return await search_skills(
        query=query,
        skills=skills,
        top_k=top_k,
        router=router,
        embedding_model=embedding_model,
        index=index,
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
    )
