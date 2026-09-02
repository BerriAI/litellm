import asyncio
import json
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import APIConnectionError

import litellm
from litellm.llms.litellm_proxy.skills.skill_search import (
    SkillSearchEmbeddingFailed,
    SkillSearchHits,
    SkillSearchIndex,
    SkillSearchNotConfigured,
    search_skills,
    skill_search_text,
)
from litellm.proxy._types import LiteLLM_SkillsTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.skills_endpoints import router
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.semantic_text_index import Vector, cosine_similarity

CALLER: Final = UserAPIKeyAuth(api_key="hashed-caller-key", team_id="team-1", user_id="user-1")

TRANSLATOR: Final = LiteLLM_SkillsTable(
    skill_id="translate-file",
    display_title="Document Translator",
    description="Converts files from one language into another",
    instructions="Take an uploaded document and produce it in the target language",
)
SQL_ANALYST: Final = LiteLLM_SkillsTable(
    skill_id="warehouse-sql-analyst",
    display_title="Warehouse SQL Analyst",
    description="Runs SQL against the inventory database",
)
TRIP_PLANNER: Final = LiteLLM_SkillsTable(
    skill_id="trip-planner",
    display_title="Trip Planner",
    description="Books flights and hotels",
)
SKILLS: Final = (TRANSLATOR, SQL_ANALYST, TRIP_PLANNER)

VECTORS: Final = MappingProxyType(
    {
        "language translation": (1.0, 0.0, 0.0),
        skill_search_text(TRANSLATOR): (0.9, 0.1, 0.0),
        skill_search_text(SQL_ANALYST): (0.0, 1.0, 0.0),
        skill_search_text(TRIP_PLANNER): (0.3, 0.0, 1.0),
    }
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        return tuple(VECTORS[text] for text in texts)


class FixedDimensionEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions: Final = dimensions
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        await asyncio.sleep(0)
        return tuple((1.0,) * self.dimensions for _ in texts)


class TestSkillSearchText:
    def test_joins_title_description_and_instructions(self) -> None:
        assert skill_search_text(TRANSLATOR) == (
            "Document Translator\n"
            "Converts files from one language into another\n"
            "Take an uploaded document and produce it in the target language"
        )

    def test_missing_fields_fall_back_to_whatever_is_present(self) -> None:
        assert skill_search_text(LiteLLM_SkillsTable(skill_id="bare", display_title="bare")) == "bare"

    def test_all_fields_absent_is_an_empty_string(self) -> None:
        assert skill_search_text(LiteLLM_SkillsTable(skill_id="empty")) == ""


class TestCosineSimilarity:
    def test_identical_direction_scores_one(self) -> None:
        assert cosine_similarity((2.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_orthogonal_scores_zero(self) -> None:
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_instead_of_dividing(self) -> None:
        assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0


class TestSkillSearchIndex:
    @pytest.mark.asyncio
    async def test_ranks_by_similarity_and_truncates_to_top_k(self) -> None:
        outcome = await SkillSearchIndex().search(
            "language translation", SKILLS, top_k=2, embed=FakeEmbedder(), embedding_model="m"
        )
        assert isinstance(outcome, SkillSearchHits)
        assert [hit.skill.skill_id for hit in outcome.hits] == ["translate-file", "trip-planner"]
        assert outcome.hits[0].score > outcome.hits[1].score

    @pytest.mark.asyncio
    async def test_second_search_only_embeds_the_query(self) -> None:
        index = SkillSearchIndex()
        embedder = FakeEmbedder()
        await index.search("language translation", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        await index.search("language translation", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        assert len(embedder.calls[0]) == 1 + len(SKILLS)
        assert embedder.calls[1] == ("language translation",)

    @pytest.mark.asyncio
    async def test_switching_embedding_models_does_not_reuse_cached_vectors(self) -> None:
        index = SkillSearchIndex()
        await index.search("language translation", SKILLS, top_k=5, embed=FakeEmbedder(), embedding_model="small")
        wide = FixedDimensionEmbedder(2)
        outcome = await index.search("language translation", SKILLS, top_k=5, embed=wide, embedding_model="wide")
        assert isinstance(outcome, SkillSearchHits)
        assert len(wide.calls[0]) == 1 + len(SKILLS)

    @pytest.mark.asyncio
    async def test_cached_vectors_of_another_dimension_are_re_embedded(self) -> None:
        index = SkillSearchIndex()
        await index.search("language translation", SKILLS, top_k=5, embed=FakeEmbedder(), embedding_model="m")
        fallback = FixedDimensionEmbedder(2)
        outcome = await index.search("language translation", SKILLS, top_k=5, embed=fallback, embedding_model="m")
        assert isinstance(outcome, SkillSearchHits)
        assert fallback.calls == [
            ("language translation",),
            ("language translation", *(skill_search_text(skill) for skill in SKILLS)),
        ]

    @pytest.mark.asyncio
    async def test_re_embedding_a_subset_drops_the_other_skills_old_vectors(self) -> None:
        index = SkillSearchIndex()
        await index.search("language translation", SKILLS, top_k=5, embed=FakeEmbedder(), embedding_model="m")
        wide = FixedDimensionEmbedder(2)
        await index.search("language translation", SKILLS[:1], top_k=5, embed=wide, embedding_model="m")
        await index.search("language translation", SKILLS, top_k=5, embed=wide, embedding_model="m")
        assert wide.calls[-1] == ("language translation", *(skill_search_text(skill) for skill in SKILLS[1:]))

    @pytest.mark.asyncio
    async def test_concurrent_searches_keep_each_others_vectors(self) -> None:
        index = SkillSearchIndex()
        embedder = FixedDimensionEmbedder(3)
        await asyncio.gather(
            index.search("q", SKILLS[:1], top_k=5, embed=embedder, embedding_model="m"),
            index.search("q", SKILLS[1:], top_k=5, embed=embedder, embedding_model="m"),
        )
        await index.search("q", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        assert embedder.calls[-1] == ("q",)

    @pytest.mark.asyncio
    async def test_mixed_dimensions_in_one_batch_become_embedding_failed(self) -> None:
        async def mixed(texts: Sequence[str]) -> Sequence[Vector]:
            return ((1.0, 0.0), *((1.0, 0.0, 0.0) for _ in texts[1:]))

        outcome = await SkillSearchIndex().search("q", SKILLS, top_k=5, embed=mixed, embedding_model="m")
        assert isinstance(outcome, SkillSearchEmbeddingFailed)
        assert "mixed dimensions" in outcome.reason

    @pytest.mark.asyncio
    async def test_no_accessible_skills_returns_no_hits_without_embedding(self) -> None:
        embedder = FakeEmbedder()
        outcome = await SkillSearchIndex().search("anything", (), top_k=5, embed=embedder, embedding_model="m")
        assert outcome == SkillSearchHits(hits=())
        assert embedder.calls == []

    @pytest.mark.asyncio
    async def test_provider_error_becomes_embedding_failed(self) -> None:
        async def failing(texts: Sequence[str]) -> Sequence[Vector]:
            raise APIConnectionError(request=MagicMock())

        outcome = await SkillSearchIndex().search("q", SKILLS, top_k=5, embed=failing, embedding_model="m")
        assert isinstance(outcome, SkillSearchEmbeddingFailed)
        assert "embedding the search query failed" in outcome.reason

    @pytest.mark.asyncio
    async def test_wrong_vector_count_becomes_embedding_failed(self) -> None:
        async def short(texts: Sequence[str]) -> Sequence[Vector]:
            return ((1.0, 0.0, 0.0),)

        outcome = await SkillSearchIndex().search("q", SKILLS, top_k=5, embed=short, embedding_model="m")
        assert isinstance(outcome, SkillSearchEmbeddingFailed)


class TestSearchSkills:
    @pytest.mark.asyncio
    async def test_no_embedding_model_is_not_configured(self) -> None:
        outcome = await search_skills(
            "q", SKILLS, 5, router=MagicMock(), embedding_model=None, index=SkillSearchIndex(), user_api_key_dict=CALLER
        )
        assert isinstance(outcome, SkillSearchNotConfigured)
        assert "skill_search_embedding_model" in outcome.reason

    @pytest.mark.asyncio
    async def test_no_router_is_not_configured(self) -> None:
        outcome = await search_skills(
            "q", SKILLS, 5, router=None, embedding_model="m", index=SkillSearchIndex(), user_api_key_dict=CALLER
        )
        assert isinstance(outcome, SkillSearchNotConfigured)

    @pytest.mark.asyncio
    async def test_router_embeddings_are_read_from_the_response(self) -> None:
        router = MagicMock()
        router.aembedding = AsyncMock(
            side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
                model=model,
                data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
            )
        )
        outcome = await search_skills(
            "language translation",
            SKILLS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
        )
        assert isinstance(outcome, SkillSearchHits)
        assert [hit.skill.skill_id for hit in outcome.hits] == ["translate-file"]
        assert router.aembedding.await_args.kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embedding_spend_is_attributed_to_the_calling_key(self) -> None:
        router = MagicMock()
        router.aembedding = AsyncMock(
            side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
                model=model,
                data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
            )
        )
        await search_skills(
            "language translation",
            SKILLS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
        )
        metadata = router.aembedding.await_args.kwargs["metadata"]
        assert metadata["user_api_key"] == "hashed-caller-key"
        assert metadata["user_api_key_team_id"] == "team-1"
        assert metadata["user_api_key_user_id"] == "user-1"


def _client(role: LitellmUserRoles) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="u", user_role=role)
    return TestClient(app)


@pytest.fixture
def accessible_skills(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    list_for_search = AsyncMock(return_value=list(SKILLS))
    monkeypatch.setattr(
        "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.list_skills_for_search", list_for_search
    )
    return list_for_search


@pytest.fixture
def embedding_router(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    router = MagicMock()
    router.aembedding = AsyncMock(
        side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
            model=model,
            data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
        )
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "skill_search_embedding_model", "text-embedding-3-small")
    return router


class TestGetSkillsQuery:
    def test_query_ranks_and_scores_and_truncates(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "language translation", "top_k": 2},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 200
        body = response.json()["data"]
        assert [skill["id"] for skill in body] == ["translate-file", "trip-planner"]
        assert body[0]["search_score"] > body[1]["search_score"]
        assert embedding_router.aembedding.await_args.kwargs["metadata"]["user_api_key_user_id"] == "u"

    def test_restricted_key_only_ranks_the_skills_it_can_access(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        accessible_skills.return_value = [SQL_ANALYST]
        response = _client(LitellmUserRoles.INTERNAL_USER).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "language translation"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 200
        assert [skill["id"] for skill in response.json()["data"]] == ["warehouse-sql-analyst"]

    def test_no_accessible_skills_is_a_no_match_empty_result(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        accessible_skills.return_value = []
        response = _client(LitellmUserRoles.INTERNAL_USER).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 200
        assert response.json()["data"] == []
        embedding_router.aembedding.assert_not_awaited()

    def test_query_is_unsupported_for_the_anthropic_passthrough_provider(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills", params={"query": "anything"}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "skill_search_unsupported_provider"
        accessible_skills.assert_not_awaited()

    def test_missing_embedding_model_is_a_400(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "skill_search_embedding_model", None)
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "skill_search_not_configured"

    def test_embedding_provider_failure_is_a_503(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        embedding_router.aembedding = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "skill_search_unavailable"

    def test_top_k_is_validated(self, accessible_skills: AsyncMock, embedding_router: MagicMock) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything", "top_k": 0},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 422


class TestHandleSkillSearchMCP:
    @pytest.mark.asyncio
    async def test_top_k_is_clamped_to_the_same_ceiling_as_rest(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        from litellm.llms.litellm_proxy.skills.skill_search import MAX_SKILL_SEARCH_TOP_K
        from litellm.proxy._experimental.mcp_server.tool_search import handle_skill_search

        many_skills: Final = tuple(
            LiteLLM_SkillsTable(
                skill_id=f"skill-{i}", display_title=TRIP_PLANNER.display_title, description=TRIP_PLANNER.description
            )
            for i in range(MAX_SKILL_SEARCH_TOP_K + 50)
        )
        accessible_skills.return_value = list(many_skills)

        result = await handle_skill_search(
            query="language translation", top_k=10_000, user_api_key_dict=UserAPIKeyAuth(user_id="u")
        )
        assert result.isError is False
        assert len(json.loads(result.content[0].text)) == MAX_SKILL_SEARCH_TOP_K

    @pytest.mark.asyncio
    async def test_top_k_below_one_is_raised_to_one(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        from litellm.proxy._experimental.mcp_server.tool_search import handle_skill_search

        result = await handle_skill_search(
            query="language translation", top_k=0, user_api_key_dict=UserAPIKeyAuth(user_id="u")
        )
        assert result.isError is False
        assert len(json.loads(result.content[0].text)) == 1
