from collections.abc import Sequence
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import APIConnectionError

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_search import (
    AgentSearchEmbeddingFailed,
    AgentSearchHits,
    AgentSearchIndex,
    AgentSearchNotConfigured,
    Vector,
    agent_search_text,
    cosine_similarity,
    search_agents,
)
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import RestrictedAgentAccess
from litellm.proxy.agent_endpoints.endpoints import router, user_api_key_auth
from litellm.types.agents import AgentResponse

CALLER: Final = UserAPIKeyAuth(api_key="hashed-caller-key", team_id="team-1", user_id="user-1")

TRANSLATOR: Final = AgentResponse(
    agent_id="translator",
    agent_name="document-translator",
    agent_card_params={
        "name": "Document Translator",
        "description": "Converts files from one language into another",
        "skills": [
            {
                "id": "t",
                "name": "Translate a file",
                "description": "Produce the document in the target language",
                "tags": ["localization", "documents"],
            }
        ],
    },
)
SQL_ANALYST: Final = AgentResponse(
    agent_id="sql",
    agent_name="warehouse-sql-analyst",
    agent_card_params={
        "name": "Warehouse SQL Analyst",
        "description": "Runs SQL against the inventory database",
        "skills": [],
    },
)
TRIP_PLANNER: Final = AgentResponse(
    agent_id="trip",
    agent_name="trip-planner",
    agent_card_params={"name": "Trip Planner", "description": "Books flights and hotels"},
)
AGENTS: Final = (TRANSLATOR, SQL_ANALYST, TRIP_PLANNER)

VECTORS: Final = MappingProxyType(
    {
        "language translation": (1.0, 0.0, 0.0),
        agent_search_text(TRANSLATOR): (0.9, 0.1, 0.0),
        agent_search_text(SQL_ANALYST): (0.0, 1.0, 0.0),
        agent_search_text(TRIP_PLANNER): (0.3, 0.0, 1.0),
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
        return tuple((1.0,) * self.dimensions for _ in texts)


class TestAgentSearchText:
    def test_joins_name_description_and_skills_with_tags(self) -> None:
        assert agent_search_text(TRANSLATOR) == (
            "document-translator\n"
            "Converts files from one language into another\n"
            "Translate a file Produce the document in the target language localization documents"
        )

    def test_missing_card_fields_fall_back_to_the_name(self) -> None:
        assert agent_search_text(AgentResponse(agent_id="x", agent_name="bare", agent_card_params={})) == "bare"

    def test_malformed_skills_do_not_break_the_text(self) -> None:
        agent = AgentResponse(
            agent_id="x", agent_name="odd", agent_card_params={"skills": "not-a-list", "description": "d"}
        )
        assert agent_search_text(agent) == "odd"


class TestCosineSimilarity:
    def test_identical_direction_scores_one(self) -> None:
        assert cosine_similarity((2.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_orthogonal_scores_zero(self) -> None:
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_instead_of_dividing(self) -> None:
        assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0


class TestAgentSearchIndex:
    @pytest.mark.asyncio
    async def test_ranks_by_similarity_and_truncates_to_top_k(self) -> None:
        outcome = await AgentSearchIndex().search(
            "language translation", AGENTS, top_k=2, embed=FakeEmbedder(), embedding_model="m"
        )
        assert isinstance(outcome, AgentSearchHits)
        assert [hit.agent.agent_id for hit in outcome.hits] == ["translator", "trip"]
        assert outcome.hits[0].score > outcome.hits[1].score

    @pytest.mark.asyncio
    async def test_second_search_only_embeds_the_query(self) -> None:
        index = AgentSearchIndex()
        embedder = FakeEmbedder()
        await index.search("language translation", AGENTS, top_k=5, embed=embedder, embedding_model="m")
        await index.search("language translation", AGENTS, top_k=5, embed=embedder, embedding_model="m")
        assert len(embedder.calls[0]) == 1 + len(AGENTS)
        assert embedder.calls[1] == ("language translation",)

    @pytest.mark.asyncio
    async def test_switching_embedding_models_does_not_reuse_cached_vectors(self) -> None:
        index = AgentSearchIndex()
        await index.search("language translation", AGENTS, top_k=5, embed=FakeEmbedder(), embedding_model="small")
        wide = FixedDimensionEmbedder(2)
        outcome = await index.search("language translation", AGENTS, top_k=5, embed=wide, embedding_model="wide")
        assert isinstance(outcome, AgentSearchHits)
        assert len(wide.calls[0]) == 1 + len(AGENTS)

    @pytest.mark.asyncio
    async def test_cached_vectors_of_another_dimension_are_re_embedded(self) -> None:
        index = AgentSearchIndex()
        await index.search("language translation", AGENTS, top_k=5, embed=FakeEmbedder(), embedding_model="m")
        fallback = FixedDimensionEmbedder(2)
        outcome = await index.search("language translation", AGENTS, top_k=5, embed=fallback, embedding_model="m")
        assert isinstance(outcome, AgentSearchHits)
        assert fallback.calls == [
            ("language translation",),
            ("language translation", *(agent_search_text(agent) for agent in AGENTS)),
        ]

    @pytest.mark.asyncio
    async def test_mixed_dimensions_in_one_batch_become_embedding_failed(self) -> None:
        async def mixed(texts: Sequence[str]) -> Sequence[Vector]:
            return ((1.0, 0.0), *((1.0, 0.0, 0.0) for _ in texts[1:]))

        outcome = await AgentSearchIndex().search("q", AGENTS, top_k=5, embed=mixed, embedding_model="m")
        assert isinstance(outcome, AgentSearchEmbeddingFailed)
        assert "mixed dimensions" in outcome.reason

    @pytest.mark.asyncio
    async def test_empty_registry_returns_no_hits_without_embedding(self) -> None:
        embedder = FakeEmbedder()
        outcome = await AgentSearchIndex().search("anything", (), top_k=5, embed=embedder, embedding_model="m")
        assert outcome == AgentSearchHits(hits=())
        assert embedder.calls == []

    @pytest.mark.asyncio
    async def test_provider_error_becomes_embedding_failed(self) -> None:
        async def failing(texts: Sequence[str]) -> Sequence[Vector]:
            raise APIConnectionError(request=MagicMock())

        outcome = await AgentSearchIndex().search("q", AGENTS, top_k=5, embed=failing, embedding_model="m")
        assert isinstance(outcome, AgentSearchEmbeddingFailed)
        assert "embedding the search query failed" in outcome.reason

    @pytest.mark.asyncio
    async def test_wrong_vector_count_becomes_embedding_failed(self) -> None:
        async def short(texts: Sequence[str]) -> Sequence[Vector]:
            return ((1.0, 0.0, 0.0),)

        outcome = await AgentSearchIndex().search("q", AGENTS, top_k=5, embed=short, embedding_model="m")
        assert isinstance(outcome, AgentSearchEmbeddingFailed)


class TestSearchAgents:
    @pytest.mark.asyncio
    async def test_no_embedding_model_is_not_configured(self) -> None:
        outcome = await search_agents(
            "q", AGENTS, 5, router=MagicMock(), embedding_model=None, index=AgentSearchIndex(), user_api_key_dict=CALLER
        )
        assert isinstance(outcome, AgentSearchNotConfigured)
        assert "agent_search_embedding_model" in outcome.reason

    @pytest.mark.asyncio
    async def test_no_router_is_not_configured(self) -> None:
        outcome = await search_agents(
            "q", AGENTS, 5, router=None, embedding_model="m", index=AgentSearchIndex(), user_api_key_dict=CALLER
        )
        assert isinstance(outcome, AgentSearchNotConfigured)

    @pytest.mark.asyncio
    async def test_router_embeddings_are_read_from_the_response(self) -> None:
        router = MagicMock()
        router.aembedding = AsyncMock(
            side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
                model=model,
                data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
            )
        )
        outcome = await search_agents(
            "language translation",
            AGENTS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=AgentSearchIndex(),
            user_api_key_dict=CALLER,
        )
        assert isinstance(outcome, AgentSearchHits)
        assert [hit.agent.agent_id for hit in outcome.hits] == ["translator"]
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
        await search_agents(
            "language translation",
            AGENTS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=AgentSearchIndex(),
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
def registry(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    from litellm.proxy.agent_endpoints import agent_registry as registry_module

    mock_registry = MagicMock()
    mock_registry.get_agent_list = MagicMock(return_value=AGENTS)
    mock_registry.ids_for_agent = MagicMock(side_effect=lambda agent_id: frozenset({agent_id}))
    monkeypatch.setattr(registry_module, "global_agent_registry", mock_registry)
    monkeypatch.setattr("litellm.proxy.agent_endpoints.endpoints.global_agent_search_index", AgentSearchIndex())
    return mock_registry


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
    monkeypatch.setattr(litellm, "agent_search_embedding_model", "text-embedding-3-small")
    return router


@pytest.fixture
def no_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)


class TestGetAgentsQuery:
    def test_query_ranks_and_scores_and_truncates(
        self, registry: MagicMock, embedding_router: MagicMock, no_db: None
    ) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/agents", params={"query": "language translation", "top_k": 2}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 200
        body = response.json()
        assert [agent["agent_id"] for agent in body] == ["translator", "trip"]
        assert body[0]["search_score"] > body[1]["search_score"]
        assert embedding_router.aembedding.await_args.kwargs["metadata"]["user_api_key_user_id"] == "u"

    def test_without_query_the_list_is_unchanged_and_unscored(
        self, registry: MagicMock, embedding_router: MagicMock, no_db: None
    ) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert response.status_code == 200
        assert [agent["agent_id"] for agent in response.json()] == ["translator", "sql", "trip"]
        assert all(agent["search_score"] is None for agent in response.json())
        embedding_router.aembedding.assert_not_awaited()

    def test_restricted_key_only_ranks_its_own_agents(
        self, registry: MagicMock, embedding_router: MagicMock, no_db: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.resolve_agent_access",
            AsyncMock(return_value=RestrictedAgentAccess(frozenset({"sql"}))),
        )
        response = _client(LitellmUserRoles.INTERNAL_USER).get(
            "/v1/agents", params={"query": "language translation"}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 200
        assert [agent["agent_id"] for agent in response.json()] == ["sql"]

    def test_missing_embedding_model_is_a_400(
        self, registry: MagicMock, embedding_router: MagicMock, no_db: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "agent_search_embedding_model", None)
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/agents", params={"query": "anything"}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "agent_search_not_configured"

    def test_embedding_provider_failure_is_a_503(
        self, registry: MagicMock, embedding_router: MagicMock, no_db: None
    ) -> None:
        embedding_router.aembedding = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/agents", params={"query": "anything"}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "agent_search_unavailable"

    def test_top_k_is_validated(self, registry: MagicMock, embedding_router: MagicMock, no_db: None) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/agents", params={"query": "anything", "top_k": 0}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 422
