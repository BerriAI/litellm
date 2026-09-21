from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import APIConnectionError

import litellm
from litellm.llms.litellm_proxy.skills.skill_search import skill_search_text
from litellm.proxy._types import LiteLLM_SkillsTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.skills_endpoints import router
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

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
def key_limits(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    limits = MagicMock()
    limits.pre_call_hook = AsyncMock(side_effect=lambda user_api_key_dict, data, call_type: data)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", limits)
    return limits


@pytest.fixture
def embedding_router(monkeypatch: pytest.MonkeyPatch, key_limits: MagicMock) -> MagicMock:
    embedding_router = MagicMock()
    embedding_router.aembedding = AsyncMock(
        side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
            model=model,
            data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
        )
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", embedding_router)
    monkeypatch.setattr(litellm, "skill_search_embedding_model", "text-embedding-3-small")
    return embedding_router


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

    def test_a_key_over_its_rate_limit_gets_a_429_without_embedding(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock, key_limits: MagicMock
    ) -> None:
        key_limits.pre_call_hook = AsyncMock(side_effect=ProxyRateLimitError(detail="rpm exceeded"))
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "language translation"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 429
        embedding_router.aembedding.assert_not_awaited()

    def test_top_k_is_validated(self, accessible_skills: AsyncMock, embedding_router: MagicMock) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything", "top_k": 0},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 422
