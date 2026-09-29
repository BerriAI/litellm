from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as proxy_server
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.search_endpoints.endpoints import router


def _search_router() -> MagicMock:
    llm_router = MagicMock()
    llm_router.search_tools = [
        {"search_tool_name": "search-a", "litellm_params": {"search_provider": "tavily", "api_key": "fake"}},
    ]
    llm_router.asearch = AsyncMock(return_value=SearchResponse(object="search", results=[]))
    return llm_router


def _client(caller: UserAPIKeyAuth) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def user_without_search_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = UserApiKeyCache()
    cache.set_cache(key="user-1", value=LiteLLM_UserTable(user_id="user-1"))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())


@pytest.fixture
def team_with_search_tools(monkeypatch: pytest.MonkeyPatch, user_without_search_grants: None):
    def set_team_search_tools(search_tools: list[str]) -> None:
        team = LiteLLM_TeamTable(
            team_id="team-1",
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-team", search_tools=search_tools),
        )
        monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", AsyncMock(return_value=team))

    return set_team_search_tools


@pytest.mark.parametrize(
    "general_settings, team_search_tools, expected_status",
    [
        ({}, [], 200),
        ({"default_search_list_deny": False}, [], 200),
        ({"default_search_list_deny": True}, [], 403),
        ({"default_search_list_deny": True}, ["search-b"], 403),
        ({"default_search_list_deny": True}, ["search-a"], 200),
    ],
)
@pytest.mark.parametrize("path", ["/v1/search/search-a", "/search/search-a"])
def test_direct_search_team_key_follows_default_search_list_deny(
    monkeypatch, team_with_search_tools, path, general_settings, team_search_tools, expected_status
):
    llm_router = _search_router()
    monkeypatch.setattr(proxy_server, "llm_router", llm_router)
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
    team_with_search_tools(team_search_tools)
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user-1", team_id="team-1")

    response = _client(caller).post(path, json={"query": "what is litellm"})

    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        assert response.json()["object"] == "search"
        llm_router.asearch.assert_awaited_once()
    else:
        assert "search-a" in response.text
        llm_router.asearch.assert_not_awaited()


def test_direct_search_body_tool_name_is_denied_under_default_search_list_deny(monkeypatch):
    llm_router = _search_router()
    monkeypatch.setattr(proxy_server, "llm_router", llm_router)
    monkeypatch.setattr(proxy_server, "general_settings", {"default_search_list_deny": True})
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    response = _client(caller).post("/v1/search", json={"search_tool_name": "search-a", "query": "what is litellm"})

    assert response.status_code == 403, response.text
    llm_router.asearch.assert_not_awaited()
