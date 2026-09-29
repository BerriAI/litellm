from collections.abc import Iterator
from typing import Final

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.search_endpoints.endpoints import router

TAVILY_SEARCH_URL: Final = "https://api.tavily.com/search"
TAVILY_RESULT: Final = {"title": "LiteLLM", "url": "https://docs.litellm.ai", "content": "LLM gateway"}


def _search_router() -> Router:
    return Router(
        model_list=[],
        search_tools=[
            {"search_tool_name": "search-a", "litellm_params": {"search_provider": "tavily", "api_key": "fake"}},
        ],
        num_retries=0,
    )


def _client(caller: UserAPIKeyAuth) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def tavily(monkeypatch: pytest.MonkeyPatch) -> Iterator[respx.Route]:
    monkeypatch.setattr(  # test-quality-ok: respx needs HTTPX enabled to fake the provider HTTP boundary.
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=False) as mock:
        yield mock.post(TAVILY_SEARCH_URL).respond(200, json={"results": [TAVILY_RESULT]})
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> UserApiKeyCache:
    user_api_key_cache = UserApiKeyCache()
    user_api_key_cache.set_cache(key="user-1", value=LiteLLM_UserTable(user_id="user-1"))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", user_api_key_cache)
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "llm_router", _search_router())
    return user_api_key_cache


def _cache_team(cache: UserApiKeyCache, search_tools: list[str]) -> None:
    team = LiteLLM_TeamTableCachedObj(
        team_id="team-1",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-team", search_tools=search_tools),
    )
    cache.set_cache(key="team_id:team-1", value=team)


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
    monkeypatch, cache, tavily, path, general_settings, team_search_tools, expected_status
):
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
    _cache_team(cache, team_search_tools)
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user-1", team_id="team-1")

    response = _client(caller).post(path, json={"query": "what is litellm"})

    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        assert response.json()["results"][0]["url"] == TAVILY_RESULT["url"]
        assert tavily.call_count == 1
    else:
        assert "search-a" in response.text
        assert tavily.call_count == 0


def test_direct_search_body_tool_name_is_denied_under_default_search_list_deny(monkeypatch, cache, tavily):
    monkeypatch.setattr(proxy_server, "general_settings", {"default_search_list_deny": True})
    caller = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    response = _client(caller).post("/v1/search", json={"search_tool_name": "search-a", "query": "what is litellm"})

    assert response.status_code == 403, response.text
    assert tavily.call_count == 0
