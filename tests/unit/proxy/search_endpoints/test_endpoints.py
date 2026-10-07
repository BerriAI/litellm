from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import orjson
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
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.route_llm_request import ProxyMissingRequiredParamError
from litellm.proxy.search_endpoints.endpoints import router, search

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
    app: Final = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    return TestClient(app, raise_server_exceptions=False)


def _team_key(key_search_tools: list[str]) -> UserAPIKeyAuth:
    caller: Final = UserAPIKeyAuth(
        api_key="sk-team-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="user-1",
        team_id="team-1",
        object_permission_id="op-key",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-key", search_tools=key_search_tools),
    )
    caller.via_virtual_key = True
    return caller


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
    user_api_key_cache: Final = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", user_api_key_cache)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "llm_router", _search_router())
    return user_api_key_cache


def _cache_team(cache: UserApiKeyCache, search_tools: list[str]) -> None:
    team: Final = LiteLLM_TeamTableCachedObj(
        team_id="team-1",
        object_permission_id="op-team",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="op-team", search_tools=search_tools),
    )
    cache.set_cache(key="team_id:team-1", value=team)


@pytest.mark.parametrize(
    "general_settings, key_search_tools, team_search_tools, expected_status",
    [
        ({}, [], [], 200),
        ({"search_tool_deny_by_default": False}, [], [], 200),
        ({"search_tool_deny_by_default": True}, ["search-a"], [], 403),
        ({"search_tool_deny_by_default": True}, [], ["search-a"], 403),
        ({"search_tool_deny_by_default": True}, ["search-a"], ["search-b"], 403),
        ({"search_tool_deny_by_default": True}, ["search-a"], ["search-a"], 200),
    ],
)
@pytest.mark.parametrize("path", ["/v1/search/search-a", "/search/search-a"])
def test_direct_search_team_key_follows_search_tool_deny_by_default(
    monkeypatch: pytest.MonkeyPatch,
    cache: UserApiKeyCache,
    tavily: respx.Route,
    path: str,
    general_settings: dict[str, bool],
    key_search_tools: list[str],
    team_search_tools: list[str],
    expected_status: int,
):
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
    _cache_team(cache, team_search_tools)

    response: Final = _client(_team_key(key_search_tools)).post(path, json={"query": "what is litellm"})

    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        assert response.json()["results"][0]["url"] == TAVILY_RESULT["url"]
        assert tavily.call_count == 1
    else:
        assert "search-a" in response.text
        assert tavily.call_count == 0


def test_direct_search_body_tool_name_is_denied_under_search_tool_deny_by_default(monkeypatch, cache, tavily):
    monkeypatch.setattr(proxy_server, "general_settings", {"search_tool_deny_by_default": True})
    caller: Final = UserAPIKeyAuth(api_key="sk-standalone", user_role=LitellmUserRoles.INTERNAL_USER)
    caller.via_virtual_key = True

    response: Final = _client(caller).post(
        "/v1/search", json={"search_tool_name": "search-a", "query": "what is litellm"}
    )

    assert response.status_code == 403, response.text
    assert response.json()["error"]["type"] == "key_search_tool_access_denied"
    assert tavily.call_count == 0


def _json_request(body: dict[str, object]) -> MagicMock:
    request = MagicMock()
    request.body = AsyncMock(return_value=orjson.dumps(body))
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"query": "litellm"}, {"query": "litellm", "search_tool_name": ""}])
async def test_search_without_search_tool_name_or_model_is_a_400(body):
    with pytest.raises(ProxyMissingRequiredParamError) as exc_info:
        await search(
            request=_json_request(body),
            fastapi_response=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )

    assert exc_info.value.code == "400"
    assert exc_info.value.param == "search_tool_name"
    assert exc_info.value.message == "/search: Missing required parameter: 'search_tool_name'."


@pytest.mark.asyncio
@pytest.mark.parametrize("default_source", ["cli_model", "completion_model"])
async def test_search_with_only_a_query_falls_back_to_the_proxy_default_model(monkeypatch, default_source):
    if default_source == "cli_model":
        monkeypatch.setattr(proxy_server, "user_model", "perplexity-search")
    else:
        monkeypatch.setitem(proxy_server.general_settings, "completion_model", "perplexity-search")
    search_result = {"object": "search", "results": []}
    router = MagicMock()
    router.asearch = AsyncMock(return_value=search_result)
    monkeypatch.setattr(proxy_server, "llm_router", router)

    response = await search(
        request=_json_request({"query": "litellm"}),
        fastapi_response=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )

    assert response == search_result, response
    router.asearch.assert_awaited_once()
    assert router.asearch.await_args.kwargs["query"] == "litellm"
    assert router.asearch.await_args.kwargs["model"] == "perplexity-search"


def _standalone_key(key_search_tools: list[str] | None) -> UserAPIKeyAuth:
    caller: Final = UserAPIKeyAuth(
        api_key="sk-standalone",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission_id=None if key_search_tools is None else "op-key",
        object_permission=(
            None
            if key_search_tools is None
            else LiteLLM_ObjectPermissionTable(object_permission_id="op-key", search_tools=key_search_tools)
        ),
    )
    caller.via_virtual_key = True
    return caller


@pytest.mark.parametrize(
    "general_settings, body",
    [
        ({"search_tool_deny_by_default": True}, {"model": "search-a", "query": "what is litellm"}),
        (
            {"search_tool_deny_by_default": True, "completion_model": "search-a"},
            {"query": "what is litellm"},
        ),
    ],
    ids=["body-model", "completion-model"],
)
@pytest.mark.parametrize("key_search_tools, expected_status", [(None, 403), (["search-a"], 200)])
def test_direct_search_authorizes_the_tool_resolved_from_model_settings(
    monkeypatch, cache, tavily, general_settings, body, key_search_tools, expected_status
):
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)

    response: Final = _client(_standalone_key(key_search_tools)).post("/v1/search", json=body)

    assert response.status_code == expected_status, response.text
    assert tavily.call_count == (1 if expected_status == 200 else 0)


@pytest.mark.parametrize(
    "key_search_tools, expected_status, fallback_calls",
    [(["search-a"], 500, 0), (["search-a", "search-b"], 200, 1)],
)
def test_router_search_fallback_target_must_be_granted(
    monkeypatch, cache, key_search_tools, expected_status, fallback_calls
):
    from litellm.proxy.auth.fallback_model_access import router_fallback_access_check

    monkeypatch.setattr(proxy_server, "general_settings", {"search_tool_deny_by_default": True})
    monkeypatch.setattr(  # test-quality-ok: respx needs HTTPX enabled to fake the provider HTTP boundary.
        litellm,
        "disable_aiohttp_transport",
        True,
    )
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[],
            search_tools=[
                {"search_tool_name": "search-a", "litellm_params": {"search_provider": "exa_ai", "api_key": "fake"}},
                {"search_tool_name": "search-b", "litellm_params": {"search_provider": "tavily", "api_key": "fake"}},
            ],
            fallbacks=[{"search-a": ["search-b"]}],
            fallback_access_check=router_fallback_access_check,
            num_retries=0,
        ),
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=False) as mock:
        failing_tool: Final = mock.post(url__regex=r"https://api\.exa\.ai/.*").respond(500, json={"error": "down"})
        fallback_tool: Final = mock.post(TAVILY_SEARCH_URL).respond(200, json={"results": [TAVILY_RESULT]})
        response: Final = _client(_standalone_key(key_search_tools)).post(
            "/v1/search/search-a", json={"query": "what is litellm"}
        )
    litellm.in_memory_llm_clients_cache.flush_cache()

    assert response.status_code == expected_status, response.text
    assert failing_tool.call_count == 1
    assert fallback_tool.call_count == fallback_calls


@pytest.mark.parametrize(
    "requested_tool, completion_model, expected_status",
    [("search-b", "search-a", 403), ("search-a", "search-b", 200)],
    ids=["requested-tool-ungranted", "completion-model-ungranted"],
)
def test_direct_search_authorizes_the_requested_tool_over_completion_model(
    monkeypatch, cache, tavily, requested_tool, completion_model, expected_status
):
    monkeypatch.setattr(
        proxy_server, "general_settings", {"search_tool_deny_by_default": True, "completion_model": completion_model}
    )

    response: Final = _client(_standalone_key(["search-a"])).post(
        f"/v1/search/{requested_tool}", json={"query": "what is litellm"}
    )

    assert response.status_code == expected_status, response.text
    assert tavily.call_count == (1 if expected_status == 200 else 0)
