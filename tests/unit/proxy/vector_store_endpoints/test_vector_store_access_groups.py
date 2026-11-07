"""Group grants open the read gate and the list, and never the manage gate."""

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.proxy._types import UI_TEAM_ID, LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.proxy_server import app
from litellm.proxy.vector_store_endpoints.utils import (
    can_user_read_vector_store,
    filter_listable_vector_stores,
)
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry
from tests.unit.proxy.vector_store_access_group_state import (
    TEAM_ID,
    TOKEN,
    ProxyState,
    make_key,
    make_team,
    team_owned_store,
)


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> ProxyState:
    return ProxyState(monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_id", "allowed"),
    [pytest.param("store-a", True, id="group-store"), pytest.param("store-b", False, id="other-store")],
)
async def test_search_and_files_gate_counts_team_group_stores(proxy: ProxyState, store_id: str, allowed: bool) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))

    assert await can_user_read_vector_store(team_owned_store(store_id), make_key(team_id=TEAM_ID)) is allowed


@pytest.mark.asyncio
async def test_search_and_files_gate_counts_a_personal_keys_group_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))

    assert await can_user_read_vector_store(team_owned_store("store-a"), make_key(groups=("g1",))) is True
    assert await can_user_read_vector_store(team_owned_store("store-b"), make_key(groups=("g1",))) is False


@pytest.mark.asyncio
async def test_models_only_group_grants_nothing_on_search_and_list(proxy: ProxyState) -> None:
    await proxy.add_group("models-only", models=("gpt-4",))
    await proxy.add_team(make_team(groups=("models-only",)))

    assert await can_user_read_vector_store(team_owned_store("store-a"), make_key(team_id=TEAM_ID)) is False
    assert await filter_listable_vector_stores((team_owned_store("store-a"),), make_key(team_id=TEAM_ID)) == ()


@pytest.mark.asyncio
async def test_direct_grant_never_reads_teams_or_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    team_reads: Final = AsyncMock(side_effect=RuntimeError("database unavailable"))
    proxy: Final = ProxyState(monkeypatch, team_reads=team_reads)

    assert (
        await can_user_read_vector_store(team_owned_store("store-a"), make_key(("store-a",), team_id=TEAM_ID)) is True
    )
    assert team_reads.await_count == 0
    assert proxy.group_reads.await_count == 0


@pytest.mark.asyncio
async def test_list_shows_only_granted_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))

    listed: Final = await filter_listable_vector_stores(
        (team_owned_store("store-a"), team_owned_store("store-b")), make_key(team_id=TEAM_ID)
    )

    assert listed == (team_owned_store("store-a"),)


@pytest.mark.asyncio
async def test_list_reads_the_group_once_however_many_stores_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = MagicMock(wraps=UserApiKeyCache())
    proxy: Final = ProxyState(monkeypatch, cache=cache)
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))

    await filter_listable_vector_stores(
        (team_owned_store("store-a"), team_owned_store("store-b"), team_owned_store("store-c")),
        make_key(team_id=TEAM_ID),
    )

    group_lookups: Final = [
        call for call in cache.async_get_cache.call_args_list if call.kwargs["key"] == "access_group_id:g1"
    ]
    assert len(group_lookups) == 1


@pytest.mark.asyncio
async def test_unreadable_group_grants_nothing_on_search_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    await proxy.add_team(make_team(groups=("g1",)))

    assert await can_user_read_vector_store(team_owned_store("store-a"), make_key(team_id=TEAM_ID)) is False
    assert await filter_listable_vector_stores((team_owned_store("store-a"),), make_key(team_id=TEAM_ID)) == ()


@pytest.mark.asyncio
async def test_one_unreadable_group_drops_the_readable_groups_grants_too(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1", "unreadable")))

    assert await can_user_read_vector_store(team_owned_store("store-a"), make_key(team_id=TEAM_ID)) is False


@pytest.mark.asyncio
async def test_failed_team_read_grants_nothing_and_keeps_direct_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    assert await can_user_read_vector_store(team_owned_store("store-a"), make_key(team_id=TEAM_ID)) is False
    assert (
        await can_user_read_vector_store(team_owned_store("store-a"), make_key(("store-a",), team_id=TEAM_ID)) is True
    )
    listed: Final = await filter_listable_vector_stores(
        (team_owned_store("store-a"),), make_key(("store-a",), team_id=TEAM_ID)
    )
    assert listed == (team_owned_store("store-a"),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_stores", "key_groups", "team_groups", "store", "allowed"),
    [
        pytest.param(None, (), ("team-a",), "store-a", True, id="team-group-and-unrestricted-key"),
        pytest.param(("store-x",), (), ("team-a",), "store-a", False, id="key-own-list-narrows-team-group"),
        pytest.param(None, ("key-b",), ("team-a",), "store-a", False, id="key-group-narrows-team-group"),
        pytest.param(None, ("key-b",), ("team-a",), "store-b", False, id="team-group-narrows-key-group"),
        pytest.param(None, ("team-a",), ("team-a",), "store-a", True, id="both-layers-list-the-store"),
        pytest.param(("store-a",), (), ("team-a",), "store-a", True, id="key-own-list-agrees-with-team-group"),
    ],
)
async def test_read_gate_and_list_apply_the_request_checks_rule_to_group_grants(
    proxy: ProxyState,
    key_stores: Sequence[str] | None,
    key_groups: Sequence[str],
    team_groups: Sequence[str],
    store: str,
    allowed: bool,
) -> None:
    await proxy.add_group("team-a", ("store-a",))
    await proxy.add_group("key-b", ("store-b",))
    await proxy.add_team(make_team(groups=team_groups))
    key: Final = make_key(key_stores, key_groups, TEAM_ID)

    assert await can_user_read_vector_store(team_owned_store(store), key) is allowed
    listed: Final = await filter_listable_vector_stores((team_owned_store(store),), key)
    assert listed == ((team_owned_store(store),) if allowed else ())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_groups", "allowed"),
    [pytest.param((), False, id="only-the-team-grants"), pytest.param(("g1",), True, id="key-and-team-grant")],
)
async def test_read_gate_requires_both_layers_under_deny_by_default(
    proxy: ProxyState, monkeypatch: pytest.MonkeyPatch, key_groups: Sequence[str], allowed: bool
) -> None:
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"vector_store_deny_by_default": True})
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))
    key: Final = UserAPIKeyAuth(token=TOKEN, api_key=TOKEN, team_id=TEAM_ID, access_group_ids=list(key_groups))
    key.via_virtual_key = True

    assert await can_user_read_vector_store(team_owned_store("store-a"), key) is allowed


@pytest.mark.asyncio
async def test_read_gate_and_list_match_a_group_store_by_its_provider_id(proxy: ProxyState) -> None:
    managed_id: Final = base64.urlsafe_b64encode(
        b"litellm_proxy:vector_store;unified_id,abc;target_model_names,gpt-4;resource_id,store-a"
    ).decode()
    await proxy.add_group("g1", (managed_id,))
    await proxy.add_team(make_team(groups=("g1",)))
    key: Final = make_key(team_id=TEAM_ID)

    assert await can_user_read_vector_store(team_owned_store("store-a"), key) is True
    assert await filter_listable_vector_stores((team_owned_store("store-a"),), key) == (team_owned_store("store-a"),)
    assert await can_user_read_vector_store(team_owned_store("store-b"), key) is False
    assert await can_user_read_vector_store(team_owned_store(managed_id), key) is True


@pytest.mark.asyncio
async def test_group_grant_for_one_managed_store_does_not_open_another_with_the_same_provider_id(
    proxy: ProxyState,
) -> None:
    def managed_id(unified_id: str) -> str:
        return base64.urlsafe_b64encode(
            f"litellm_proxy:vector_store;unified_id,{unified_id};target_model_names,gpt-4;resource_id,vs_shared".encode()
        ).decode()

    await proxy.add_group("g1", (managed_id("granted"),))
    await proxy.add_team(make_team(groups=("g1",)))
    key: Final = make_key(team_id=TEAM_ID)

    assert await can_user_read_vector_store(team_owned_store(managed_id("granted")), key) is True
    assert await can_user_read_vector_store(team_owned_store(managed_id("other")), key) is False
    assert await filter_listable_vector_stores((team_owned_store(managed_id("other")),), key) == ()


@pytest.mark.asyncio
async def test_dashboard_session_gets_the_group_stores_of_each_of_the_users_teams(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), team_ids=("team-1",))
    await proxy.add_group("g2", ("store-c",), team_ids=("team-2",))
    await proxy.cache.async_set_cache(
        key="team_id:team-1", value=LiteLLM_TeamTableCachedObj(team_id="team-1", access_group_ids=["g1"])
    )
    await proxy.cache.async_set_cache(
        key="team_id:team-2", value=LiteLLM_TeamTableCachedObj(team_id="team-2", access_group_ids=["g2"])
    )
    await proxy.cache.async_set_cache(
        key="session-user", value=LiteLLM_UserTable(user_id="session-user", teams=["team-1", "team-2"])
    )
    session: Final = UserAPIKeyAuth(token=TOKEN, user_id="session-user", team_id=UI_TEAM_ID)
    stores: Final = tuple(team_owned_store(store_id) for store_id in ("store-a", "store-b", "store-c"))

    assert await filter_listable_vector_stores(stores, session) == (stores[0], stores[2])
    assert await can_user_read_vector_store(stores[0], session) is True
    assert await can_user_read_vector_store(stores[1], session) is False
    assert await can_user_read_vector_store(stores[2], session) is True


READ_ROUTES: Final = (
    pytest.param("POST", "/v1/vector_stores/{}/search", {"query": "hi"}, id="search"),
    pytest.param("GET", "/v1/vector_stores/{}", None, id="retrieve"),
    pytest.param("GET", "/v1/vector_stores/{}/files", None, id="file-list"),
    pytest.param("GET", "/v1/vector_stores/{}/files/file-1", None, id="file-retrieve"),
    pytest.param("GET", "/v1/vector_stores/{}/files/file-1/content", None, id="file-content"),
    pytest.param("POST", "/vector_store/info", {"vector_store_id": "{}"}, id="info"),
    pytest.param(
        "POST",
        "/v1/rag/query",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "retrieval_config": {"vector_store_id": "{}"}},
        id="rag-query",
    ),
    pytest.param(
        "POST",
        "/v1/responses",
        {"model": "m", "input": "hi", "tools": [{"type": "file_search", "vector_store_ids": ["{}"]}]},
        id="responses",
    ),
)

MANAGE_ROUTES: Final = (
    pytest.param(
        "POST", "/vector_store/update", {"vector_store_id": "{}", "vector_store_name": "renamed"}, id="update"
    ),
    pytest.param("POST", "/vector_store/delete", {"vector_store_id": "{}"}, id="delete"),
    pytest.param("POST", "/v1/vector_stores/{}", {"name": "renamed"}, id="provider-update"),
    pytest.param("DELETE", "/v1/vector_stores/{}", None, id="provider-delete"),
    pytest.param("POST", "/v1/vector_stores/{}/files", {"file_id": "file-1"}, id="file-create"),
    pytest.param("POST", "/v1/vector_stores/{}/files/file-1", {"attributes": {"topic": "parks"}}, id="file-update"),
    pytest.param("DELETE", "/v1/vector_stores/{}/files/file-1", None, id="file-delete"),
)


def _body_for(body: Mapping[str, object] | None, store_id: str) -> object:
    return None if body is None else json.loads(json.dumps(body).replace("{}", store_id))


def _openai_answer(request: httpx.Request) -> httpx.Response:
    path: Final = request.url.path
    if path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )
    if path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "chat_1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    if path.endswith("/search"):
        return httpx.Response(
            200,
            json={
                "object": "vector_store.search_results.page",
                "search_query": "hi",
                "data": [],
                "has_more": False,
                "next_page": None,
            },
        )
    if path.endswith("/files"):
        return httpx.Response(
            200, json={"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}
        )
    if path.endswith("/content"):
        return httpx.Response(200, json={"object": "vector_store.file_content.page", "data": [], "has_more": False})
    if "/files/" in path:
        return httpx.Response(
            200,
            json={
                "id": "file-1",
                "object": "vector_store.file",
                "created_at": 1,
                "usage_bytes": 0,
                "vector_store_id": "store-a",
                "status": "completed",
            },
        )
    return httpx.Response(
        200,
        json={
            "id": "store-a",
            "object": "vector_store",
            "created_at": 1,
            "name": "store-a",
            "usage_bytes": 0,
            "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "status": "completed",
            "metadata": {},
        },
    )


@pytest.fixture
async def group_member(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, httpx_transport: None
) -> AsyncIterator[TestClient]:
    """A client for a team key whose team's group lists store-a. Another team owns both stores, and OpenAI is stubbed."""
    respx_mock.route(host="api.openai.com").mock(side_effect=_openai_answer)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[{"model_name": "m", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}}]
        ),
    )
    stores: Final = {store_id: team_owned_store(store_id) for store_id in ("store-a", "store-b")}
    proxy: Final = ProxyState(
        monkeypatch,
        store_reads=AsyncMock(
            side_effect=lambda where: MagicMock(model_dump=MagicMock(return_value=stores[where["vector_store_id"]]))
        ),
    )
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=list(stores.values())))
    original_overrides: Final = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: make_key(team_id=TEAM_ID)
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides = original_overrides


def _send(client: TestClient, method: str, path: str, body: Mapping[str, object] | None, store_id: str) -> int:
    return client.request(method, path.format(store_id), json=_body_for(body, store_id)).status_code


@pytest.mark.parametrize(("method", "path", "body"), READ_ROUTES)
def test_read_routes_serve_a_group_member_only_group_stores(
    group_member: TestClient, method: str, path: str, body: Mapping[str, object] | None
) -> None:
    assert _send(group_member, method, path, body, "store-a") == 200
    assert _send(group_member, method, path, body, "store-b") == 403


@pytest.mark.parametrize(("method", "path", "body"), MANAGE_ROUTES)
def test_manage_routes_refuse_a_group_member(
    group_member: TestClient, method: str, path: str, body: Mapping[str, object] | None
) -> None:
    assert _send(group_member, method, path, body, "store-a") == 403


def test_info_reads_a_database_store_through_the_read_gate(
    group_member: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vector_store_registry", None)

    assert group_member.post("/vector_store/info", json={"vector_store_id": "store-a"}).status_code == 200
    assert group_member.post("/vector_store/info", json={"vector_store_id": "store-b"}).status_code == 403


def test_info_refuses_a_group_store_the_keys_own_list_excludes(group_member: TestClient) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: make_key(("store-x",), team_id=TEAM_ID)

    assert group_member.post("/vector_store/info", json={"vector_store_id": "store-a"}).status_code == 403


def test_rag_ingest_refuses_a_group_member(group_member: TestClient) -> None:
    response: Final = group_member.post(
        "/v1/rag/ingest",
        json={
            "ingest_options": {"vector_store": {"custom_llm_provider": "openai", "vector_store_id": "store-a"}},
            "file_url": "https://example.com/a.txt",
        },
    )

    assert response.status_code == 403
