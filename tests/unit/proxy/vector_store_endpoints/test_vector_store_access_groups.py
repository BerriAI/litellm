"""A store an access group lists is answered exactly like the same store on the key's or team's own list."""

import json
from collections.abc import Mapping, Sequence
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import UI_TEAM_ID, LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import hash_token
from litellm.proxy.vector_store_endpoints.utils import can_user_access_vector_store, filter_listable_vector_stores
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry
from tests._master_key import MASTER_KEY
from tests.unit.proxy.vector_store_access_group_state import (
    TEAM_ID,
    TOKEN,
    ProxyState,
    make_key,
    make_team,
    own_permission,
    team_owned_store,
)

Layer = Literal["key", "team"]
_KEY: Final = "sk-group-store"
_STORES: Final = (
    team_owned_store("store-a"),
    team_owned_store("store-b"),
    LiteLLM_ManagedVectorStore(vector_store_id="store-free", custom_llm_provider="openai"),
)


async def _caller(proxy: ProxyState, layer: Layer, how: Literal["own-list", "group"]) -> UserAPIKeyAuth:
    """A team key granted store-a on `layer`, either on that layer's own list or through its access group."""
    if how == "own-list":
        await proxy.add_team(make_team(("store-a",) if layer == "team" else None))
        key: Final = make_key(("store-a",) if layer == "key" else None, team_id=TEAM_ID)
        return (
            key.model_copy(update={"team_object_permission": own_permission(("store-a",))}) if layer == "team" else key
        )
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",) if layer == "team" else ()))
    return make_key(groups=("g1",) if layer == "key" else (), team_id=TEAM_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["key", "team"])
@pytest.mark.parametrize("how", ["own-list", "group"])
async def test_the_store_gate_and_the_list_answer_a_group_store_like_an_own_list_store(
    monkeypatch: pytest.MonkeyPatch, layer: Layer, how: Literal["own-list", "group"]
) -> None:
    caller: Final = await _caller(ProxyState(monkeypatch), layer, how)

    assert [await can_user_access_vector_store(store, caller) for store in _STORES] == [True, False, True]
    assert await filter_listable_vector_stores(_STORES, caller) == (_STORES[0],)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_is_readable", "allowed"),
    [pytest.param(True, [True, True, True], id="readable"), pytest.param(False, [True, False, True], id="unreadable")],
)
async def test_a_team_that_cannot_be_read_loses_its_group_stores_and_the_keys_still_count(
    monkeypatch: pytest.MonkeyPatch, team_is_readable: bool, allowed: Sequence[bool]
) -> None:
    """The key's group lists store-a and names only the key's team. The team's group lists store-b."""
    proxy: Final = ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    await proxy.add_group("key-group", ("store-a",), key_ids=())
    await proxy.add_group("team-group", ("store-b",))
    if team_is_readable:
        await proxy.add_team(make_team(groups=("team-group",)))
    caller: Final = make_key(groups=("key-group",), team_id=TEAM_ID)

    assert [await can_user_access_vector_store(store, caller) for store in _STORES] == allowed


@pytest.mark.asyncio
async def test_a_caller_without_groups_keeps_its_answers_when_its_team_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    caller: Final = make_key(("store-a",), team_id=TEAM_ID)

    assert [await can_user_access_vector_store(store, caller) for store in _STORES] == [True, False, True]
    assert await filter_listable_vector_stores(_STORES, caller) == (_STORES[0],)


@pytest.mark.asyncio
async def test_a_group_that_cannot_be_read_grants_nothing_and_own_list_stores_still_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    await proxy.add_team(make_team(groups=("g1",)))
    caller: Final = make_key(("store-b",), ("g1",), TEAM_ID)

    assert [await can_user_access_vector_store(store, caller) for store in _STORES] == [False, True, True]


@pytest.mark.asyncio
async def test_a_dashboard_session_gets_the_group_stores_of_each_of_the_users_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy: Final = ProxyState(monkeypatch)
    await proxy.add_group("g1", ("store-a",), team_ids=("team-1",))
    await proxy.add_group("g2", ("store-c",), team_ids=("team-2",))
    for team_id, group in (("team-1", "g1"), ("team-2", "g2")):
        await proxy.cache.async_set_cache(
            key=f"team_id:{team_id}", value=LiteLLM_TeamTableCachedObj(team_id=team_id, access_group_ids=[group])
        )
    await proxy.cache.async_set_cache(
        key="session-user", value=LiteLLM_UserTable(user_id="session-user", teams=["team-1", "team-2"])
    )
    session: Final = UserAPIKeyAuth(token=TOKEN, user_id="session-user", team_id=UI_TEAM_ID)
    stores: Final = tuple(team_owned_store(store_id) for store_id in ("store-a", "store-b", "store-c"))

    assert await filter_listable_vector_stores(stores, session) == (stores[0], stores[2])
    assert [await can_user_access_vector_store(store, session) for store in stores] == [True, False, True]


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
    if request.method == "DELETE":
        return httpx.Response(
            200, json={"id": path.rsplit("/", 1)[-1], "object": "vector_store.deleted", "deleted": True}
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
def openai_stub(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, httpx_transport: None) -> None:
    respx_mock.route(host="api.openai.com").mock(side_effect=_openai_answer)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(
            model_list=[{"model_name": "m", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}}]
        ),
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=list(_STORES)))


async def _statuses(
    monkeypatch: pytest.MonkeyPatch,
    layer: Layer,
    how: Literal["own-list", "group"],
    method: str,
    path: str,
    body: Mapping[str, object] | None,
) -> Sequence[int]:
    stores: Final = {store["vector_store_id"]: store for store in _STORES}

    def row(where: Mapping[str, str], data: object = None) -> MagicMock:
        return MagicMock(model_dump=MagicMock(return_value=stores[where["vector_store_id"]]))

    proxy: Final = ProxyState(monkeypatch, store_reads=AsyncMock(side_effect=row))
    monkeypatch.setattr(proxy.prisma.db.litellm_managedvectorstorestable, "update", AsyncMock(side_effect=row))
    monkeypatch.setattr(proxy.prisma.db.litellm_managedvectorstorestable, "delete", AsyncMock())
    token: Final = hash_token(_KEY)
    caller: Final = (await _caller(proxy, layer, how)).model_copy(
        update={"token": token, "allowed_routes": [path] if path.startswith("/vector_store/") else []}
    )
    await proxy.cache.async_set_cache(key=token, value=caller)
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    client: Final = TestClient(app, raise_server_exceptions=False)
    return [
        client.request(
            method,
            path.format(store_id),
            json=_body_for(body, store_id),
            headers={"Authorization": f"Bearer {_KEY}"},
        ).status_code
        for store_id in stores
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["key", "team"])
@pytest.mark.parametrize(("method", "path", "body"), [*READ_ROUTES, *MANAGE_ROUTES])
async def test_every_store_route_answers_a_group_store_like_an_own_list_store(
    monkeypatch: pytest.MonkeyPatch,
    openai_stub: None,
    layer: Layer,
    method: str,
    path: str,
    body: Mapping[str, object] | None,
) -> None:
    """Real auth runs the request check and the route's store gate. Statuses are for store-a (granted),
    store-b (another team's) and store-free (no team), in that order."""
    listed: Final = await _statuses(monkeypatch, layer, "own-list", method, path, body)
    grouped: Final = await _statuses(monkeypatch, layer, "group", method, path, body)

    assert grouped == listed
    assert grouped[0] == 200
    assert grouped[1] in (401, 403)
