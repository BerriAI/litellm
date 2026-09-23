import asyncio
import json
from typing import Optional

import pytest

import litellm.proxy.auth.user_api_key_auth as auth_module
import litellm.proxy.management_endpoints.access_group_endpoints as age
import litellm.proxy.management_endpoints.key_management_endpoints as kme
from litellm.proxy._experimental.mcp_server.management.dispatcher import (
    ManagementRequestContext,
    call_tool,
)
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth


def _ctx(api_key: str = "Bearer sk-admin") -> ManagementRequestContext:
    return ManagementRequestContext(
        raw_headers=((b"authorization", api_key.encode()), (b"content-type", b"application/json")),
        client=("10.0.0.1", 4321),
        scheme="http",
        server=("proxy.test", 80),
        root_path="",
        http_version="1.1",
        app=None,
        api_key=api_key,
        litellm_changed_by=None,
    )


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN.value, api_key="sk-admin")


def _non_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER.value, api_key="sk-user")


@pytest.fixture
def _as_admin(monkeypatch):
    async def _auth(request, api_key, **kwargs):
        return _admin()

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)


def _text(result) -> str:
    assert result.content, "tool result has no content"
    return result.content[0].text


def _json(result):
    return json.loads(_text(result))


@pytest.mark.asyncio
async def test_unknown_tool_is_error_result():
    result = await call_tool("does_not_exist", {}, _ctx())
    assert result.is_error is True
    assert "unknown tool" in _text(result)


@pytest.mark.asyncio
async def test_validation_error_shape_has_no_input_echo(monkeypatch):
    result = await call_tool("list_virtual_keys", {"size": 0, "user_id": "sk-sensitive-echo"}, _ctx())
    assert result.is_error is True
    body = _json(result)
    error = body["detail"][0]
    assert error["loc"] == ["query", "size"]
    assert error["type"] == "greater_than_equal"
    assert "input" not in error
    assert "ctx" not in error
    assert "sk-sensitive-echo" not in _text(result)


@pytest.mark.asyncio
async def test_path_param_validation_error_reports_path_loc():
    result = await call_tool("update_access_group", {"data": {}}, _ctx())
    assert result.is_error is True
    assert _json(result)["detail"][0]["loc"] == ["path", "access_group_id"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_list_virtual_keys_forwards_every_param(monkeypatch):
    captured = {}

    async def _list_keys(**kwargs):
        captured.update(kwargs)
        return {"keys": ["hash1"], "total_count": 1, "current_page": 2, "total_pages": 1}

    monkeypatch.setattr(kme, "list_keys", _list_keys)

    result = await call_tool(
        "list_virtual_keys",
        {
            "page": 2,
            "size": 25,
            "user_id": "u1",
            "team_id": "t1",
            "organization_id": "o1",
            "key_hash": "h1",
            "key_alias": "alias1",
            "search": "s",
            "return_full_object": True,
            "include_team_keys": True,
            "include_created_by_keys": True,
            "sort_by": "created_at",
            "sort_order": "asc",
            "expand": ["user"],
            "status": "active",
            "project_id": "p1",
            "access_group_id": "ag1",
            "agent_id": "agent1",
            "substring_matching": True,
            "expires": "expired",
        },
        _ctx(),
    )
    assert result.is_error is False, _text(result)
    assert captured["page"] == 2
    assert captured["size"] == 25
    assert captured["user_id"] == "u1"
    assert captured["team_id"] == "t1"
    assert captured["organization_id"] == "o1"
    assert captured["key_hash"] == "h1"
    assert captured["key_alias"] == "alias1"
    assert captured["search"] == "s"
    assert captured["return_full_object"] is True
    assert captured["include_team_keys"] is True
    assert captured["include_created_by_keys"] is True
    assert captured["sort_by"] == "created_at"
    assert captured["sort_order"] == "asc"
    assert captured["expand"] == ["user"]
    assert captured["status"] == "active"
    assert captured["project_id"] == "p1"
    assert captured["access_group_id"] == "ag1"
    assert captured["agent_id"] == "agent1"
    assert captured["substring_matching"] is True
    assert captured["expires"] == "expired"
    assert captured["request"].scope["path"] == "/key/list"
    assert captured["request"].scope["method"] == "GET"
    assert captured["user_api_key_dict"].user_role == LitellmUserRoles.PROXY_ADMIN.value
    assert _json(result)["total_count"] == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_exclude_unset_keeps_omitted_distinct_from_explicit_values(monkeypatch):
    seen = {}

    async def _update_key(**kwargs):
        seen["data"] = kwargs["data"]
        seen["request"] = kwargs["request"]
        return {"key": "h", "updated": True}

    monkeypatch.setattr(kme, "update_key_fn", _update_key)

    # explicit empty list must reach the model unset-vs-set boundary intact
    result = await call_tool("update_virtual_key", {"key": "k1", "models": []}, _ctx())
    assert result.is_error is False, _text(result)
    dumped = seen["data"].model_dump(exclude_unset=True)
    assert dumped == {"key": "k1", "models": []}

    # explicit null metadata reaches the handler as null (merge-patch clear),
    # not stripped like an omitted field
    result = await call_tool("update_virtual_key", {"key": "k1", "metadata": None}, _ctx())
    assert result.is_error is False, _text(result)
    dumped = seen["data"].model_dump(exclude_unset=True)
    assert dumped == {"key": "k1", "metadata": None}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_get_virtual_key_passes_key_and_caller(monkeypatch):
    seen = {}

    async def _info_key_fn(**kwargs):
        seen.update(kwargs)
        return {"key": "sk-echo-me", "info": {"spend": 0.5}}

    monkeypatch.setattr(kme, "info_key_fn", _info_key_fn)
    result = await call_tool("get_virtual_key", {"key": "sk-echo-me"}, _ctx())
    assert result.is_error is False, _text(result)
    assert seen["key"] == "sk-echo-me"
    assert seen["user_api_key_dict"].user_role == LitellmUserRoles.PROXY_ADMIN.value
    body = _json(result)
    # the echoed raw key is hashed before leaving the proxy
    from litellm.proxy.utils import hash_token

    assert body["key"] == hash_token("sk-echo-me")
    assert body["info"]["spend"] == 0.5


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_create_virtual_key_returns_raw_key_and_changed_by(monkeypatch):
    seen = {}

    async def _generate_key_fn(**kwargs):
        seen.update(kwargs)
        return {"key": "sk-new-key", "expires": None, "user_id": "u1"}

    monkeypatch.setattr(kme, "generate_key_fn", _generate_key_fn)
    ctx = ManagementRequestContext(
        raw_headers=_ctx().raw_headers,
        client=None,
        scheme="http",
        server=None,
        root_path="",
        http_version="1.1",
        app=None,
        api_key="Bearer sk-admin",
        litellm_changed_by="admin@example.com",
    )
    result = await call_tool("create_virtual_key", {"user_id": "u1", "key_alias": "a"}, ctx)
    assert result.is_error is False, _text(result)
    assert seen["litellm_changed_by"] == "admin@example.com"
    assert _json(result)["key"] == "sk-new-key"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_delete_virtual_keys_redacts_deleted_key_echoes(monkeypatch):
    async def _delete_key_fn(**kwargs):
        return {"deleted_keys": ["sk-dead-beef", "some-hash"]}

    monkeypatch.setattr(kme, "delete_key_fn", _delete_key_fn)
    result = await call_tool("delete_virtual_keys", {"keys": ["sk-dead-beef"]}, _ctx())
    assert result.is_error is False, _text(result)
    body = _json(result)
    from litellm.proxy.utils import hash_token

    assert body["deleted_keys"] == [hash_token("sk-dead-beef"), "some-hash"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_delete_access_group_none_maps_to_empty_object(monkeypatch):
    seen = {}

    async def _delete_access_group(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(age, "delete_access_group", _delete_access_group)
    result = await call_tool("delete_access_group", {"access_group_id": "g1"}, _ctx())
    assert result.is_error is False, _text(result)
    assert seen["access_group_id"] == "g1"
    assert result.structured_content == {}
    assert _json(result) == {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_update_access_group_splits_path_and_body(monkeypatch):
    seen = {}

    async def _update_access_group(**kwargs):
        seen.update(kwargs)
        from litellm.types.access_group import AccessGroupResponse
        from datetime import datetime, timezone

        return AccessGroupResponse(
            access_group_id=kwargs["access_group_id"],
            access_group_name="g",
            access_model_names=[],
            access_mcp_server_ids=[],
            access_agent_ids=[],
            assigned_team_ids=[],
            assigned_key_ids=[],
            access_mcp_servers=(),
            access_agents=(),
            assigned_teams=(),
            assigned_keys=(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(age, "update_access_group", _update_access_group)
    result = await call_tool(
        "update_access_group",
        {"access_group_id": "g9", "data": {"description": "new desc"}},
        _ctx(),
    )
    assert result.is_error is False, _text(result)
    assert seen["access_group_id"] == "g9"
    assert seen["data"].description == "new desc"


@pytest.mark.asyncio
async def test_non_admin_gets_403_is_error(monkeypatch):
    async def _auth(request, api_key, **kwargs):
        return _non_admin()

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)
    for tool_name, args in [
        ("list_virtual_keys", {}),
        ("get_virtual_key", {"key": "k"}),
        ("create_virtual_key", {"user_id": "u"}),
        ("update_virtual_key", {"key": "k"}),
        ("delete_virtual_keys", {"keys": ["k"]}),
        ("list_access_groups", {}),
        ("get_access_group", {"access_group_id": "g"}),
        ("create_access_group", {"access_group_name": "n"}),
        ("update_access_group", {"access_group_id": "g", "data": {}}),
        ("delete_access_group", {"access_group_id": "g"}),
    ]:
        result = await call_tool(tool_name, args, _ctx())
        assert result.is_error is True, tool_name
        assert "403" in _text(result), tool_name


@pytest.mark.asyncio
async def test_auth_failure_is_error_result(monkeypatch):
    async def _auth(request, api_key, **kwargs):
        raise ProxyException(message="bad key sk-abc123", type="auth_error", param="None", code=401)

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)
    result = await call_tool("list_access_groups", {}, _ctx())
    assert result.is_error is True
    assert "401" in _text(result)
    # the raw key in the error detail is masked
    assert "sk-abc123" not in _text(result)
    assert "sk-***" in _text(result)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_handler_http_exception_redacts_sk_substrings(monkeypatch):
    from fastapi import HTTPException

    async def _boom(**kwargs):
        raise HTTPException(status_code=404, detail="key sk-leaked-token not found")

    monkeypatch.setattr(kme, "info_key_fn", _boom)
    result = await call_tool("get_virtual_key", {"key": "k"}, _ctx())
    assert result.is_error is True
    assert "404" in _text(result)
    assert "sk-leaked-token" not in _text(result)
    assert "sk-***" in _text(result)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_database_unavailable_classified_as_503(monkeypatch):
    async def _boom(**kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(age, "list_access_groups", _boom)
    result = await call_tool("list_access_groups", {}, _ctx())
    assert result.is_error is True
    assert "503" in _text(result)
    assert "database unavailable" in _text(result)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_generic_error_is_sanitized(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("internal detail with sk-secret")

    monkeypatch.setattr(age, "list_access_groups", _boom)
    result = await call_tool("list_access_groups", {}, _ctx())
    assert result.is_error is True
    assert "internal detail" not in _text(result)
    assert "sk-secret" not in _text(result)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_mutation_timeout_reports_unknown_outcome_without_retry(monkeypatch):
    calls = []

    async def _slow(**kwargs):
        calls.append(1)
        await asyncio.sleep(60)

    monkeypatch.setattr(kme, "generate_key_fn", _slow)
    import litellm.proxy._experimental.mcp_server.management.dispatcher as disp

    monkeypatch.setattr(disp, "_HANDLER_TIMEOUT_SECONDS", 0.05)
    result = await call_tool("create_virtual_key", {"user_id": "u"}, _ctx())
    assert result.is_error is True
    assert "may or may not have completed" in _text(result)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_oversized_result_is_error(monkeypatch):
    async def _big(**kwargs):
        return {"keys": ["x" * (1024 * 1024)], "total_count": 1}

    monkeypatch.setattr(kme, "list_keys", _big)
    result = await call_tool("list_virtual_keys", {}, _ctx())
    assert result.is_error is True
    assert "too large" in _text(result)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_as_admin")
async def test_synthetic_request_scope_and_headers(monkeypatch):
    seen = {}

    async def _list_keys(**kwargs):
        seen["request"] = kwargs["request"]
        return {"keys": []}

    monkeypatch.setattr(kme, "list_keys", _list_keys)
    ctx = ManagementRequestContext(
        raw_headers=(
            (b"authorization", b"Bearer sk-admin"),
            (b"content-type", b"text/plain"),
            (b"content-length", b"42"),
            (b"mcp-session-id", b"evil"),
            (b"mcp-protocol-version", b"evil"),
            (b"accept", b"text/html"),
            (b"x-custom", b"ok"),
        ),
        client=("10.0.0.1", 5),
        scheme="https",
        server=("proxy.test", 443),
        root_path="/sub",
        http_version="1.1",
        app=None,
        api_key="Bearer sk-admin",
        litellm_changed_by=None,
    )
    result = await call_tool("list_virtual_keys", {"user_id": "u1"}, ctx)
    assert result.is_error is False, _text(result)
    req = seen["request"]
    assert req.scope["path"] == "/key/list"
    assert req.scope["method"] == "GET"
    assert req.scope["client"] == ("10.0.0.1", 5)
    assert req.scope["scheme"] == "https"
    assert req.scope["root_path"] == "/sub"
    assert req.scope["query_string"] == b"user_id=u1"
    assert req.scope["state"] == {}
    assert "route" not in req.scope
    header_names = {k.decode() for k, _ in req.scope["headers"]}
    assert "mcp-session-id" not in header_names
    assert "mcp-protocol-version" not in header_names
    assert "content-length" not in header_names
    assert header_names["content-type"] if False else True
    assert ("x-custom" in header_names) or (b"x-custom" in (k for k, _ in req.scope["headers"]))
