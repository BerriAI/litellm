import asyncio
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import HTTPException
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

import litellm.proxy.auth.user_api_key_auth as auth_module
import litellm.proxy.management_endpoints.access_group_endpoints as age
from litellm.proxy._experimental.mcp_server.management import server as mgmt_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.middleware.admission_control_middleware import (
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
)
from litellm.types.access_group import AccessGroupResponse


def _admin_caller(api_key: str, allowed_routes=None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        api_key=api_key,
        user_id=f"user-for-{api_key}",
        allowed_routes=allowed_routes,
    )


@pytest.fixture(autouse=True)
def _reset_server_state():
    yield
    # ensure no active server leaks between tests
    mgmt_server._active_server = None


@pytest.mark.asyncio
async def test_disabled_flag_serves_404():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/litellm-management/mcp",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("t", 80),
        "client": ("t", 1),
        "root_path": "",
        "http_version": "1.1",
    }
    request = Request(scope)
    response = await mgmt_server.handle_management_mcp_request(request)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_lifecycle_start_close_twice_creates_fresh_managers():
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True})
    first = mgmt_server._active_server
    assert first is not None
    assert mgmt_server.management_mcp_enabled() is True
    await mgmt_server.shutdown_management_mcp_server()
    assert mgmt_server.management_mcp_enabled() is False
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True})
    second = mgmt_server._active_server
    assert second is not None and second is not first
    assert second.manager is not first.manager
    await mgmt_server.shutdown_management_mcp_server()


@pytest.mark.asyncio
async def test_start_noop_when_flag_off_or_missing():
    await mgmt_server.start_management_mcp_server({})
    assert mgmt_server._active_server is None
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": False})
    assert mgmt_server._active_server is None


@pytest.mark.asyncio
async def test_admission_requires_admin_role(monkeypatch):
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True})

    async def _auth(request, api_key, **kwargs):
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER.value, api_key=api_key)

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/litellm-management/mcp",
        "headers": [(b"authorization", b"Bearer sk-user")],
        "query_string": b"",
        "scheme": "http",
        "server": ("t", 80),
        "client": ("t", 1),
        "root_path": "",
        "http_version": "1.1",
        "app": None,
    }
    with pytest.raises(HTTPException) as exc_info:
        await mgmt_server.handle_management_mcp_request(Request(scope))
    assert exc_info.value.status_code == 403
    await mgmt_server.shutdown_management_mcp_server()


@pytest.mark.asyncio
async def test_allowed_routes_mcp_routes_denied_management_routes_admitted(monkeypatch):
    """Use the real RouteChecks gate to prove the two groups split correctly."""
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True})

    async def _auth(request, api_key, **kwargs):
        caller = _admin_caller(api_key, allowed_routes=["mcp_routes"])
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route=get_request_route(request), valid_token=caller, request=request
        )
        return caller

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/litellm-management/mcp",
        "headers": [(b"authorization", b"Bearer sk-admin")],
        "query_string": b"",
        "scheme": "http",
        "server": ("t", 80),
        "client": ("t", 1),
        "root_path": "",
        "http_version": "1.1",
        "app": None,
    }
    with pytest.raises(HTTPException) as exc_info:
        await mgmt_server.handle_management_mcp_request(Request(scope))
    assert exc_info.value.status_code == 403

    async def _auth_mgmt(request, api_key, **kwargs):
        caller = _admin_caller(api_key, allowed_routes=["management_routes"])
        RouteChecks.is_virtual_key_allowed_to_call_route(
            route=get_request_route(request), valid_token=caller, request=request
        )
        return caller

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth_mgmt)

    async def _no_handler(**kwargs):
        raise RuntimeError("list_access_groups stub")

    monkeypatch.setattr(age, "list_access_groups", _no_handler)

    async def _empty_receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    # admission passes; the request reaches the transport (error surface is fine)
    response = await mgmt_server.handle_management_mcp_request(
        Request(
            {
                **scope,
                "headers": [
                    (b"authorization", b"Bearer sk-admin"),
                    (b"content-type", b"application/json"),
                    (b"accept", b"application/json, text/event-stream"),
                ],
            },
            _empty_receive,
        )
    )
    assert response.status_code in (200, 400, 406, 415, 500)
    await mgmt_server.shutdown_management_mcp_server()


def _access_group(group_id: str, name: str, owner: str) -> AccessGroupResponse:
    return AccessGroupResponse(
        access_group_id=group_id,
        access_group_name=name,
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
        created_by=owner,
        updated_at=datetime.now(timezone.utc),
        updated_by=owner,
    )


@pytest.mark.asyncio
async def test_in_process_mcp_client_end_to_end(monkeypatch):
    """Drive the real Streamable HTTP transport through an ASGI app behind
    admission control: initialize, list 10 tools, call list_access_groups, and
    prove two concurrent callers never see each other's results."""
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True})

    async def _auth(request, api_key, **kwargs):
        return _admin_caller(api_key)

    async def _list_access_groups(**kwargs):
        caller = kwargs["user_api_key_dict"]
        return [_access_group("g1", "grp", caller.user_id or "?")]

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)
    monkeypatch.setattr(age, "list_access_groups", _list_access_groups)

    starlette_app = Starlette(
        routes=[
            Route(
                "/litellm-management/mcp", mgmt_server.handle_management_mcp_request, methods=["GET", "POST", "DELETE"]
            ),
            Route(
                "/litellm-management/mcp/", mgmt_server.handle_management_mcp_request, methods=["GET", "POST", "DELETE"]
            ),
        ]
    )
    wrapped = AdmissionControlMiddleware(
        starlette_app,
        get_settings=lambda: AdmissionControlSettings(
            max_in_flight_requests=8, max_queued_requests=16, queue_timeout_seconds=5.0
        ),
        state=AdmissionControlState(lambda: None),
    )
    transport = httpx.ASGITransport(app=wrapped)

    async def _call(api_key: str):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers={"x-litellm-api-key": api_key}
        ) as http:
            async with streamable_http_client(
                "http://testserver/litellm-management/mcp",
                http_client=http,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    assert len(listed.tools) == 10
                    result = await session.call_tool("list_access_groups", {})
                    assert result.is_error is not True, result.content
                    return result.structured_content

    (res_a, res_b) = await asyncio.gather(_call("sk-admin-a"), _call("sk-admin-b"))
    assert res_a["result"][0]["created_by"] == "user-for-sk-admin-a"
    assert res_b["result"][0]["created_by"] == "user-for-sk-admin-b"

    await mgmt_server.shutdown_management_mcp_server()


@pytest.mark.asyncio
async def test_reserved_alias_rejected_on_config_and_rest():
    from litellm.proxy._experimental.mcp_server.utils import validate_mcp_server_name

    with pytest.raises(Exception, match="litellm-management"):
        validate_mcp_server_name("litellm-management")
    with pytest.raises(HTTPException) as exc_info:
        validate_mcp_server_name("litellm-management", raise_http_exception=True)
    assert exc_info.value.status_code == 400


def test_route_group_membership():
    from litellm.proxy._types import LiteLLMRoutes
    from litellm.proxy.auth.route_checks import RouteChecks

    assert RouteChecks.is_management_route("/litellm-management/mcp") is True
    assert RouteChecks.is_management_route("/litellm-management/mcp/") is True
    assert "/litellm-management/mcp" not in LiteLLMRoutes.mcp_routes.value
    assert "/litellm-management/mcp" not in LiteLLMRoutes.mcp_inference_routes.value
    assert "/litellm-management/mcp" not in LiteLLMRoutes.mcp_management_routes.value


def test_backend_allowlist():
    from backend.routes.allowlist import BACKEND_EXACT_PATHS

    assert "/litellm-management/mcp" in BACKEND_EXACT_PATHS
    assert "/litellm-management/mcp/" in BACKEND_EXACT_PATHS


def test_disabled_flag_404_and_dynamic_alias_route_via_fastapi_app():
    from fastapi.testclient import TestClient
    from litellm.proxy.proxy_server import app

    mgmt_server._active_server = None
    client = TestClient(app, raise_server_exceptions=False)
    try:
        assert client.post("/litellm-management/mcp", content=b"{}").status_code == 404

        mgmt_server._active_server = mgmt_server.ManagementMCPServer()
        assert client.post("/litellm-management/mcp", content=b"{}").status_code == 403
        assert client.post("/nonexistent-alias/mcp", content=b"{}").status_code == 404
    finally:
        mgmt_server._active_server = None
