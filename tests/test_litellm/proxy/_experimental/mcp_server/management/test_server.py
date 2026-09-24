import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

import litellm.proxy.auth.user_api_key_auth as auth_module
from litellm.proxy._experimental.mcp_server.management import server as mgmt_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.middleware.admission_control_middleware import (
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
)


def _fixture_app() -> FastAPI:
    app = FastAPI()

    @app.get("/admin/keys", operation_id="list_admin_keys")
    async def list_admin_keys(request: FastAPIRequest):
        from litellm.proxy import proxy_server
        from litellm.proxy.auth.auth_utils import _get_request_ip_address
        from litellm.proxy.auth.ip_address_utils import IPAddressUtils

        return {
            "keys": [],
            "caller": request.headers.get("x-litellm-api-key") or request.headers.get("authorization"),
            "client_ip": _get_request_ip_address(
                request, use_x_forwarded_for=proxy_server.general_settings.get("use_x_forwarded_for") is True
            ),
            "mcp_client_ip": IPAddressUtils.get_mcp_client_ip(request),
        }

    return app


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
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())
    first = mgmt_server._active_server
    assert first is not None
    assert mgmt_server.management_mcp_enabled() is True
    await mgmt_server.shutdown_management_mcp_server()
    assert mgmt_server.management_mcp_enabled() is False
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())
    second = mgmt_server._active_server
    assert second is not None and second is not first
    assert second.manager is not first.manager
    await mgmt_server.shutdown_management_mcp_server()


@pytest.mark.asyncio
async def test_start_noop_when_flag_off_or_missing():
    await mgmt_server.start_management_mcp_server({}, _fixture_app())
    assert mgmt_server._active_server is None
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": False}, _fixture_app())
    assert mgmt_server._active_server is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.PROXY_ADMIN,
    ],
)
async def test_admission_preserves_authenticated_role(monkeypatch, role):
    caller = UserAPIKeyAuth(user_role=role.value, api_key="sk-user")

    async def _auth(request, api_key, **kwargs):
        assert await request.body() == b""
        return caller

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
    authenticated = await mgmt_server._authenticate_admission(Request(scope))
    assert authenticated is caller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "Bearer sk-header-parity"},
        {"x-litellm-api-key": "sk-header-parity"},
        {"x-litellm-api-key": "Bearer sk-header-parity"},
        {"x-litellm-api-key": "sk-header-parity", "authorization": "Bearer sk-wrong"},
    ],
    ids=["authorization", "raw-custom-header", "bearer-custom-header", "custom-header-precedence"],
)
async def test_admission_uses_rest_credential_header_parsing(monkeypatch, headers):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-header-parity")
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/litellm-management/mcp",
            "headers": [(key.encode(), value.encode()) for key, value in headers.items()],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 1234),
            "root_path": "",
        }
    )
    authenticated = await mgmt_server._authenticate_admission(request)
    assert authenticated.user_role == LitellmUserRoles.PROXY_ADMIN
    assert dict(request.headers) == headers


@pytest.mark.asyncio
async def test_allowed_routes_mcp_routes_denied_management_routes_admitted(monkeypatch):
    """Use the real RouteChecks gate to prove the two groups split correctly."""
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_forwarded,trusted",
    [(True, False), (False, False), (True, True)],
    ids=["untrusted-forwarding", "direct-peer", "trusted-hop-chain"],
)
async def test_in_process_mcp_client_end_to_end(monkeypatch, use_forwarded, trusted):
    """Drive the real Streamable HTTP transport behind admission control:
    initialize, list the catalog tools, call one, and prove two concurrent
    callers reusing one transport only ever see their own credential."""
    from litellm.proxy import proxy_server

    settings = {"use_x_forwarded_for": use_forwarded}
    if trusted:
        settings.update({"mcp_trusted_proxy_ranges": ["127.0.0.0/8"], "mcp_xff_num_trusted_hops": 1})
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())

    async def _auth(request, api_key, **kwargs):
        return _admin_caller(api_key)

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)

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

    async def _call(api_key: str, client_ip: str):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"x-litellm-api-key": api_key, "x-forwarded-for": client_ip},
        ) as http:
            async with streamable_http_client(
                "http://testserver/litellm-management/mcp",
                http_client=http,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    assert [tool.name for tool in listed.tools] == ["list_admin_keys"]
                    result = await session.call_tool("list_admin_keys", {})
                    assert result.is_error is not True, result.content
                    return json.loads(result.content[0].text)

    forwarded_a = "10.0.0.1, 203.0.113.1" if trusted else "203.0.113.1"
    forwarded_b = "10.0.0.1, 203.0.113.2" if trusted else "203.0.113.2"
    (res_a, res_b) = await asyncio.gather(_call("sk-admin-a", forwarded_a), _call("sk-admin-b", forwarded_b))
    assert res_a["caller"] == "sk-admin-a"
    assert res_b["caller"] == "sk-admin-b"
    assert res_a["client_ip"] == (forwarded_a if use_forwarded else "127.0.0.1")
    assert res_b["client_ip"] == (forwarded_b if use_forwarded else "127.0.0.1")
    assert res_a["mcp_client_ip"] == ("203.0.113.1" if trusted else ("" if use_forwarded else "127.0.0.1"))
    assert res_b["mcp_client_ip"] == ("203.0.113.2" if trusted else ("" if use_forwarded else "127.0.0.1"))

    await mgmt_server.shutdown_management_mcp_server()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin, status",
    [
        ("https://untrusted.example", 403),
        ("null", 403),
        ("https://allowed.example.attacker.example", 403),
        ("https://allowed.example", 200),
    ],
)
async def test_origin_requires_explicit_allowed_origin(monkeypatch, origin, status):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "origins", ["*", "https://allowed.example"])
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())

    async def _auth(request, api_key, **kwargs):
        return _admin_caller(api_key)

    monkeypatch.setattr(auth_module, "user_api_key_auth", _auth)
    app = Starlette(
        routes=[Route("/litellm-management/mcp", mgmt_server.handle_management_mcp_request, methods=["POST"])]
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/litellm-management/mcp",
            headers={
                "authorization": "Bearer sk-caller",
                "origin": origin,
                "accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
    assert response.status_code == status
    if status == 200:
        assert response.json()["result"]["serverInfo"]["name"] == "litellm-management"
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

    from litellm.proxy._experimental.mcp_server.management.catalog import build_catalog
    from litellm.proxy.proxy_server import app

    mgmt_server._active_server = None
    client = TestClient(app, raise_server_exceptions=False)
    try:
        assert client.post("/litellm-management/mcp", content=b"{}").status_code == 404

        mgmt_server._active_server = mgmt_server.ManagementMCPServer(build_catalog({}))
        assert client.post("/litellm-management/mcp", content=b"{}").status_code == 401
        assert client.post("/nonexistent-alias/mcp", content=b"{}").status_code == 404
    finally:
        mgmt_server._active_server = None


@pytest.mark.asyncio
async def test_configured_credential_header_admission_and_forwarding(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-configured-header")
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "X-Company-Key"})
    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())
    observed = []

    async def bridge(handle_fn, scope, receive):
        observed.append(mgmt_server._request_context.get())
        return httpx.Response(200)

    monkeypatch.setattr(proxy_server, "_stream_mcp_asgi_response", bridge)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/litellm-management/mcp",
            "headers": [(b"x-company-key", b"Bearer sk-configured-header")],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 1234),
            "root_path": "",
        }
    )
    try:
        response = await mgmt_server.handle_management_mcp_request(request)
        assert response.status_code == 200
        assert observed[0].credential_header == "x-company-key"
        assert observed[0].credential_value == "Bearer sk-configured-header"
    finally:
        await mgmt_server.shutdown_management_mcp_server()


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_fails", [False, True], ids=["clean-close", "failed-close"])
async def test_shutdown_closes_manager_before_client(monkeypatch, manager_fails):
    from litellm.proxy._experimental.mcp_server.management import dispatcher

    await mgmt_server.start_management_mcp_server({"enable_management_mcp": True}, _fixture_app())
    server = mgmt_server._active_server
    client = dispatcher._active_dispatch.http_client

    async def close_manager():
        assert not client.is_closed
        if manager_fails:
            raise RuntimeError("manager close failed")

    monkeypatch.setattr(server, "close", close_manager)
    if manager_fails:
        with pytest.raises(RuntimeError, match="manager close failed"):
            await mgmt_server.shutdown_management_mcp_server()
    else:
        await mgmt_server.shutdown_management_mcp_server()
    assert client.is_closed
