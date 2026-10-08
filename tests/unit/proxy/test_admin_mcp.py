import asyncio
import json
import sys
from typing import Final

import httpx2
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.testclient import TestClient

from litellm.constants import STANDARD_CUSTOMER_ID_HEADERS
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_UserTable, ProxyException, SpecialHeaders
from litellm.proxy.admin_mcp import admin_mcp_lifespan
from litellm.proxy.auth.user_api_key_auth import get_api_key, get_api_key_from_custom_header
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_endpoints import internal_user_endpoints
from litellm.proxy.middleware.admission_control_middleware import (
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
    AdmissionControlStats,
)
from litellm.proxy.middleware.per_request_root_path_middleware import PerRequestRootPathMiddleware


class KeyRequest(BaseModel):
    key_alias: str


def _native_credential(request: Request) -> str:
    key, _ = get_api_key(
        custom_litellm_key_header=request.headers.get("x-litellm-api-key"),
        api_key=request.headers.get("authorization", ""),
        azure_api_key_header=request.headers.get("api-key"),
        anthropic_api_key_header=request.headers.get("x-api-key"),
        google_ai_studio_api_key_header=request.headers.get("x-goog-api-key"),
        azure_apim_header=request.headers.get("ocp-apim-subscription-key"),
        pass_through_endpoints=None, route=request.url.path, request=request,
    )
    configured: Final = proxy_server.general_settings.get("litellm_key_header_name")
    return get_api_key_from_custom_header(request, configured) if configured is not None else key


@pytest.fixture
def management_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", "true")
    monkeypatch.setenv("LITELLM_ADMIN_TOOLS", "list_keys")
    for name in (
        "PROXY_BASE_URL", "LITELLM_MCP_PUBLIC_URL", "LITELLM_ADMIN_READ_ONLY",
        "LITELLM_ADMIN_RESPONSE_VIEW", "LITELLM_ADMIN_SCHEMA_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    app: Final = FastAPI(lifespan=admin_mcp_lifespan)

    @app.get("/user/info")
    async def user_info(request: Request) -> dict[str, object]:
        await asyncio.sleep(0)
        credential: Final = _native_credential(request)
        if credential == "team-key":
            raise HTTPException(status_code=404, detail="User None not found")
        if credential not in {"admin-a", "admin-b", "admin-b-limited", "member", "viewer"}:
            raise HTTPException(status_code=401)
        user_id: Final = "admin-b" if credential == "admin-b-limited" else credential
        return {
            "user_id": user_id,
            "user_info": {
                "user_id": user_id,
                "user_role": {"member": "internal_user", "viewer": "proxy_admin_viewer"}.get(user_id, "proxy_admin"),
            },
        }

    @app.get("/key/list", operation_id="list_keys_key_list_get")
    async def list_keys(request: Request, large: bool = False) -> dict[str, object]:
        credential: Final = _native_credential(request)
        if credential == "admin-b-limited":
            raise HTTPException(403, "This key cannot list keys")
        if large:
            return {"keys": [{"key_alias": ("a" if credential == "admin-a" else "b") * 20000}]}
        return {
            "keys": ["owned-by-a" if credential == "admin-a" else "owned-by-b"],
            "client": request.client.host if request.client else None,
            "scheme": request.url.scheme,
            "forwarded_for": request.headers.get("x-forwarded-for"),
            "cookie": request.headers.get("cookie"),
            "caller_is_changed_by": request.headers.get("litellm-changed-by")
            == credential,
            "policy_team": request.headers.get("x-litellm-team-id"),
            "alternate_credentials_absent": not any(
                name in request.headers
                for name in SpecialHeaders.litellm_credential_header_names() - {"authorization"}
            ),
        }

    app.state.created_aliases = []

    @app.post("/key/generate", operation_id="generate_key_fn_key_generate_post")
    async def create_key(payload: KeyRequest) -> dict[str, str]:
        app.state.created_aliases.append(payload.key_alias)
        if payload.key_alias == "fail-after-write":
            raise HTTPException(status_code=500)
        return {"key": "sk-new-key", "key_alias": payload.key_alias}

    @app.post("/mcp")
    async def existing_mcp() -> dict[str, str]:
        return {"server": "existing"}

    @app.post("/{server_name}/mcp")
    async def existing_namespace(server_name: str) -> dict[str, str]:
        return {"server": server_name}

    return app


@pytest.mark.parametrize("enabled", [None, "false", "0", "off", "No"])
def test_disabled_preserves_existing_admin_namespace(monkeypatch: pytest.MonkeyPatch, enabled: str | None) -> None:
    monkeypatch.setattr(proxy_server, "premium_user", False)
    monkeypatch.setitem(sys.modules, "litellm_admin_mcp.config", None)
    if enabled is None:
        monkeypatch.delenv("LITELLM_ENABLE_ADMIN_MCP", raising=False)
    else:
        monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", enabled)
    app: Final = FastAPI(lifespan=admin_mcp_lifespan)

    @app.post("/{server_name}/mcp")
    async def namespace(server_name: str) -> dict[str, str]:
        return {"server": server_name}

    with TestClient(app) as client:
        response: Final = client.post("/admin/mcp")
    assert response.status_code == 200
    assert response.json() == {"server": "admin"}


def test_enabled_without_connector_explains_installation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", "true")
    monkeypatch.setitem(sys.modules, "litellm_admin_mcp.config", None)
    with pytest.raises(RuntimeError, match="admin-mcp dependency group"):
        with TestClient(FastAPI(lifespan=admin_mcp_lifespan)):
            pytest.fail("Enabling the connector without its dependency must fail startup")


@pytest.mark.parametrize("enabled", ["true", "Yes", "on"])
def test_unlicensed_opt_in_fails_before_loading_connector(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, enabled: str
) -> None:
    monkeypatch.setattr(proxy_server, "premium_user", False)
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", enabled)
    monkeypatch.setitem(sys.modules, "litellm_admin_mcp.config", None)
    with pytest.raises(HTTPException) as exc:
        with TestClient(management_app):
            pytest.fail("An unlicensed deployment must not serve the hosted admin connector")
    assert exc.value.status_code == 403
    assert "LITELLM_LICENSE" in str(exc.value.detail)
    assert all(route.name != "admin_mcp" for route in management_app.routes)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_losing_enterprise_status_blocks_admin_tool_calls(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_ADMIN_TOOLS", "create_key")
    headers: Final = {"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"}
    payload: Final = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "create_key", "arguments": {"body": {"key_alias": "licensed-write"}}},
    }
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        licensed: Final = client.post("/admin/mcp", headers=headers, json=payload)
        assert licensed.status_code == 200, licensed.text
        assert licensed.json()["result"]["isError"] is False
        monkeypatch.setattr(proxy_server, "premium_user", False)
        denied: Final = client.post("/admin/mcp", headers=headers, json=payload)
    assert denied.status_code == 403, denied.text
    assert "LITELLM_LICENSE" in denied.text
    assert management_app.state.created_aliases == ["licensed-write"]


def test_invalid_flag_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", "treu")
    with pytest.raises(ValueError, match="LITELLM_ENABLE_ADMIN_MCP"):
        with TestClient(FastAPI(lifespan=admin_mcp_lifespan)):
            pytest.fail("Invalid flag must fail startup")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize(
    "authorization,status",
    [(None, 401), ("Bearer invalid", 401), ("Bearer member", 403), ("Bearer viewer", 403), ("Bearer team-key", 403)],
)
def test_admin_endpoint_rejects_unauthorized_callers(
    management_app: FastAPI, authorization: str | None, status: int
) -> None:
    headers: Final = {
        **({"Authorization": authorization} if authorization else {}),
        "x-litellm-api-key": "Bearer admin-a",
    }
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        response: Final = client.post("/admin/mcp", headers=headers, json={})
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_mount_keeps_existing_mcp_and_manages_restart(management_app: FastAPI) -> None:
    for _ in range(2):
        with TestClient(management_app, base_url="http://localhost:4000") as client:
            assert client.post("/mcp").json() == {"server": "existing"}
            assert client.post("/tools/mcp").json() == {"server": "tools"}
            response: Final = client.post(
                "/admin/mcp",
                headers={"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            assert response.status_code == 200, response.text
            assert {tool["name"] for tool in response.json()["result"]["tools"]} == {
                "list_keys",
                "describe_admin_tool",
                "read_admin_result",
            }
        assert all(route.name != "admin_mcp" for route in management_app.routes)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize("root_path", ["", "/gateway"])
@pytest.mark.parametrize("prefix_mode", ["ingress", "scalar", "multiple"])
async def test_concurrent_calls_preserve_identity_network_context_and_strip_cookies(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, root_path: str, prefix_mode: str
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com" + root_path)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://must-not-call.example.com")
    monkeypatch.setenv("LITELLM_API_KEY", "must-not-use-shared-credential")
    if prefix_mode == "scalar":
        management_app.root_path = root_path
    elif prefix_mode == "multiple":
        management_app.add_middleware(PerRequestRootPathMiddleware, root_paths=("/other", root_path))

    async def call(credential: str) -> dict[str, object]:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(
                app=management_app,
                root_path=root_path if prefix_mode == "ingress" else "",
                client=("198.51.100.7", 4567),
            ),
            base_url="https://gateway.example.com",
        ) as client:
            response: Final = await client.post(
                root_path + "/admin/mcp",
                headers={
                    **{
                        name: "Bearer member"
                        for name in SpecialHeaders.litellm_credential_header_names() - {"authorization"}
                    },
                    "Authorization": "Bearer " + credential,
                    "Accept": "application/json, text/event-stream",
                    "X-Forwarded-For": "203.0.113.8",
                    "Cookie": "session=must-not-forward",
                    "litellm-changed-by": "forged-actor",
                    "x-litellm-team-id": "policy-team",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_keys"}},
            )
            assert response.status_code == 200, response.text
            return response.json()

    async with management_app.router.lifespan_context(management_app):
        results: Final = await asyncio.gather(call("admin-a"), call("admin-b"))

    for credential, result in zip(("admin-a", "admin-b"), results):
        assert json.loads(result["result"]["content"][0]["text"]) == {
            "keys": ["owned-by-a" if credential == "admin-a" else "owned-by-b"],
            "client": "198.51.100.7",
            "scheme": "https",
            "forwarded_for": "203.0.113.8",
            "cookie": "",
            "caller_is_changed_by": True,
            "policy_team": "policy-team",
            "alternate_credentials_absent": True,
        }


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize("header_name", ["X-Custom-Key", "Authorization", "x-litellm-api-key"])
@pytest.mark.parametrize("caller_value", [None, "Bearer member"])
def test_configured_key_header_uses_mcp_bearer_after_settings_change(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, header_name: str, caller_value: str | None
) -> None:
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        monkeypatch.setitem(proxy_server.general_settings, "litellm_key_header_name", header_name)
        response: Final = client.post(
            "/admin/mcp",
            headers={
                **({header_name: caller_value} if caller_value else {}),
                "Authorization": "Bearer admin-a",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_keys"}},
        )
    assert response.status_code == 200, response.text
    assert json.loads(response.json()["result"]["content"][0]["text"])["keys"] == ["owned-by-a"]


@pytest.mark.parametrize("header_name", [
    "", "bad name", "x-ключ", 7, "Cookie", "litellm-changed-by",
    "Content-Type", "Host", "X-Forwarded-For", "x-litellm-team-id",
    *STANDARD_CUSTOMER_ID_HEADERS,
])
def test_configured_key_header_rejects_invalid_or_reserved_names(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, header_name: object
) -> None:
    monkeypatch.setitem(proxy_server.general_settings, "litellm_key_header_name", header_name)
    with pytest.raises(ValueError, match="litellm_key_header_name"):
        with TestClient(management_app):
            pytest.fail("An ambiguous or malformed credential header must fail startup")


@pytest.mark.parametrize("policy", [
    {"user_header_name": "X-Custom-Key"},
    {"user_header_mappings": {"header_name": "X-Custom-Key", "litellm_user_role": "customer"}},
    {"user_header_mappings": [{"header_name": "X-Custom-Key", "litellm_user_role": "internal_user"}]},
    {"enable_oauth2_proxy_auth": True, "oauth2_config_mappings": {"user_id": "X-Custom-Key"}},
    {"mcp_client_id_header": "X-Custom-Key"},
])
def test_configured_key_header_cannot_replace_configured_identity(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, policy: dict[str, object]
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {**policy, "litellm_key_header_name": "x-custom-key"})
    with pytest.raises(ValueError, match="litellm_key_header_name"):
        with TestClient(management_app):
            pytest.fail("A credential header must not replace a configured identity header")


@pytest.mark.parametrize("policy", [
    {"user_header_name": "x-api-key"},
    {"user_header_mappings": {"header_name": "Cookie", "litellm_user_role": "customer"}},
    {"enable_oauth2_proxy_auth": True, "oauth2_config_mappings": {"user_id": "x-litellm-api-key"}},
])
def test_policy_headers_cannot_share_overwritten_slots_without_a_custom_key(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, policy: dict[str, object]
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", policy)
    with pytest.raises(ValueError, match="configured identity headers"):
        with TestClient(management_app):
            pytest.fail("Native credentials must not overwrite configured policy headers")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_compact_results_bind_to_bearer_despite_conflicting_alternate_keys(management_app: FastAPI) -> None:
    headers: Final = {
        "Authorization": "Bearer admin-a", "x-litellm-api-key": "Bearer admin-b",
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        saved: Final = client.post(
            "/admin/mcp", headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "list_keys", "arguments": {"query": {"large": True}, "response": {"view": "compact"}},
            }},
        )
        assert saved.status_code == 200, saved.text
        result_id: Final = json.loads(saved.json()["result"]["content"][0]["text"])["result_id"]
        read_payload: Final = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "read_admin_result", "arguments": {"result_id": result_id, "view": "full"},
        }}
        replayed: Final = client.post(
            "/admin/mcp", headers={**headers, "x-litellm-api-key": "Bearer admin-b-limited"}, json=read_payload,
        )
        other_bearer: Final = client.post(
            "/admin/mcp", headers={**headers, "Authorization": "Bearer admin-b"}, json=read_payload,
        )
        limited: Final = client.post(
            "/admin/mcp",
            headers={"Authorization": "Bearer admin-b-limited", "Accept": headers["Accept"]},
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_keys"}},
        )
    assert replayed.status_code == 200, replayed.text
    assert json.loads(replayed.json()["result"]["content"][0]["text"]) == {"keys": [{"key_alias": "a" * 20000}]}
    assert other_bearer.status_code == 200, other_bearer.text
    assert other_bearer.json()["result"]["isError"] is True
    assert limited.status_code == 200, limited.text
    assert limited.json()["result"]["isError"] is True


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize("public_url", ["gateway.example.com", "https:/broken", "http://gateway.example.com"])
def test_invalid_public_origin_fails_startup(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, public_url: str
) -> None:
    monkeypatch.setenv("LITELLM_MCP_PUBLIC_URL", public_url)
    with pytest.raises(ValueError, match="HTTPS gateway origin"):
        with TestClient(management_app):
            pytest.fail("An invalid trusted public origin must fail startup")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_public_host_and_origin_are_checked(management_app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com")
    headers: Final = {"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"}
    with TestClient(management_app, base_url="https://gateway.example.com") as client:
        hostile_host: Final = client.post("/admin/mcp", headers={**headers, "Host": "hostile.example.com"}, json={})
        hostile_origin: Final = client.post(
            "/admin/mcp", headers={**headers, "Origin": "https://hostile.example.com"}, json={}
        )
    assert hostile_host.status_code == 421
    assert hostile_origin.status_code == 403


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize("read_only,alias", [(False, "created"), (False, "fail-after-write"), (True, "denied")])
def test_writes_respect_read_only_and_are_never_retried(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, read_only: bool, alias: str
) -> None:
    monkeypatch.setenv("LITELLM_ADMIN_TOOLS", "list_keys,create_key")
    monkeypatch.setenv("LITELLM_ADMIN_READ_ONLY", str(read_only).lower())
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        response: Final = client.post(
            "/admin/mcp",
            headers={"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "create_key", "arguments": {"body": {"key_alias": alias}}},
            },
        )
    assert response.status_code == 200, response.text
    assert management_app.state.created_aliases == ([] if read_only else [alias])
    assert response.json()["result"]["isError"] == (read_only or alias == "fail-after-write")
    if alias == "created":
        assert json.loads(response.json()["result"]["content"][0]["text"]) == {"key": "sk-new-key", "key_alias": alias}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_full_results_do_not_require_worker_affinity(management_app: FastAPI) -> None:
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        response: Final = client.post(
            "/admin/mcp",
            headers={"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_keys", "arguments": {"query": {"large": True}}},
            },
        )
    assert response.status_code == 200, response.text
    assert json.loads(response.json()["result"]["content"][0]["text"]) == {"keys": [{"key_alias": "a" * 20000}]}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_nested_management_calls_share_one_admission_slot(management_app: FastAPI) -> None:
    def no_metrics() -> None:
        return None

    def single_slot() -> AdmissionControlSettings:
        return AdmissionControlSettings(1, 0, 1.0)

    state: Final = AdmissionControlState(no_metrics)
    management_app.add_middleware(
        AdmissionControlMiddleware,
        get_settings=single_slot,
        state=state,
    )
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        response: Final = client.post(
            "/admin/mcp",
            headers={"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_keys"}},
        )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["isError"] is False
    assert json.loads(response.json()["result"]["content"][0]["text"])["keys"] == ["owned-by-a"]
    assert state.get_stats() == AdmissionControlStats(0, 0, 0)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
def test_oversized_mcp_body_is_rejected_before_a_write(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_ADMIN_TOOLS", "create_key")
    with TestClient(management_app, base_url="http://localhost:4000") as client:
        response: Final = client.post(
            "/admin/mcp",
            headers={"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "create_key", "arguments": {"body": {"key_alias": "x" * 300_000}}},
            },
        )
    assert response.status_code == 413, response.text
    assert management_app.state.created_aliases == []


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize("peer,mapped_user,bearer,status", [
    ("10.0.0.8", "admin-a", "member", 200),
    ("10.0.0.8", "member", "admin-a", 403),
    ("203.0.113.8", "admin-a", "admin-a", 401),
])
async def test_oauth2_proxy_auth_preserves_native_identity_and_peer_trust(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch,
    peer: str, mapped_user: str, bearer: str, status: int,
) -> None:
    users: Final = {
        user_id: LiteLLM_UserTable(user_id=user_id, user_role=role, teams=[])
        for user_id, role in (("admin-a", "proxy_admin"), ("member", "internal_user"))
    }

    class EmptyTable:
        async def find_many(self, **_: object) -> list[object]:
            return []

    class DatabaseTables:
        litellm_teamtable: Final = EmptyTable()
        litellm_teammembership: Final = EmptyTable()

    class UserInfoDatabase:
        db: Final = DatabaseTables()

        async def get_data(
            self, *, user_id: str | None = None, table_name: str | None = None, **_: object
        ) -> LiteLLM_UserTable | list[object] | None:
            return users.get(user_id) if user_id is not None and table_name is None else []

    cache: Final = UserApiKeyCache()
    for user_id, user in users.items():
        cache.set_cache(user_id, user)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", UserInfoDatabase())
    monkeypatch.setattr(proxy_server, "general_settings", {
        "enable_oauth2_proxy_auth": True,
        "trusted_proxy_ranges": ["10.0.0.0/24"],
        "oauth2_config_mappings": {"user_id": "X-Authenticated-User"},
    })
    management_app.router.routes[:] = [
        route for route in management_app.router.routes
        if not (isinstance(route, APIRoute) and route.path == "/user/info")
    ]
    management_app.add_api_route("/user/info", internal_user_endpoints.user_info, methods=["GET"])
    management_app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)

    async with management_app.router.lifespan_context(management_app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=management_app, client=(peer, 12345)),
            base_url="http://localhost:4000",
        ) as client:
            response: Final = await client.post(
                "/admin/mcp",
                headers={
                    "Authorization": "Bearer " + bearer,
                    "X-Authenticated-User": mapped_user,
                    "X-Forwarded-For": "10.0.0.8",
                    "Accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
    assert response.status_code == status, response.text
    if status == 200:
        assert "list_keys" in {tool["name"] for tool in response.json()["result"]["tools"]}
