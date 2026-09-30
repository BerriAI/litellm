import asyncio
import json
import sys
from typing import Final

import httpx2
import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.testclient import TestClient

from litellm.proxy.admin_mcp import admin_mcp_lifespan
from litellm.proxy.middleware.admission_control_middleware import (
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
    AdmissionControlStats,
)
from litellm.proxy.middleware.in_flight_requests_middleware import InFlightRequestsMiddleware
from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager


class KeyRequest(BaseModel):
    key_alias: str


@pytest.fixture
def management_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", "true")
    monkeypatch.setenv("LITELLM_ADMIN_TOOLS", "list_keys")
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_MCP_PUBLIC_URL", raising=False)
    app: Final = FastAPI(lifespan=admin_mcp_lifespan)

    @app.get("/user/info")
    async def user_info(request: Request) -> dict[str, object]:
        authorization: Final = request.headers.get("authorization", "")
        if authorization not in {"Bearer admin-a", "Bearer admin-b", "Bearer member"}:
            raise HTTPException(status_code=401)
        user_id: Final = authorization.removeprefix("Bearer ")
        return {
            "user_id": user_id,
            "user_info": {
                "user_id": user_id,
                "user_role": "internal_user" if user_id == "member" else "proxy_admin",
            },
        }

    @app.get("/key/list", operation_id="list_keys_key_list_get")
    async def list_keys(request: Request, large: bool = False) -> dict[str, object]:
        if large:
            return {"keys": [{"key_alias": "a" * 20000}]}
        return {
            "keys": ["owned-by-a" if request.headers["authorization"] == "Bearer admin-a" else "owned-by-b"],
            "client": request.client.host if request.client else None,
            "scheme": request.url.scheme,
            "forwarded_for": request.headers.get("x-forwarded-for"),
            "cookie": request.headers.get("cookie"),
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


@pytest.mark.parametrize("enabled", [None, "false", "0"])
def test_disabled_preserves_existing_admin_namespace(monkeypatch: pytest.MonkeyPatch, enabled: str | None) -> None:
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


def test_invalid_flag_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_ADMIN_MCP", "treu")
    with pytest.raises(ValueError, match="LITELLM_ENABLE_ADMIN_MCP"):
        with TestClient(FastAPI(lifespan=admin_mcp_lifespan)):
            pytest.fail("Invalid flag must fail startup")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Admin MCP requires Python 3.12+")
@pytest.mark.parametrize("authorization,status", [(None, 401), ("Bearer invalid", 401), ("Bearer member", 403)])
def test_admin_endpoint_rejects_unauthorized_callers(
    management_app: FastAPI, authorization: str | None, status: int
) -> None:
    headers: Final = {"Authorization": authorization} if authorization else {}
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
async def test_concurrent_calls_preserve_identity_network_context_and_strip_cookies(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch, root_path: str
) -> None:
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example.com" + root_path)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://must-not-call.example.com")
    monkeypatch.setenv("LITELLM_API_KEY", "must-not-use-shared-credential")

    async def call(credential: str) -> dict[str, object]:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=management_app, root_path=root_path, client=("198.51.100.7", 4567)),
            base_url="https://gateway.example.com",
        ) as client:
            response: Final = await client.post(
                root_path + "/admin/mcp",
                headers={
                    "Authorization": "Bearer " + credential,
                    "Accept": "application/json, text/event-stream",
                    "X-Forwarded-For": "203.0.113.8",
                    "Cookie": "session=must-not-forward",
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
        }


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
    state: Final = AdmissionControlState(lambda: None)
    management_app.add_middleware(
        AdmissionControlMiddleware,
        get_settings=lambda: AdmissionControlSettings(1, 0, 1.0),
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
async def test_shutdown_drains_an_active_tool_before_closing_connector(
    management_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_ADMIN_TOOLS", "list_teams")
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    management_app.add_middleware(InFlightRequestsMiddleware)

    @management_app.get("/team/list", operation_id="list_team_team_list_get")
    async def list_teams() -> dict[str, object]:
        started.set()
        await release.wait()
        return {"teams": ["completed-before-shutdown"]}

    async def complete_during_drain() -> None:
        try:
            async with asyncio.timeout(5):
                while not GracefulShutdownManager.is_shutting_down():
                    await asyncio.sleep(0)
        finally:
            release.set()

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=management_app), base_url="http://localhost:4000"
    ) as client:
        async with management_app.router.lifespan_context(management_app):
            request: Final = asyncio.create_task(
                client.post(
                    "/admin/mcp",
                    headers={"Authorization": "Bearer admin-a", "Accept": "application/json, text/event-stream"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_teams"}},
                )
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            completion: Final = asyncio.create_task(complete_during_drain())
        await asyncio.wait_for(completion, timeout=5)
        response: Final = await asyncio.wait_for(request, timeout=5)

    assert response.status_code == 200, response.text
    assert response.json()["result"]["isError"] is False
    assert json.loads(response.json()["result"]["content"][0]["text"]) == {"teams": ["completed-before-shutdown"]}
    assert InFlightRequestsMiddleware.get_count() == 0
