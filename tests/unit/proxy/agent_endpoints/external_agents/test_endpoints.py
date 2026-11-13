from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_agents.database import Database
from litellm.proxy.agent_endpoints.external_agents.endpoints import admin, database, router
from litellm.proxy.agent_endpoints.external_agents.models import Source
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper


def endpoint_client(
    rows: tuple[object, ...], role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN
) -> httpx.AsyncClient:
    db: Final = MagicMock(spec=Database)
    db.query_raw = AsyncMock(side_effect=rows)
    db.execute_raw = AsyncMock(return_value=1)
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[database] = lambda: db
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="admin", user_role=role)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_management_rejects_non_admins_including_read_only_admin(role: LitellmUserRoles) -> None:
    with pytest.raises(HTTPException) as error:
        admin(UserAPIKeyAuth(user_role=role))
    assert error.value.status_code == 403


def test_proxy_admin_can_manage_external_agents() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert admin(auth) is auth


def test_source_response_never_serializes_current_or_previous_secret() -> None:
    source: Final = Source(
        source_id="source",
        display_name="Bots",
        signing_secret="encrypted-current",
        previous_signing_secret="encrypted-previous",
        ingest_key="encrypted-ingest",
        admin_api_key="encrypted-admin",
    )
    assert source.model_dump(mode="json") == {
        "source_id": "source",
        "display_name": "Bots",
        "vendor": "grok_bot",
        "tenant_id": None,
        "mode": "shadow",
        "default_policy_id": None,
        "last_event_at": None,
    }


@pytest.mark.asyncio
async def test_create_without_body_returns_setup_credentials_but_list_redacts_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "external-agent-endpoint-test-key")
    row: Final = Source(source_id="new-source", display_name="GrokBot", default_policy_id="default").model_dump()
    stored: Final = row | {
        "ingest_key": "encrypted-ingest",
        "signing_secret": "encrypted-current",
        "previous_signing_secret": "encrypted-previous",
        "admin_api_key": "encrypted-admin",
    }
    async with endpoint_client(([stored], [stored])) as client:
        created: Final = await client.post("/v1/external_agents/sources")
        listed: Final = await client.get("/v1/external_agents/sources")
    assert created.status_code == listed.status_code == 200
    assert created.json()["ingest_key"].startswith("litellm_ingest_")
    assert created.json() | {"ingest_key": "redacted"} == {
        **row,
        "ingest_key": "redacted",
        "hook_path": "/grok_bot/inference_hook/new-source",
        "sync_result": None,
    }
    assert listed.json() == [row]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
async def test_revealing_connection_requires_write_admin_access(
    role: LitellmUserRoles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "external-agent-endpoint-test-key")
    source: Final = Source(source_id="connection", display_name="Bots", ingest_key=encrypt_value_helper("test-ingest"))
    async with endpoint_client(([source.model_dump() | {"ingest_key": source.ingest_key}],), role) as client:
        response: Final = await client.get("/v1/external_agents/sources/connection/connection")
    if role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
        assert response.status_code == 403
        assert response.json() == {"detail": "Proxy admin access required"}
        return
    assert response.status_code == 200
    assert response.json() == {
        **source.model_dump(),
        "ingest_key": "test-ingest",
        "hook_path": "/grok_bot/inference_hook/connection",
        "sync_result": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("connection", "sync_members"))
async def test_connection_actions_report_unknown_sources(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "external-agent-endpoint-test-key")
    async with endpoint_client(([],)) as client:
        response: Final = await client.request(
            "GET" if path == "connection" else "POST", f"/v1/external_agents/sources/missing/{path}"
        )
    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown source"}


@pytest.mark.asyncio
async def test_sync_without_admin_key_returns_actionable_error() -> None:
    row: Final = Source(source_id="source", display_name="Bots").model_dump()
    async with endpoint_client(([row],)) as client:
        response: Final = await client.post("/v1/external_agents/sources/source/sync_members")
    assert response.status_code == 400
    assert response.json() == {"detail": "Save a GrokBot Admin API key before syncing members"}


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", (None, "Bearer wrong-key", "Bearer test-ingest"))
async def test_config_test_requires_bearer_and_returns_vendor_verdict(
    authorization: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "external-agent-endpoint-test-key")
    source_id: Final = "hook-" + (authorization or "missing").replace(" ", "-")
    row: Final = Source(source_id=source_id, display_name="Bots").model_dump() | {
        "ingest_key": encrypt_value_helper("test-ingest")
    }
    async with endpoint_client(([row], [])) as client:
        response: Final = await client.post(
            f"/grok_bot/inference_hook/{source_id}",
            headers={"webhook-id": "config-test"} | ({"Authorization": authorization} if authorization else {}),
            json={
                "type": "prompt",
                "request_id": "config-test",
                "tenant_id": "tenant",
                "actor": {"type": "user", "id": None},
                "session_id": "test-bot",
                "source": {"application": "config-test"},
            },
        )
    if authorization != "Bearer test-ingest":
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid ingest key"}
        return
    assert response.status_code == 200
    assert response.json() == {"action": "allow"}
