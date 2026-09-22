import time
from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import JsonValue

from litellm.litellm_core_utils.secret_redaction import REDACTED
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.liteask import endpoints
from litellm.proxy.management_endpoints.liteask.approval import ApprovalError, seal_approval
from litellm.proxy.management_endpoints.liteask.catalog import OPERATIONS, Tool
from litellm.proxy.management_endpoints.liteask.dispatch import DispatchResult, credential_fingerprint
from tests.test_litellm.proxy.management_endpoints.liteask.conftest import AtomicApprovalStore

_CONVERSATION: Final = "ce52e0ef-e86e-4a7b-a884-1760a452d428"
_PREFIX: Final = "/management/v1/liteask"
_AUTHORIZATION: Final = "Bearer endpoint-test-credential"
_ROUTES: Final = (
    ("GET", "/config", None),
    ("POST", "/chat", {"conversation_id": _CONVERSATION, "messages": [{"role": "user", "content": "List teams"}]}),
    ("POST", "/approve", {"conversation_id": _CONVERSATION, "token": "invalid"}),
)


def _app(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> FastAPI:
    app: Final = FastAPI()
    app.include_router(endpoints.router)

    async def authenticate(request: Request) -> UserAPIKeyAuth:
        if request.headers.get("authorization") != _AUTHORIZATION:
            raise HTTPException(401, "Missing credentials")
        return UserAPIKeyAuth(user_id="admin", user_role=role)

    app.dependency_overrides[user_api_key_auth] = authenticate
    return app


@pytest.fixture
def admin_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    monkeypatch.setenv("LITELLM_LITEASK_MODEL", "configured-test-model")
    monkeypatch.setenv("LITELLM_SALT_KEY", "liteask-endpoint-test-only")
    live_admin: Final = LiteLLM_UserTable(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    with patch(
        "litellm.proxy.management_endpoints.liteask.auth.get_user_object",
        new=AsyncMock(return_value=live_admin),
    ):
        yield _app()


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), _ROUTES)
@pytest.mark.parametrize("role", (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER))
async def test_every_route_denies_non_admin_before_database_lookup(
    method: str, path: str, body: JsonValue, role: LitellmUserRoles
) -> None:
    lookup: Final = AsyncMock()
    with patch("litellm.proxy.management_endpoints.liteask.auth.get_user_object", new=lookup):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(role)), base_url="http://test") as client:
            response: Final = await client.request(
                method, _PREFIX + path, json=body, headers={"Authorization": _AUTHORIZATION}
            )
    assert response.status_code == 403, response.text
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), _ROUTES)
async def test_every_route_requires_authentication(method: str, path: str, body: JsonValue) -> None:
    lookup: Final = AsyncMock()
    with patch("litellm.proxy.management_endpoints.liteask.auth.get_user_object", new=lookup):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()), base_url="http://test") as client:
            response: Final = await client.request(method, _PREFIX + path, json=body)
    assert response.status_code == 401, response.text
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_config_reports_gateway_configuration_without_caching(admin_app: FastAPI) -> None:
    with patch("litellm.proxy.management_endpoints.liteask.endpoints._store", return_value=None):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app), base_url="http://test") as client:
            response: Final = await client.get(_PREFIX + "/config", headers={"Authorization": _AUTHORIZATION})
    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": True, "model": "configured-test-model", "can_execute_mutations": False}
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), _ROUTES[1:])
async def test_chat_and_approval_are_disabled_without_a_model(
    admin_app: FastAPI, monkeypatch: pytest.MonkeyPatch, method: str, path: str, body: JsonValue
) -> None:
    monkeypatch.delenv("LITELLM_LITEASK_MODEL")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app), base_url="http://test") as client:
        response: Final = await client.request(
            method, _PREFIX + path, json=body, headers={"Authorization": _AUTHORIZATION}
        )
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "LiteAsk is not enabled on this gateway."}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra", "status"),
    (({}, 400), ({"arguments": {"body": {"max_budget": 999}}}, 422)),
)
async def test_approval_rejects_malformed_tokens_and_replacement_arguments(
    admin_app: FastAPI, extra: dict[str, JsonValue], status: int
) -> None:
    send: Final = AsyncMock()
    with patch("litellm.proxy.management_endpoints.liteask.endpoints._send", new=send):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app), base_url="http://test") as client:
            response: Final = await client.post(
                _PREFIX + "/approve",
                json={"conversation_id": _CONVERSATION, "token": "malformed", **extra},
                headers={"Authorization": _AUTHORIZATION},
            )
    assert response.status_code == status, response.text
    send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ("key_create", "user_create"))
@pytest.mark.parametrize("generated_key", ("sk-test-generated-credential-12345678901234567890", "a" * 64))
async def test_new_credentials_only_appear_in_the_dedicated_generated_key_field(
    admin_app: FastAPI, approval_store: AtomicApprovalStore, operation_name: str, generated_key: str
) -> None:
    operation: Final = next(item for item in OPERATIONS if item.name == operation_name)
    tool: Final = Tool(
        operation,
        {"type": "object", "properties": {"body": {"type": "object"}}, "required": ["body"]},
    )
    fingerprint: Final = credential_fingerprint(
        Request({"type": "http", "headers": [(b"authorization", _AUTHORIZATION.encode())]})
    )
    assert fingerprint is not None
    approval: Final = await seal_approval(
        user_id="admin",
        credential=fingerprint,
        conversation_id=_CONVERSATION,
        tool=operation_name,
        arguments={"body": {}},
        now=int(time.time()),
        store=approval_store,
    )
    assert not isinstance(approval, ApprovalError)
    send: Final = AsyncMock(
        return_value=DispatchResult(
            200, {"key": generated_key, "message": f"Created {generated_key}", "key_alias": "demo"}
        )
    )
    with (
        patch("litellm.proxy.management_endpoints.liteask.endpoints._catalog", return_value=(tool,)),
        patch("litellm.proxy.management_endpoints.liteask.endpoints._store", return_value=approval_store),
        patch("litellm.proxy.management_endpoints.liteask.endpoints._send", new=send),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin_app), base_url="http://test") as client:
            response: Final = await client.post(
                _PREFIX + "/approve",
                json={"conversation_id": _CONVERSATION, "token": approval[0]},
                headers={"Authorization": _AUTHORIZATION},
            )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "message": f"Completed: {operation.title}.",
        "proposal": None,
        "result": {"key": REDACTED, "message": f"Created {REDACTED}", "key_alias": "demo"},
        "generated_key": generated_key,
    }
    assert response.text.count(generated_key) == 1
    assert response.headers["cache-control"] == "no-store"
    assert approval_store.records == {}
    send.assert_awaited_once()
