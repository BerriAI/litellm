from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.requests import Request

from litellm.proxy.mcp_server import discoverable_endpoints as de


def _make_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/sunny_server/token",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("litellm.local", 4000),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_exchange_token_falls_back_to_caller_client_when_shared_client_rejected():
    calls: list[str | None] = []

    class FakeClient:
        async def post(self, url, headers=None, data=None):
            cid = (data or {}).get("client_id")
            calls.append(cid)
            req = httpx.Request("POST", url)
            if cid == "shared-client":
                return httpx.Response(
                    400,
                    json={"error": "invalid_grant"},
                    request=req,
                )
            return httpx.Response(
                200,
                json={"access_token": "caller-access-token", "token_type": "bearer"},
                request=req,
            )

    fake_server = SimpleNamespace(
        server_id="srv-1",
        name="sunny_server",
        auth_type="oauth2",
        client_id="shared-client",
        client_secret="shared-secret",
        token_endpoint_auth_method="client_secret_post",
        dcr_issuer=None,
        dcr_server_url=None,
        issuer="https://auth.example.com",
        url="https://mcp.example.com",
        effective_token_url="https://auth.example.com/token",
        needs_user_oauth_token=False,
        oauth_identity_binding=None,
        is_oauth_delegate=False,
        is_dcr_bridge=True,
        token_validation=None,
    )

    with (
        patch.object(de, "raise_if_not_oauth2"),
        patch.object(de, "_server_with_oauth_endpoints", new=AsyncMock(return_value=fake_server)),
        patch.object(de, "oauth_client_registration_matches", return_value=True),
        patch.object(de, "_open_managed_authorization_code", return_value=None),
        patch.object(de, "get_cimd_client_id", return_value=None),
        patch.object(de, "_hydrate_bridge_shared_client", new=AsyncMock()),
        patch.object(de, "_dcr_bridge_relays_client_registration", return_value=True),
        patch.object(de, "get_async_httpx_client", return_value=FakeClient()),
    ):
        result = await de.exchange_token_with_server(
            request=_make_request(),
            mcp_server=fake_server,  # type: ignore[arg-type]
            grant_type="authorization_code",
            code="code-1",
            redirect_uri="https://caller.example/cb",
            client_id="caller-client",
            client_secret="caller-secret",
            code_verifier=None,
        )

    assert calls == ["shared-client", "caller-client"]
    assert result["access_token"] == "caller-access-token"
