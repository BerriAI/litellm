"""MCPJWTSigner must not replace an MCP server's own Authorization credential.

Regression tests for BerriAI/litellm#47988: a server configured with bearer_token / token /
authorization / basic plus authentication_token lost its key to the signer's JWT on both
tools/list and tools/call.
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _server_credential_occupies_authorization,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

SIGNER_SINGLETON = "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer.get_mcp_jwt_signer"
SIGNER_INJECT = (
    "litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer.inject_mcp_jwt_headers_for_upstream"
)

AUTHORIZATION_AUTH_TYPES = [MCPAuth.bearer_token, MCPAuth.token, MCPAuth.authorization, MCPAuth.basic]


def _server(
    auth_type: Optional[MCPAuth],
    authentication_token: Optional[str] = "sk-upstream-key",
    upstream_token_header: Optional[str] = None,
) -> MCPServer:
    return MCPServer(
        server_id="srv",
        name="srv",
        server_name="srv",
        url="https://upstream.example/mcp",
        transport=MCPTransport.http,
        auth_type=auth_type,
        authentication_token=authentication_token,
        upstream_token_header=upstream_token_header,
    )


@pytest.mark.parametrize(
    ("auth_type", "token", "header", "expected"),
    [
        pytest.param(MCPAuth.bearer_token, "k", None, True, id="bearer-default-slot"),
        pytest.param(MCPAuth.token, "k", None, True, id="token-default-slot"),
        pytest.param(MCPAuth.authorization, "k", None, True, id="authorization-default-slot"),
        pytest.param(MCPAuth.basic, "user:pass", None, True, id="basic-default-slot"),
        pytest.param(MCPAuth.api_key, "k", None, False, id="api-key-uses-x-api-key"),
        pytest.param(MCPAuth.api_key, "k", "Authorization", True, id="api-key-moved-to-authorization"),
        pytest.param(MCPAuth.bearer_token, "k", "X-Upstream-Key", False, id="bearer-moved-off-authorization"),
        pytest.param(MCPAuth.bearer_token, "k", "authorization", True, id="header-name-case-insensitive"),
        pytest.param(MCPAuth.bearer_token, None, None, False, id="no-credential-configured"),
        pytest.param(MCPAuth.none, "k", None, False, id="auth-none"),
        pytest.param(None, "k", None, False, id="auth-unset"),
    ],
)
def test_server_credential_occupies_authorization(auth_type, token, header, expected):
    assert _server_credential_occupies_authorization(_server(auth_type, token, header)) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", AUTHORIZATION_AUTH_TYPES)
async def test_tools_list_skips_signer_when_server_key_rides_authorization(  # test-quality-ok: signer is a process-wide singleton with no injection seam; the hook call is the observable
    auth_type,
):
    manager = MCPServerManager()
    manager.create_mcp_client = AsyncMock(return_value=object())
    manager._fetch_tools_with_timeout = AsyncMock(return_value=[])
    user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")

    with (
        patch(  # test-quality-ok: the signer is a process-wide singleton the manager reads, no injection seam
            SIGNER_SINGLETON, return_value=MagicMock()
        ),
        patch(  # test-quality-ok: same singleton's header injection, asserted on by call
            SIGNER_INJECT, new=AsyncMock(return_value={"Authorization": "Bearer signed-jwt"})
        ) as mock_inject,
    ):
        await manager._get_tools_from_server(_server(auth_type), user_api_key_auth=user_auth)

    mock_inject.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        pytest.param(_server(MCPAuth.api_key), id="api-key-rides-x-api-key"),
        pytest.param(_server(MCPAuth.bearer_token, authentication_token=None), id="no-server-credential"),
        pytest.param(
            _server(MCPAuth.bearer_token, upstream_token_header="X-Upstream-Key"),
            id="credential-moved-off-authorization",
        ),
    ],
)
async def test_tools_list_still_runs_signer_when_authorization_is_free(  # test-quality-ok: signer is a process-wide singleton with no injection seam; the hook call is the observable
    server,
):
    manager = MCPServerManager()
    manager.create_mcp_client = AsyncMock(return_value=object())
    manager._fetch_tools_with_timeout = AsyncMock(return_value=[])
    user_auth = UserAPIKeyAuth(api_key="sk-test", user_id="alice")

    with (
        patch(  # test-quality-ok: the signer is a process-wide singleton the manager reads, no injection seam
            SIGNER_SINGLETON, return_value=MagicMock()
        ),
        patch(  # test-quality-ok: same singleton's header injection, asserted on by call
            SIGNER_INJECT, new=AsyncMock(return_value={"Authorization": "Bearer signed-jwt"})
        ) as mock_inject,
    ):
        await manager._get_tools_from_server(server, user_api_key_auth=user_auth)

    mock_inject.assert_awaited_once()


async def _call_tool_and_capture_extra_headers(server: MCPServer, hook_headers: Dict[str, str]) -> Dict[str, str]:
    manager = MCPServerManager()
    captured: Dict[str, Any] = {}

    async def fake_create_mcp_client(server, mcp_auth_header=None, extra_headers=None, stdio_env=None, **kwargs):
        captured["extra_headers"] = extra_headers
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value=MagicMock())
        return mock_client

    with patch.object(manager, "create_mcp_client", side_effect=fake_create_mcp_client):
        with patch.object(manager, "build_stdio_env", return_value=None):
            try:
                await manager._call_regular_mcp_tool(
                    mcp_server=server,
                    original_tool_name="test_tool",
                    arguments={"key": "val"},
                    tasks=[],
                    mcp_auth_header=None,
                    mcp_server_auth_headers=None,
                    oauth2_headers=None,
                    raw_headers=None,
                    proxy_logging_obj=None,
                    hook_extra_headers=hook_headers,
                )
            except Exception:
                pass  # the mocked client's result is not what is under test

    assert "extra_headers" in captured, "create_mcp_client was never reached"
    return captured["extra_headers"] or {}


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", AUTHORIZATION_AUTH_TYPES)
async def test_tools_call_drops_hook_authorization_when_server_key_rides_authorization(auth_type):
    headers = await _call_tool_and_capture_extra_headers(
        _server(auth_type), {"Authorization": "Bearer hook-jwt", "X-Trace-Id": "trace-123"}
    )

    assert "Authorization" not in headers
    assert headers.get("X-Trace-Id") == "trace-123"


@pytest.mark.asyncio
async def test_tools_call_keeps_hook_authorization_when_server_key_rides_x_api_key():
    headers = await _call_tool_and_capture_extra_headers(_server(MCPAuth.api_key), {"Authorization": "Bearer hook-jwt"})

    assert headers.get("Authorization") == "Bearer hook-jwt"


def test_byok_server_stored_secret_never_occupies_authorization():
    server = _server(MCPAuth.bearer_token).model_copy(update={"is_byok": True})

    assert _server_credential_occupies_authorization(server) is False
