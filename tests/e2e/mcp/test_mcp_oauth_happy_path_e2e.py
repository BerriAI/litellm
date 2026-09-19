"""Live e2e coverage for the gateway-managed MCP OAuth protocol path.

The test creates a JWT-authorized user, completes real Linear authorization
consent, lists and calls a tool immediately through the per-server MCP route,
and verifies the canonical per-user credential row. It then uses a fresh SDK
client against one gateway URL or a configured replica URL. With one gateway
URL, that second run proves fresh-client reuse only. With replica URLs, it
proves that a process which did not run consent resolves the stored token.
"""

from __future__ import annotations

import os
from typing import Final

import pytest
from e2e_config import (
    LINEAR_MCP_URL,
    LINEAR_READONLY_TOOL,
    LINEAR_STORAGE_STATE,
    PROXY_BASE_URL,
    PROXY_REPLICA_URLS,
    unique_marker,
)
from e2e_http import AuthHeaders
from lifecycle import ResourceManager
from models import McpServerCreateBody, ObjectPermission, TeamUpdateBody
from proxy_client import ProxyClient

pytest.importorskip("mcp", reason="mcp SDK not installed; run `uv sync --inexact --group e2e-dev`")
pytest.importorskip(
    "playwright.async_api",
    reason="playwright not installed; run `uv pip install playwright` and `playwright install chromium`",
)

from idp import Identity, Keycloak  # noqa: E402
from oauth_chat_client import ChatMcpClient, InMemoryTokenStorage, build_chat_client  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.mcp_oauth_live]


@pytest.fixture(scope="session")
def chat_client(proxy: ProxyClient) -> ChatMcpClient:
    return build_chat_client(proxy)


class TestMcpOauthHappyPath:
    @pytest.mark.covers("mcp.list_tools.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.persists_across_processes")
    def test_jwt_user_lists_and_calls_then_reconnects_from_another_gateway(
        self,
        chat_client: ChatMcpClient,
        resources: ResourceManager,
        jwt_identity: Identity,
        idp: Keycloak,
    ) -> None:
        assert LINEAR_STORAGE_STATE and os.path.exists(LINEAR_STORAGE_STATE), (
            "E2E_MCP_OAUTH_LIVE is set but E2E_LINEAR_STORAGE_STATE does not point at a captured "
            "Linear session (run mcp/linear_session_capture.py)"
        )

        alias: Final = f"e2elinear{unique_marker()}"
        created: Final = chat_client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=LINEAR_MCP_URL,
                allow_all_keys=False,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
                per_server_oauth_discovery=True,
            )
        )
        resources.defer(lambda: chat_client.delete_server(created.server_id))

        chat_client.proxy.update_team(
            TeamUpdateBody(
                team_id=jwt_identity.group,
                object_permission=ObjectPermission(mcp_servers=[created.server_id]),
            )
        )

        token: Final = idp.access_token(jwt_identity)
        headers: Final = {"x-litellm-api-key": f"Bearer {token}"}
        storage: Final = InMemoryTokenStorage()
        first_run: Final = chat_client.list_and_call(
            alias,
            headers,
            storage,
            LINEAR_STORAGE_STATE,
            LINEAR_READONLY_TOOL,
            {},
        )
        assert f"{alias}-{LINEAR_READONLY_TOOL}" in first_run.tools
        assert first_run.is_error is False
        assert first_run.text.strip() != ""

        credentials: Final = chat_client.server_user_credentials(created.server_id)
        assert len(credentials) == 1
        assert credentials[0].user_id == jwt_identity.user_id
        assert credentials[0].credential_type == "oauth2"
        resources.defer(
            lambda: chat_client.revoke_user_token(
                created.server_id,
                AuthHeaders.model_validate(headers),
            )
        )

        replica: Final = PROXY_REPLICA_URLS[-1] if len(PROXY_REPLICA_URLS) > 1 else PROXY_BASE_URL
        second_run: Final = chat_client.list_and_call(
            alias,
            {"x-litellm-api-key": f"Bearer {idp.access_token(jwt_identity)}"},
            InMemoryTokenStorage(),
            None,
            LINEAR_READONLY_TOOL,
            {},
            base_url=replica,
        )
        assert f"{alias}-{LINEAR_READONLY_TOOL}" in second_run.tools
        assert second_run.is_error is False
        assert second_run.text.strip() != ""
