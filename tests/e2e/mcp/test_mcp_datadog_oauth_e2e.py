"""Live e2e: the Datadog MCP server through gateway-managed OAuth2 (PKCE).

Registers the Datadog MCP server with auth_type=oauth2,
oauth2_flow=authorization_code. The gateway discovers the OAuth endpoints via
/.well-known metadata. A real PKCE authorize dance (DCR, browser consent,
token exchange) produces an access token, which is stored in the gateway's
per-user credential vault. Then the key lists tools, calls one, and drives a
chat completion through the MCP bridge, all using the stored per-user token.

Requires E2E_DD_STORAGE_STATE pointing at a saved Datadog browser session.
"""

from __future__ import annotations

import os

import pytest

from dd_oauth import (
    assert_dd_oauth_env,
    delete_dd_oauth_server,
    fetch_dd_oauth_token,
    register_dd_oauth_server,
    store_dd_oauth_token,
)
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import ChatBody, ChatMessage, KeyGenerateBody, McpChatTool, ObjectPermission

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("E2E_DD_STORAGE_STATE"),
        reason="set E2E_DD_STORAGE_STATE to a Datadog session captured via mcp/dd_session_capture.py",
    ),
]

pytest.importorskip("mcp", reason="mcp SDK not installed")
pytest.importorskip("playwright.async_api", reason="playwright not installed")

SEARCH_LOGS_TOOL = "search_datadog_logs"


class TestDatadogMcpOAuth:
    @pytest.mark.covers(
        "mcp.list_tools.oauth.succeeds",
        "mcp.call_tool.oauth.succeeds",
    )
    def test_oauth_list_and_call_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_oauth_env()

        marker = unique_marker()
        alias = f"e2e_dd_oauth_{marker}"
        dd = register_dd_oauth_server(client.proxy, alias)
        resources.defer(lambda: delete_dd_oauth_server(client.proxy, dd.server_id))
        client.await_registered(dd.server_id)

        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[CHEAP_ANTHROPIC_MODEL],
                user_id=f"e2e-dd-oauth-{marker}",
                object_permission=ObjectPermission(mcp_servers=[dd.server_id]),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        token = fetch_dd_oauth_token(os.environ["E2E_DD_STORAGE_STATE"])
        store_dd_oauth_token(client.proxy, dd.server_id, key, token)

        tool_name = client.await_tool(key, dd.server_id, SEARCH_LOGS_TOOL)
        assert tool_name, f"OAuth token did not surface any tools for server {dd.server_id}"

        call = client.await_call_tool(
            key,
            server_id=dd.server_id,
            name=tool_name,
            arguments={
                "query": "service:litellm",
                "from": "now-30m",
                "to": "now",
                "max_tokens": 500,
            },
        )
        assert call.is_error is not True, f"search_datadog_logs errored via OAuth: {call}"
        assert call.all_text, f"OAuth tool call returned empty text: {call}"

    @pytest.mark.covers("mcp.chat_completion.oauth.auto_executes_tools")
    def test_oauth_chat_completion_auto_executes_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_oauth_env()

        marker = unique_marker()
        alias = f"e2e_dd_oauth_chat_{marker}"
        dd = register_dd_oauth_server(client.proxy, alias)
        resources.defer(lambda: delete_dd_oauth_server(client.proxy, dd.server_id))
        client.await_registered(dd.server_id)

        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[CHEAP_ANTHROPIC_MODEL],
                user_id=f"e2e-dd-oauth-chat-{marker}",
                object_permission=ObjectPermission(mcp_servers=[dd.server_id]),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        token = fetch_dd_oauth_token(os.environ["E2E_DD_STORAGE_STATE"])
        store_dd_oauth_token(client.proxy, dd.server_id, key, token)

        response = unwrap(
            client.chat_with_mcp(
                key,
                ChatBody(
                    model=CHEAP_ANTHROPIC_MODEL,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                "Use the search_datadog_logs tool to search for logs "
                                f"with query 'e2e-mcp-oauth-nohit-{marker}' from now-30m to now with "
                                "max_tokens 100. After you get results, reply with ok only."
                            ),
                        )
                    ],
                    max_tokens=1024,
                    tools=[
                        McpChatTool(
                            type="mcp",
                            server_url=f"litellm_proxy/mcp/{dd.alias}",
                            server_label="datadog",
                            require_approval="never",
                        )
                    ],
                ),
            )
        )

        assert response.choices, f"chat completion returned no choices: {response}"
        message = response.choices[0].message
        assert message is not None, f"choice had no message: {response}"
        psf = message.provider_specific_fields
        assert psf is not None, (
            f"provider_specific_fields missing; the gateway did not attach MCP metadata: {message}"
        )
        assert psf.mcp_list_tools, (
            f"mcp_list_tools is empty; the gateway never listed tools via the stored OAuth token: {psf}"
        )
        assert psf.mcp_call_results, (
            f"mcp_call_results is empty; the gateway did not execute any tool via OAuth: {psf}"
        )
