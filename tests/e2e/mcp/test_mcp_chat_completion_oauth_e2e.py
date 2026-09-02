"""On-demand e2e: a chat completion drives a gateway-managed OAuth MCP server.

The real end-user flow for MCP over an OAuth server: a user registers a Linear
authorization_code server, authorizes it once so the gateway stores their
upstream token, then sends a normal /chat/completions request with the Linear
MCP attached. The gateway resolves the user from the LiteLLM key, lists Linear's
tools with the stored per-user token, lets the model call one, executes it
upstream with that token, and returns the answer. This is proven against the
real Linear MCP server (mcp.linear.app) and a real Anthropic model, once per
documented ingress header (x-litellm-api-key and Authorization).

The authorize dance is seeded through the mcp SDK's OAuthClientProvider; the one
step Linear cannot auto-approve is the human consent, so it is captured once out
of band (mcp/linear_session_capture.py) into a saved browser session and a
headless Chromium clicks Approve every run. The test therefore skips unless
E2E_LINEAR_STORAGE_STATE points at that session, so it never runs on the per-PR
CI path; it is a nightly/on-demand real-server smoke test.

Fail-before-fix: without the stored per-user token the gateway lists no Linear
tools, so mcp_list_tools comes back empty, nothing is called, and the
assertions fail; a served, called, non-empty Linear tool proves the gateway
pulled and used the user's token.
"""

from __future__ import annotations

import os

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL, LINEAR_MCP_URL, LINEAR_STORAGE_STATE, unique_marker
from e2e_http import AuthHeaders
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, KeyGenerateBody, McpChatTool, McpServerCreateBody, ObjectPermission
from proxy_client import ProxyClient

pytest.importorskip("mcp", reason="mcp SDK not installed; run `uv sync --inexact --group e2e-dev`")
pytest.importorskip(
    "playwright.async_api",
    reason="playwright not installed; run `uv pip install playwright` and `playwright install chromium`",
)

from oauth_chat_client import ChatMcpClient, build_chat_client  # noqa: E402  # imports follow the importorskip guards

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not LINEAR_STORAGE_STATE or not os.path.exists(LINEAR_STORAGE_STATE),
        reason="set E2E_LINEAR_STORAGE_STATE to a Linear session captured via mcp/linear_session_capture.py",
    ),
]

# Pinned from a live dance during verification (never guessed); the gateway
# prefixes every upstream tool name with the server alias. list_teams is a
# read-only Linear tool that takes no arguments and returns the caller's teams.
LINEAR_READONLY_TOOL = "list_teams"
LINEAR_PROMPT = "Use the list_teams tool to list my Linear teams, then reply with the name of one of them."


@pytest.fixture(scope="session")
def chat_client(proxy: ProxyClient) -> ChatMcpClient:
    return build_chat_client(proxy)


class TestMcpChatCompletionOauth:
    """A scoped internal-user key on a real Linear authorization_code server,
    used through /chat/completions once per ingress header: the gateway pulls
    the user's stored upstream token, lists and executes Linear's tools during
    the completion, and returns the answer."""

    @pytest.mark.covers("mcp.list_tools.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.succeeds")
    def test_chat_completion_uses_linear_with_x_litellm_api_key_header(
        self, chat_client: ChatMcpClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        alias = f"e2elinear{marker}"
        created = chat_client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=LINEAR_MCP_URL,
                allow_all_keys=False,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
            )
        )
        resources.defer(lambda: chat_client.delete_server(created.server_id))

        stored = chat_client.server_info(created.server_id)
        assert stored.auth_type == "oauth2"
        assert stored.oauth2_flow == "authorization_code"
        assert stored.allow_all_keys is False

        key = chat_client.proxy.generate_key(
            KeyGenerateBody(
                user_id="e2e-test-user",
                object_permission=ObjectPermission(mcp_servers=[created.server_id]),
            )
        )
        resources.defer(lambda: chat_client.proxy.delete_key(key))

        seeded = chat_client.seed_user_token(alias, key, LINEAR_STORAGE_STATE)
        assert f"{alias}-{LINEAR_READONLY_TOOL}" in seeded, (
            f"the authorize dance listed {seeded}, expected it to include {alias}-{LINEAR_READONLY_TOOL}"
        )

        response = chat_client.chat_with_mcp(
            AuthHeaders.model_validate({"x-litellm-api-key": f"Bearer {key}"}),
            ChatBody(
                model=CHEAP_ANTHROPIC_MODEL,
                messages=[ChatMessage(role="user", content=LINEAR_PROMPT)],
                tools=[
                    McpChatTool(
                        server_url=f"litellm_proxy/mcp/{alias}",
                        server_label=alias,
                        require_approval="never",
                    )
                ],
            ),
        )

        message = response.choices[0].message
        assert message is not None and message.content, f"completion returned no answer: {response}"
        meta = message.provider_specific_fields
        assert meta is not None, f"no MCP metadata on the completion: {response}"
        listed = {t.function.name for t in (meta.mcp_list_tools or []) if t.function}
        assert f"{alias}-{LINEAR_READONLY_TOOL}" in listed, (
            f"the gateway listed {sorted(listed)}, expected the stored token to surface {alias}-{LINEAR_READONLY_TOOL}"
        )
        results = [r for r in (meta.mcp_call_results or []) if r.name == f"{alias}-{LINEAR_READONLY_TOOL}"]
        assert results and results[0].result, (
            f"Linear tool {alias}-{LINEAR_READONLY_TOOL} was not executed with a result: {meta.mcp_call_results}"
        )

    @pytest.mark.covers("mcp.list_tools.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.succeeds")
    def test_chat_completion_uses_linear_with_authorization_bearer_header(
        self, chat_client: ChatMcpClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        alias = f"e2elinear{marker}"
        created = chat_client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=LINEAR_MCP_URL,
                allow_all_keys=False,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
            )
        )
        resources.defer(lambda: chat_client.delete_server(created.server_id))

        stored = chat_client.server_info(created.server_id)
        assert stored.auth_type == "oauth2"
        assert stored.oauth2_flow == "authorization_code"
        assert stored.allow_all_keys is False

        key = chat_client.proxy.generate_key(
            KeyGenerateBody(
                user_id="e2e-test-user",
                object_permission=ObjectPermission(mcp_servers=[created.server_id]),
            )
        )
        resources.defer(lambda: chat_client.proxy.delete_key(key))

        seeded = chat_client.seed_user_token(alias, key, LINEAR_STORAGE_STATE)
        assert f"{alias}-{LINEAR_READONLY_TOOL}" in seeded, (
            f"the authorize dance listed {seeded}, expected it to include {alias}-{LINEAR_READONLY_TOOL}"
        )

        response = chat_client.chat_with_mcp(
            AuthHeaders.model_validate({"authorization": f"Bearer {key}"}),
            ChatBody(
                model=CHEAP_ANTHROPIC_MODEL,
                messages=[ChatMessage(role="user", content=LINEAR_PROMPT)],
                tools=[
                    McpChatTool(
                        server_url=f"litellm_proxy/mcp/{alias}",
                        server_label=alias,
                        require_approval="never",
                    )
                ],
            ),
        )

        message = response.choices[0].message
        assert message is not None and message.content, f"completion returned no answer: {response}"
        meta = message.provider_specific_fields
        assert meta is not None, f"no MCP metadata on the completion: {response}"
        listed = {t.function.name for t in (meta.mcp_list_tools or []) if t.function}
        assert f"{alias}-{LINEAR_READONLY_TOOL}" in listed, (
            f"the gateway listed {sorted(listed)}, expected the stored token to surface {alias}-{LINEAR_READONLY_TOOL}"
        )
        results = [r for r in (meta.mcp_call_results or []) if r.name == f"{alias}-{LINEAR_READONLY_TOOL}"]
        assert results and results[0].result, (
            f"Linear tool {alias}-{LINEAR_READONLY_TOOL} was not executed with a result: {meta.mcp_call_results}"
        )
