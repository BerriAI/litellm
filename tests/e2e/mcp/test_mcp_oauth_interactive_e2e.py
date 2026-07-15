"""Live e2e: the interactive (authorization_code) OAuth MCP flow.

Covers mcp.list_tools.oauth.completes_authorization_code_flow and
mcp.call_tool.oauth.uses_per_user_token: a gateway-managed oauth2 server in
the authorization_code flow must challenge an unauthorized MCP session with
401 + WWW-Authenticate, let a real MCP host complete the whole interactive
dance (RFC 9728/8414 discovery against the gateway, RFC 7591 dynamic client
registration, the authorize redirect through the upstream IdP, the
PKCE-verified token exchange), and then serve tools/list and tools/call using
the per-user upstream token the gateway obtained; the caller's virtual key
must never leave the gateway.

The MCP-host side is the official mcp SDK's own OAuth machinery
(OAuthClientProvider), the same code path desktop MCP hosts run; only the
browser leg is replaced by a redirect chaser that follows the authorize chain
(gateway -> stub IdP -> gateway callback -> host redirect_uri) with plain
GETs. The upstream is the /oauthuser stub mount, which 401s anything but the
exact access token the stub IdP's authorization_code grant hands out, so a
served call proves the gateway completed the code exchange (the stub verifies
client credentials, redirect_uri, one-time code, and the S256 code_verifier)
rather than forwarding anything it already had. The recorded_headers
read-back then makes the injected token explicit and adds the leak check.
"""

from __future__ import annotations

import pytest

from e2e_config import (
    MCP_STUB_AUTHORIZE_BROWSER_URL,
    MCP_STUB_OAUTH_USER_ACCESS_TOKEN,
    MCP_STUB_OAUTH_USER_CLIENT_ID,
    MCP_STUB_OAUTH_USER_CLIENT_SECRET,
    MCP_STUB_OAUTHUSER_URL,
    MCP_STUB_TOKEN_URL,
    MCP_STUB_URL,
    unique_marker,
)
from lifecycle import ResourceManager
from mcp_client import InMemoryTokenStorage, McpClient, McpDenied, StubRecordedHeaders
from models import KeyGenerateBody, McpServerCreateBody, McpServerCredentials

pytestmark = pytest.mark.e2e

GUARDED_STUB_TOOLS = ("echo", "recorded_headers")


class TestMcpOauthAuthorizationCode:
    """An authorization_code server challenges unauthenticated sessions, runs
    the full interactive dance with a real MCP host, and serves tools with the
    per-user token the gateway obtained from the IdP."""

    @pytest.mark.covers("mcp.list_tools.oauth.completes_authorization_code_flow")
    @pytest.mark.covers("mcp.call_tool.oauth.uses_per_user_token")
    def test_challenge_then_authorization_code_dance_reaches_tools(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        alias = f"e2emcpauthcode{marker}"
        created = client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=MCP_STUB_OAUTHUSER_URL,
                allow_all_keys=True,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
                authorization_url=MCP_STUB_AUTHORIZE_BROWSER_URL,
                token_url=MCP_STUB_TOKEN_URL,
                credentials=McpServerCredentials(
                    client_id=MCP_STUB_OAUTH_USER_CLIENT_ID,
                    client_secret=MCP_STUB_OAUTH_USER_CLIENT_SECRET,
                ),
            )
        )
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.auth_type == "oauth2"
        assert stored.oauth2_flow == "authorization_code"
        assert stored.authorization_url == MCP_STUB_AUTHORIZE_BROWSER_URL
        assert stored.token_url == MCP_STUB_TOKEN_URL
        assert stored.credentials is None, f"client secret must be redacted on read-back, got {stored.credentials}"

        key = client.gateway.generate_key(KeyGenerateBody(user_id="e2e-test-user"))
        resources.defer(lambda: client.gateway.delete_key(key))
        headers = {"x-litellm-api-key": f"Bearer {key}"}

        control_alias = f"e2emcpauthcodectl{marker}"
        control = client.create_server(
            McpServerCreateBody(alias=control_alias, url=MCP_STUB_URL, allow_all_keys=True)
        )
        resources.defer(lambda: client.delete_server(control.server_id))
        _ = client.poll_tool_names(control_alias, headers)

        challenged = client.list_tools_once(alias, headers)
        assert isinstance(challenged, McpDenied), f"session without a user token was served tools: {challenged}"
        assert challenged.status_code == 401, f"expected the 401 OAuth challenge, got {challenged}"

        storage = InMemoryTokenStorage()
        names = client.poll_oauth_tool_names(alias, headers, storage)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in GUARDED_STUB_TOOLS))
        assert names == expected, f"post-dance listing was {names}, expected exactly {expected}"

        payload = f"e2e-{marker}"
        result = client.oauth_call_tool(alias, headers, storage, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo through the user-token upstream errored: {result.text[:300]}"
        assert result.text == payload

        recorded = client.oauth_call_tool(alias, headers, storage, f"{alias}-recorded_headers", {})
        assert recorded.is_error is False, f"recorded_headers call errored: {recorded.text[:300]}"
        upstream_headers = StubRecordedHeaders.model_validate_json(recorded.text).root
        assert upstream_headers.get("authorization") == f"Bearer {MCP_STUB_OAUTH_USER_ACCESS_TOKEN}", (
            "upstream must receive exactly the per-user token the stub IdP hands out for the code exchange, "
            f"got {upstream_headers.get('authorization')!r}"
        )
        leaked = sorted(name for name, value in upstream_headers.items() if key in value)
        assert leaked == [], f"caller's virtual key crossed the gateway boundary in header(s) {leaked}"
