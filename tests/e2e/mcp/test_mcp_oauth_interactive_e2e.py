"""Live e2e: list-and-call through the interactive (authorization_code) OAuth flow.

Covers the core list_tools/call_tool happy path against a real OAuth MCP
server, once per documented ingress header, plus
mcp.list_tools.oauth.completes_authorization_code_flow and
mcp.call_tool.oauth.uses_per_user_token: a gateway-managed oauth2 server in
the authorization_code flow must challenge an unauthorized MCP session with
401 + WWW-Authenticate, let a real MCP host complete the whole interactive
dance (RFC 9728/8414 discovery against the gateway, RFC 7591 dynamic client
registration, the authorize redirect through the upstream IdP, the
PKCE-verified token exchange), and then serve tools/list and tools/call using
the per-user upstream token the gateway obtained; the caller's virtual key
must never leave the gateway.

The caller is a real scoped internal user, not a wildcard: the virtual key is
user-bound and granted the oauth server through object_permission (the server
is not allow_all_keys), so the flow tested is exactly what a permissioned
customer key experiences. The MCP-host side is the official mcp SDK's own
OAuth machinery (OAuthClientProvider), the same code path desktop MCP hosts
run; only the browser leg is replaced by a redirect chaser that follows the
authorize chain (gateway -> stub IdP -> gateway callback -> host
redirect_uri) with plain GETs. The upstream is the /oauthuser stub mount,
which 401s anything but the exact access token the stub IdP's
authorization_code grant hands out, so a served call proves the gateway
completed the code exchange rather than forwarding anything it already had;
the recorded_headers read-back makes the injected token explicit and adds the
leak check.

The two header styles are deliberately separate, explicitly named tests, and
they assert the same contract: which header carries the key must not change
what the OAuth flow does. Once the dance completes, the SDK carries the
gateway-minted OAuth token in `Authorization` on MCP traffic (overwriting a
key placed there), which is why the x-litellm-api-key form is what production
hosts configure; the Authorization form pins that a key presented the other
documented way is still challenged into the same working flow.
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
from models import KeyGenerateBody, KeyObjectPermission, McpServerCreateBody, McpServerCredentials

pytestmark = pytest.mark.e2e

GUARDED_STUB_TOOLS = ("echo", "recorded_headers")


class TestMcpOauthAuthorizationCode:
    """A scoped internal-user key on an authorization_code server is
    challenged, completes the interactive dance, and lists and calls tools
    with the per-user token the gateway obtained from the IdP."""

    @pytest.mark.covers("mcp.list_tools.api_key.succeeds")
    @pytest.mark.covers("mcp.call_tool.api_key.succeeds")
    @pytest.mark.covers("mcp.list_tools.oauth.completes_authorization_code_flow")
    @pytest.mark.covers("mcp.call_tool.oauth.uses_per_user_token")
    def test_list_and_call_tools_with_x_litellm_api_key_header(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        alias = f"e2emcpauthcode{marker}"
        created = client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=MCP_STUB_OAUTHUSER_URL,
                allow_all_keys=False,
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
        assert stored.allow_all_keys is False
        assert stored.credentials is None, f"client secret must be redacted on read-back, got {stored.credentials}"

        key = client.gateway.generate_key(
            KeyGenerateBody(
                user_id="e2e-test-user",
                object_permission=KeyObjectPermission(mcp_servers=[created.server_id]),
            )
        )
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

    @pytest.mark.covers("mcp.list_tools.bearer.succeeds")
    @pytest.mark.covers("mcp.call_tool.bearer.succeeds")
    def test_list_and_call_tools_with_authorization_bearer_header(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        """The identical contract with the key presented as
        `Authorization: Bearer <key>` instead: the gateway must challenge the
        pre-dance session with 401 exactly like the x-litellm-api-key form,
        and the completed dance must serve the same tools."""
        marker = unique_marker()
        alias = f"e2emcpauthcodez{marker}"
        created = client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=MCP_STUB_OAUTHUSER_URL,
                allow_all_keys=False,
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
        assert stored.allow_all_keys is False

        key = client.gateway.generate_key(
            KeyGenerateBody(
                user_id="e2e-test-user",
                object_permission=KeyObjectPermission(mcp_servers=[created.server_id]),
            )
        )
        resources.defer(lambda: client.gateway.delete_key(key))
        headers = {"Authorization": f"Bearer {key}"}

        control_alias = f"e2emcpauthcodezctl{marker}"
        control = client.create_server(
            McpServerCreateBody(alias=control_alias, url=MCP_STUB_URL, allow_all_keys=True)
        )
        resources.defer(lambda: client.delete_server(control.server_id))
        _ = client.poll_tool_names(control_alias, headers)

        challenged = client.list_tools_once(alias, headers)
        assert isinstance(challenged, McpDenied), (
            f"session without a user token was served tools instead of the OAuth challenge: {challenged}"
        )
        assert challenged.status_code == 401, f"expected the 401 OAuth challenge, got {challenged}"

        storage = InMemoryTokenStorage()
        names = client.poll_oauth_tool_names(alias, headers, storage)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in GUARDED_STUB_TOOLS))
        assert names == expected, f"post-dance listing was {names}, expected exactly {expected}"

        payload = f"e2e-{marker}"
        result = client.oauth_call_tool(alias, headers, storage, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo through the user-token upstream errored: {result.text[:300]}"
        assert result.text == payload
