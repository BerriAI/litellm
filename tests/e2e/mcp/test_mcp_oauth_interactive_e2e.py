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

The documented header placements are covered by separate tests asserting
different, achievable contracts. The x-litellm-api-key form runs the full
interactive round-trip, because that header stays free for the LiteLLM key
while the SDK owns `Authorization` for the minted OAuth token. The
Authorization form is split by whether a per-user token already exists: with
no stored token, the LiteLLM key in Authorization cannot complete the dance
(the token would evict the key on the post-dance retry, so the gateway loses
the identity it needs), so the contract is that the gateway fails closed and
loudly with the 401 challenge rather than silently masking an empty tool list;
with a stored token, no dance is needed, so a plain Authorization-key session
lists and calls using the token the gateway already holds for that user.
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
    """A scoped internal-user key on an authorization_code server, across three
    behaviors: the x-litellm-api-key form is challenged, completes the
    interactive dance, and lists/calls with the per-user token the gateway
    obtained; a key in Authorization with no stored token yet is challenged
    (not masked), since the OAuth token would claim the Authorization slot and
    the dance cannot complete there; and once a per-user token is stored, a
    plain Authorization-key session lists/calls using that stored token."""

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

    @pytest.mark.covers("mcp.list_tools.oauth.challenges_bearer_key_not_masked")
    def test_key_in_authorization_header_is_challenged_not_masked(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        """A virtual key presented in `Authorization: Bearer <key>` on a
        gateway-managed authorization_code server, with no stored per-user
        token, must get the 401 OAuth challenge, not a masked empty tool list.

        This is deliberately a challenge-guard, not a full list-and-call: the
        full interactive round-trip is unreachable via this header, because
        once the host completes the OAuth dance the SDK carries the minted
        upstream token in `Authorization`, evicting the LiteLLM key the gateway
        needs to resolve the per-user token. So the only supported way to
        complete the flow is the x-litellm-api-key form (covered above); the
        contract this pins is that the Authorization form fails closed and
        loudly (challenge) rather than silently (200 + empty tools). Before the
        gateway classified the challenge per oauth2 sub-mode, a bearer in
        Authorization suppressed the challenge and the session opened masked;
        this test is the regression guard for that."""
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
            f"key in Authorization was served a (masked) tool list instead of the OAuth challenge: {challenged}"
        )
        assert challenged.status_code == 401, f"expected the 401 OAuth challenge, got {challenged}"

    @pytest.mark.covers("mcp.list_tools.bearer.succeeds")
    @pytest.mark.covers("mcp.call_tool.bearer.succeeds")
    def test_stored_token_serves_list_and_call_with_key_in_authorization_header(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        """A returning user who has already authorized presents the LiteLLM key
        in `Authorization: Bearer <key>` and the gateway lists and calls tools
        using the stored per-user token.

        This is the positive counterpart to the challenge-guard above. It works
        because no interactive dance is needed once a per-user token exists: the
        gateway resolves the user from the LiteLLM key in Authorization and
        injects the stored upstream token at egress. (The dance itself cannot be
        completed with the key in Authorization, since the SDK would overwrite
        that header with the minted token; that is why the token is seeded here
        via the x-litellm-api-key dance first, then the plain Authorization-key
        session is what serves.) The recorded_headers read-back proves the
        upstream received the stored per-user token, not the LiteLLM key, and
        the key crosses the gateway boundary in no header."""
        marker = unique_marker()
        alias = f"e2emcpauthcodestored{marker}"
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
        assert stored.credentials is None, f"client secret must be redacted on read-back, got {stored.credentials}"

        key = client.gateway.generate_key(
            KeyGenerateBody(
                user_id="e2e-test-user",
                object_permission=KeyObjectPermission(mcp_servers=[created.server_id]),
            )
        )
        resources.defer(lambda: client.gateway.delete_key(key))

        storage = InMemoryTokenStorage()
        seeded = client.poll_oauth_tool_names(alias, {"x-litellm-api-key": f"Bearer {key}"}, storage)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in GUARDED_STUB_TOOLS))
        assert seeded == expected, f"the authorizing dance listed {seeded}, expected exactly {expected}"

        authorization_headers = {"Authorization": f"Bearer {key}"}
        names = client.poll_tool_names(alias, authorization_headers)
        assert names == expected, f"Authorization-key listing was {names}, expected exactly {expected}"

        payload = f"e2e-{marker}"
        result = client.call_tool(alias, authorization_headers, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo with the key in Authorization errored: {result.text[:300]}"
        assert result.text == payload

        recorded = client.call_tool(alias, authorization_headers, f"{alias}-recorded_headers", {})
        assert recorded.is_error is False, f"recorded_headers call errored: {recorded.text[:300]}"
        upstream_headers = StubRecordedHeaders.model_validate_json(recorded.text).root
        assert upstream_headers.get("authorization") == f"Bearer {MCP_STUB_OAUTH_USER_ACCESS_TOKEN}", (
            "upstream must receive the stored per-user token the gateway injects, not the LiteLLM key, "
            f"got {upstream_headers.get('authorization')!r}"
        )
        leaked = sorted(name for name, value in upstream_headers.items() if key in value)
        assert leaked == [], f"caller's virtual key crossed the gateway boundary in header(s) {leaked}"
