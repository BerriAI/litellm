"""Live e2e: the gateway attaches the server's stored upstream credentials.

Covers mcp.call_tool.api_key.injects_upstream_credential (a static shared-key
credential is injected as X-API-Key) and mcp.list_tools.oauth.succeeds +
mcp.call_tool.oauth.succeeds (OAuth2 client_credentials: the gateway exchanges
the stored client id/secret at the token endpoint and sends the minted access
token upstream), against the guarded mcp-stub mounts (tests/e2e/mcp/stub/).

Both tests follow the suite lifecycle: register the server with credentials
over the management API and defer its deletion, assert the recorded state
round-trips with the secret redacted (GET /v1/mcp/server/{id} echoes the auth
config but nulls `credentials`), then drive initialize + tools/list +
tools/call through the gateway and assert the enforced behavior. The guarded
stub mounts 401 any request that does not carry exactly the expected
credential, so a served call is itself proof of injection; that is the
fail-before-fix evidence built into the design, since a gateway that stops
attaching the credential (or attaches the wrong one) cannot list a single
tool here. The `recorded_headers` read-back then makes the assertion explicit
and adds the boundary check: the caller's virtual key must appear in no
header the upstream received (the LIT-3794 class of bug, where the proxy
forwards its caller's own credential upstream).
"""

from __future__ import annotations

import pytest

from e2e_config import (
    MCP_STUB_APIKEY_URL,
    MCP_STUB_OAUTH_ACCESS_TOKEN,
    MCP_STUB_OAUTH_CLIENT_ID,
    MCP_STUB_OAUTH_CLIENT_SECRET,
    MCP_STUB_OAUTH_SCOPE,
    MCP_STUB_OAUTH_URL,
    MCP_STUB_TOKEN_URL,
    MCP_STUB_UPSTREAM_API_KEY,
    unique_marker,
)
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import KeyGenerateBody, McpServerCreateBody, McpServerCredentials

pytestmark = pytest.mark.e2e

GUARDED_STUB_TOOLS = ("echo", "recorded_headers")


class TestMcpUpstreamSharedKeyInjection:
    """A server stored with `auth_type: api_key` reaches its guarded upstream:
    the gateway injects the stored secret as X-API-Key on every egress request
    and the secret never travels back out of the management API."""

    @pytest.mark.covers("mcp.call_tool.api_key.injects_upstream_credential")
    def test_stored_api_key_reaches_upstream_and_never_leaks(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        alias = f"e2emcpkeyauth{unique_marker()}"
        created = client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=MCP_STUB_APIKEY_URL,
                allow_all_keys=True,
                auth_type="api_key",
                credentials=McpServerCredentials(auth_value=MCP_STUB_UPSTREAM_API_KEY),
            )
        )
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.auth_type == "api_key"
        assert stored.url == MCP_STUB_APIKEY_URL
        assert stored.credentials is None, f"stored secret must be redacted on read-back, got {stored.credentials}"

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))

        headers = {"x-litellm-api-key": f"Bearer {key}"}
        names = client.poll_tool_names(alias, headers)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in GUARDED_STUB_TOOLS))
        assert names == expected, f"guarded upstream listed {names}, expected exactly {expected}"

        payload = f"e2e-{unique_marker()}"
        result = client.call_tool(alias, headers, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo through the guarded upstream errored: {result.text[:300]}"
        assert result.text == payload

        upstream_headers = client.stub_recorded_headers(alias, headers, f"{alias}-recorded_headers")
        assert upstream_headers.get("x-api-key") == MCP_STUB_UPSTREAM_API_KEY
        leaked = sorted(name for name, value in upstream_headers.items() if key in value)
        assert leaked == [], f"caller's virtual key crossed the gateway boundary in header(s) {leaked}"


class TestMcpOauth2ClientCredentials:
    """A server stored with `auth_type: oauth2` in the client_credentials (M2M)
    flow works end to end: the gateway exchanges the stored client id/secret
    (with the configured scope) at the token endpoint and presents the minted
    access token upstream; the caller's own credential stays at the gateway."""

    @pytest.mark.covers("mcp.list_tools.oauth.succeeds")
    @pytest.mark.covers("mcp.call_tool.oauth.succeeds")
    def test_gateway_exchanges_client_credentials_and_sends_minted_token(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        alias = f"e2emcpoauth{unique_marker()}"
        created = client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=MCP_STUB_OAUTH_URL,
                allow_all_keys=True,
                auth_type="oauth2",
                oauth2_flow="client_credentials",
                token_url=MCP_STUB_TOKEN_URL,
                credentials=McpServerCredentials(
                    client_id=MCP_STUB_OAUTH_CLIENT_ID,
                    client_secret=MCP_STUB_OAUTH_CLIENT_SECRET,
                    scopes=[MCP_STUB_OAUTH_SCOPE],
                ),
            )
        )
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.auth_type == "oauth2"
        assert stored.oauth2_flow == "client_credentials"
        assert stored.token_url == MCP_STUB_TOKEN_URL
        assert stored.credentials is None, f"client secret must be redacted on read-back, got {stored.credentials}"

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))

        headers = {"x-litellm-api-key": f"Bearer {key}"}
        names = client.poll_tool_names(alias, headers)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in GUARDED_STUB_TOOLS))
        assert names == expected, f"oauth upstream listed {names}, expected exactly {expected}"

        payload = f"e2e-{unique_marker()}"
        result = client.call_tool(alias, headers, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo through the oauth upstream errored: {result.text[:300]}"
        assert result.text == payload

        upstream_headers = client.stub_recorded_headers(alias, headers, f"{alias}-recorded_headers")
        assert upstream_headers.get("authorization") == f"Bearer {MCP_STUB_OAUTH_ACCESS_TOKEN}", (
            "upstream must receive exactly the token the stub IdP mints for the stored client credentials, "
            f"got {upstream_headers.get('authorization')!r}"
        )
        leaked = sorted(name for name, value in upstream_headers.items() if key in value)
        assert leaked == [], f"caller's virtual key crossed the gateway boundary in header(s) {leaked}"
