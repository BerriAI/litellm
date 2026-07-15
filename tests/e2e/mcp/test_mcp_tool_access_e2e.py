"""Live e2e: MCP gateway tool access under both documented auth headers.

Covers mcp.list_tools.api_key.succeeds + mcp.call_tool.api_key.succeeds (the
`x-litellm-api-key` header), mcp.list_tools.bearer.succeeds +
mcp.call_tool.bearer.succeeds (the `Authorization` header), and
mcp.list_tools.{api_key,bearer}.rejects_unknown_key (the ingress gate),
all against the deterministic mcp-stub compose service (tests/e2e/mcp/stub/).

Each test body spells out every step a human QA run would take, in order:
create the server over the management API, read the record back, mint a
virtual key over /key/generate, build the exact wire header
(`<header>: Bearer sk-...`, the documented contract on the MCP routes; a bare
key in `x-litellm-api-key` is accepted on LLM routes but 401s here), then
drive the real MCP protocol through the gateway the way a production MCP host
does (initialize, tools/list, tools/call over streamable HTTP at
{PROXY}/{alias}/mcp) and assert the enforced behavior. Teardown is the only
thing delegated (resources.defer), because it must run even when an
assertion fails.

The two header styles are one behavior matrix, not two behaviors: the same
test body runs once per header via parametrize, so the specs cannot drift
apart, and each parametrization claims its own registry cells through
param-level `covers` marks. The unknown-key rejection is its own test (also
run per header) and settles the server with a real key first, so its 401 can
only be the gateway's ingress auth refusing the key, never record propagation.
"""

from __future__ import annotations

import pytest

from e2e_config import MCP_STUB_URL, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpClient, McpDenied, McpHeaderName
from models import KeyGenerateBody, McpServerCreateBody

pytestmark = pytest.mark.e2e

STUB_TOOLS = ("echo", "slow_echo", "stats")

AUTH_HEADER_MATRIX = (
    pytest.param(
        "x-litellm-api-key",
        id="x-litellm-api-key",
        marks=(
            pytest.mark.covers("mcp.list_tools.api_key.succeeds"),
            pytest.mark.covers("mcp.call_tool.api_key.succeeds"),
        ),
    ),
    pytest.param(
        "Authorization",
        id="authorization-bearer",
        marks=(
            pytest.mark.covers("mcp.list_tools.bearer.succeeds"),
            pytest.mark.covers("mcp.call_tool.bearer.succeeds"),
        ),
    ),
)

REJECTION_MATRIX = (
    pytest.param(
        "x-litellm-api-key",
        id="x-litellm-api-key",
        marks=pytest.mark.covers("mcp.list_tools.api_key.rejects_unknown_key"),
    ),
    pytest.param(
        "Authorization",
        id="authorization-bearer",
        marks=pytest.mark.covers("mcp.list_tools.bearer.rejects_unknown_key"),
    ),
)


class TestMcpToolAccess:
    """Any virtual key reaches an allow_all_keys server through either
    documented auth header; a key the proxy does not recognize is turned away
    at the door."""

    @pytest.mark.parametrize("header_name", AUTH_HEADER_MATRIX)
    def test_list_and_call_tools(
        self, header_name: McpHeaderName, client: McpClient, resources: ResourceManager
    ) -> None:
        alias = f"e2emcp{unique_marker()}"
        created = client.create_server(McpServerCreateBody(alias=alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_URL
        assert stored.allow_all_keys is True

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))

        headers = {header_name: f"Bearer {key}"}
        names = client.poll_tool_names(alias, headers)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in STUB_TOOLS))
        assert names == expected, f"gateway listed {names}, expected exactly {expected}"

        payload = f"e2e-{unique_marker()}"
        result = client.call_tool(alias, headers, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo call errored: {result.text[:300]}"
        assert result.text == payload

    @pytest.mark.parametrize("header_name", REJECTION_MATRIX)
    def test_unrecognized_key_is_turned_away(
        self, header_name: McpHeaderName, client: McpClient, resources: ResourceManager
    ) -> None:
        """Settle the server with a real key first, then present an unknown
        key over the same header: the refusal must be an explicit 401 from
        session establishment, and can only be the ingress auth because the
        server was just proven servable."""
        alias = f"e2emcp{unique_marker()}"
        created = client.create_server(McpServerCreateBody(alias=alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(created.server_id))

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))
        _ = client.poll_tool_names(alias, {header_name: f"Bearer {key}"})

        denied = client.list_tools_once(alias, {header_name: "Bearer sk-not-a-real-key"})
        assert isinstance(denied, McpDenied), f"unrecognized key was served tools: {denied}"
        assert denied.status_code == 401, f"expected 401 for an unrecognized key, got {denied}"
