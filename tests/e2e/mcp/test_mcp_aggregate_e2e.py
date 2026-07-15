"""Live e2e: the aggregate /mcp namespace over multiple servers.

Covers mcp.list_tools.api_key.aggregates_servers +
mcp.call_tool.api_key.routes_to_target_server: one MCP session at
{PROXY}/mcp, scoped with the `x-mcp-servers` header, must list the
alias-prefixed union of every named server's tools and dispatch each call to
the server that owns the prefixed tool. This is the mount shape production
MCP hosts use (one gateway endpoint, many upstreams), where per-server URLs
cannot catch cross-server routing bugs by construction.

The two registered servers point at stub mounts with deliberately disjoint
tool sets (tests/e2e/mcp/stub/): only the /second mount serves `second_ping`.
A successful `{second_alias}-second_ping` call answered with the /second
mount's canned reply is therefore proof of correct dispatch, and the
fail-before-fix evidence: a gateway that routed by anything other than the
prefixed owner would hit the main mount, which cannot answer that tool. The
scoped re-listing at the end pins the header contract (naming one alias must
hide the sibling's tools), which keeps the exact-equality listing assertions
meaningful when other suites register their own servers concurrently.
"""

from __future__ import annotations

import pytest

from e2e_config import MCP_STUB_SECOND_URL, MCP_STUB_URL, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpClient, McpDenied, McpToolNames
from models import KeyGenerateBody, McpServerCreateBody

pytestmark = pytest.mark.e2e

STUB_TOOLS = ("echo", "slow_echo", "stats")
SECOND_STUB_TOOLS = ("second_ping",)


class TestMcpAggregateNamespace:
    """One aggregate session spans every server named in `x-mcp-servers`,
    lists their tools under alias prefixes, and routes calls by prefix."""

    @pytest.mark.covers("mcp.list_tools.api_key.aggregates_servers")
    @pytest.mark.covers("mcp.call_tool.api_key.routes_to_target_server")
    def test_one_session_lists_both_servers_and_routes_calls(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        first_alias = f"e2emcpagga{marker}"
        first = client.create_server(McpServerCreateBody(alias=first_alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(first.server_id))

        second_alias = f"e2emcpaggb{marker}"
        second = client.create_server(
            McpServerCreateBody(alias=second_alias, url=MCP_STUB_SECOND_URL, allow_all_keys=True)
        )
        resources.defer(lambda: client.delete_server(second.server_id))

        assert client.server_info(first.server_id).url == MCP_STUB_URL
        assert client.server_info(second.server_id).url == MCP_STUB_SECOND_URL

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))

        headers = {
            "x-litellm-api-key": f"Bearer {key}",
            "x-mcp-servers": f"{first_alias},{second_alias}",
        }

        _ = client.poll_aggregate_tool_names(headers, until_listed=f"{first_alias}-echo")
        names = client.poll_aggregate_tool_names(headers, until_listed=f"{second_alias}-second_ping")
        expected = tuple(
            sorted(
                [f"{first_alias}-{tool}" for tool in STUB_TOOLS]
                + [f"{second_alias}-{tool}" for tool in SECOND_STUB_TOOLS]
            )
        )
        assert names == expected, f"aggregate session listed {names}, expected exactly {expected}"

        payload = f"e2e-{marker}"
        echoed = client.aggregate_call_tool(headers, f"{first_alias}-echo", {"text": payload})
        assert echoed.is_error is False, f"aggregate echo errored: {echoed.text[:300]}"
        assert echoed.text == payload

        pinged = client.aggregate_call_tool(headers, f"{second_alias}-second_ping", {})
        assert pinged.is_error is False, f"aggregate second_ping errored: {pinged.text[:300]}"
        assert pinged.text == "pong-from-second", (
            f"call must dispatch to the /second upstream that owns the tool, got {pinged.text!r}"
        )

        second_only_headers = {"x-litellm-api-key": f"Bearer {key}", "x-mcp-servers": second_alias}
        scoped = client.aggregate_list_tools_once(second_only_headers)
        match scoped:
            case McpToolNames(names=only):
                assert only == (f"{second_alias}-second_ping",), (
                    f"x-mcp-servers scoped to {second_alias} must hide the sibling's tools, listed {only}"
                )
            case McpDenied() as denied:
                pytest.fail(f"scoped aggregate listing was refused: {denied}")
