"""Live e2e: the MCP gateway's two governance axes.

Covers mcp.list_tools.api_key.denied_without_permission +
mcp.call_tool.api_key.denied_without_permission (server-level access: a key
without an object_permission grant for a non-allow-all server must neither
see its tools nor call them) and mcp.list_tools.api_key.filters_allowed_tools
+ mcp.call_tool.api_key.blocks_tool_outside_allowed_tools (tool-level access:
a server's `allowed_tools` subset must bound both the listing and the call
path), all against the deterministic mcp-stub compose service.

The denial test is built so it cannot pass for the wrong reason. A sibling
key with the grant drives the same server through the same machinery first
(so record propagation and a working grant path are proven), and the denied
key first lists an allow-all control server (so the key itself is proven
valid and propagated). Only then do the denial assertions run: whatever the
denied key is refused on cannot be blamed on propagation lag or a broken key,
which is the fail-before-fix evidence for the permission guard; removing the
guard makes the granted-vs-denied outcomes identical and the test fail.

The allowed_tools test asserts the exact filtered listing (a broken filter
serves all three stub tools and fails the equality immediately) and that the
excluded tool is refused on the call path, since filtering only the listing
would leave governance bypassable by anyone who knows a tool's name.

Both denial contracts are pinned to the gateway's observed live behavior: a
denied listing is a served-but-empty tool list (the session is admitted, the
tools are filtered), and a denied call is an in-band tool error ("not allowed
to call this tool" / "is not allowed for server"), not a transport-level 4xx,
because a mid-session JSON-RPC call cannot carry an HTTP status.
"""

from __future__ import annotations

import pytest

from e2e_config import MCP_STUB_URL, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpClient, McpToolNames, McpToolText
from models import KeyGenerateBody, KeyObjectPermission, McpServerCreateBody

pytestmark = pytest.mark.e2e

STUB_TOOLS = ("echo", "slow_echo", "stats")


class TestMcpServerAccessControl:
    """A non-allow-all server is invisible and uncallable to keys without its
    object_permission grant, while a granted key uses it normally."""

    @pytest.mark.covers("mcp.list_tools.api_key.denied_without_permission")
    @pytest.mark.covers("mcp.call_tool.api_key.denied_without_permission")
    def test_key_without_grant_sees_no_tools_and_cannot_call(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        guarded_alias = f"e2emcpacl{marker}"
        guarded = client.create_server(
            McpServerCreateBody(alias=guarded_alias, url=MCP_STUB_URL, allow_all_keys=False)
        )
        resources.defer(lambda: client.delete_server(guarded.server_id))

        control_alias = f"e2emcpaclctl{marker}"
        control = client.create_server(
            McpServerCreateBody(alias=control_alias, url=MCP_STUB_URL, allow_all_keys=True)
        )
        resources.defer(lambda: client.delete_server(control.server_id))

        assert client.server_info(guarded.server_id).allow_all_keys is False

        granted_key = client.gateway.generate_key(
            KeyGenerateBody(object_permission=KeyObjectPermission(mcp_servers=[guarded.server_id]))
        )
        resources.defer(lambda: client.gateway.delete_key(granted_key))
        denied_key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(denied_key))

        granted_headers = {"x-litellm-api-key": f"Bearer {granted_key}"}
        denied_headers = {"x-litellm-api-key": f"Bearer {denied_key}"}

        names = client.poll_tool_names(guarded_alias, granted_headers)
        expected = tuple(sorted(f"{guarded_alias}-{tool}" for tool in STUB_TOOLS))
        assert names == expected, f"granted key listed {names}, expected exactly {expected}"

        payload = f"e2e-{marker}"
        result = client.call_tool(guarded_alias, granted_headers, f"{guarded_alias}-echo", {"text": payload})
        assert result.is_error is False, f"granted key's echo errored: {result.text[:300]}"
        assert result.text == payload

        _ = client.poll_tool_names(control_alias, denied_headers)

        denied_list = client.list_tools_once(guarded_alias, denied_headers)
        assert denied_list == McpToolNames(names=()), (
            f"key without the grant must be served an empty listing, got: {denied_list}"
        )

        denied_call = client.call_tool_once(guarded_alias, denied_headers, f"{guarded_alias}-echo", {"text": payload})
        match denied_call:
            case McpToolText(is_error=True, text=text) if "not allowed to call this tool" in text:
                pass
            case other:
                pytest.fail(f"expected the in-band 'not allowed to call this tool' error, got: {other}")


class TestMcpAllowedToolsFilter:
    """`allowed_tools` bounds the server to a subset of its upstream tools, on
    the listing and on the call path."""

    @pytest.mark.covers("mcp.list_tools.api_key.filters_allowed_tools")
    @pytest.mark.covers("mcp.call_tool.api_key.blocks_tool_outside_allowed_tools")
    def test_allowed_tools_filters_listing_and_blocks_excluded_call(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        alias = f"e2emcptoolgov{unique_marker()}"
        created = client.create_server(
            McpServerCreateBody(
                alias=alias,
                url=MCP_STUB_URL,
                allow_all_keys=True,
                allowed_tools=["echo", "stats"],
            )
        )
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.allowed_tools == ["echo", "stats"]

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))

        headers = {"x-litellm-api-key": f"Bearer {key}"}
        names = client.poll_tool_names(alias, headers)
        expected = tuple(sorted((f"{alias}-echo", f"{alias}-stats")))
        assert names == expected, f"allowed_tools listing was {names}, expected exactly {expected}"

        payload = f"e2e-{unique_marker()}"
        result = client.call_tool(alias, headers, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"allowed tool errored: {result.text[:300]}"
        assert result.text == payload

        blocked = client.call_tool_once(
            alias, headers, f"{alias}-slow_echo", {"text": payload, "marker": unique_marker(), "sleep_seconds": 0}
        )
        match blocked:
            case McpToolText(is_error=True, text=text) if "is not allowed for server" in text:
                pass
            case other:
                pytest.fail(f"expected the in-band 'not allowed for server' error for the excluded tool, got: {other}")
