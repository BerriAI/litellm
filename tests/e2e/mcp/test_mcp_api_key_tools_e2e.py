"""Live e2e: the MCP gateway brokers a config-registered upstream server over a virtual key.

Exercises the api_key auth family end to end: a LiteLLM virtual key (x-litellm-api-key)
discovers an upstream MCP server's tools (tools/list) and invokes one (tools/call), with
the gateway injecting the upstream credential itself. The upstream here is the
config-registered "hugging_face" server; it is only the vehicle. The call runs
hub_repo_search and asserts a Hub search for GPT-2 under the openai-community namespace
surfaces openai-community/gpt2, so a broken grant, a dropped upstream credential, or a
tool that never ran fails rather than passing on a bare 200. The server is owned by
gateway config, so the test looks it up by name, hard-fails (never skips) when absent,
and tears down only the virtual key.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient

pytestmark = pytest.mark.e2e

UPSTREAM_SERVER_NAME = "hugging_face"
SEARCH_TOOL = "hub_repo_search"
SEARCH_QUERY = "gpt2"
SEARCH_AUTHOR = "openai-community"
CANONICAL_REPO_ID = "openai-community/gpt2"


def _find_server(client: McpClient, server_name: str) -> str:
    """Return the server_id of a config-registered MCP server, or hard-fail.

    Live e2e never skips for a missing upstream: a gateway without the server is a red
    run so the operator knows the suite cannot prove the product path.
    """
    for row in client.registered_servers():
        if row.server_name == server_name:
            return row.server_id
    pytest.fail(
        f"gateway has no {server_name!r} MCP server registered; add it to the proxy "
        "config and restart"
    )


class TestMcpApiKeyToolAccess:
    @pytest.mark.covers("mcp.list_tools.api_key.succeeds", "mcp.call_tool.api_key.succeeds")
    def test_virtual_key_lists_and_calls_upstream_tool(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id = _find_server(client, UPSTREAM_SERVER_NAME)

        key = client.generate_key(
            user_id=f"e2e-mcp-api-key-{unique_marker()}",
            mcp_servers=[server_id],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        tools = unwrap(client.list_tools(key))
        tool_name = tools.tool_name_containing(server_id, SEARCH_TOOL)
        assert tool_name is not None, (
            f"granted key never saw {SEARCH_TOOL} on server {UPSTREAM_SERVER_NAME!r}; "
            f"tools={tools.tool_names_for_server(server_id)}"
        )

        call = unwrap(
            client.call_tool(
                key,
                server_id=server_id,
                name=tool_name,
                arguments={
                    "query": SEARCH_QUERY,
                    "author": SEARCH_AUTHOR,
                    "repo_types": ["model"],
                    "limit": 10,
                },
            )
        )
        assert call.is_error is not True, f"{SEARCH_TOOL} errored: {call}"
        body = call.all_text
        assert CANONICAL_REPO_ID in body, (
            f"{SEARCH_TOOL} for {SEARCH_QUERY!r} under {SEARCH_AUTHOR!r} must surface "
            f"{CANONICAL_REPO_ID!r}; got: {body[:800]!r}"
        )
