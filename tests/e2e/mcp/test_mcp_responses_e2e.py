"""Live e2e: /v1/responses expands a gateway-registered MCP server and
auto-executes its tools, surfacing the results as response output items.

Registers the real Datadog remote MCP server, grants a key access to it, then
sends a /v1/responses request whose ``tools`` array carries an
``{type: "mcp", server_url: "litellm_proxy/mcp/<alias>"}`` reference. The
gateway lists the server's tools, feeds them to the model in Responses API
format, the model calls search_datadog_logs, the gateway executes the call
upstream, and appends ``mcp_tools_fetched`` and ``tool_execution_results``
output items to the response. Their presence proves the full Responses API
bridge loop ran end to end against a real MCP server and a real LLM.
"""

from __future__ import annotations

import pytest

from datadog_mcp import assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient, ResponsesMcpBody, ResponsesMcpInputMessage, ResponsesMcpTool

pytestmark = pytest.mark.e2e


class TestResponsesMcpAutoExecute:
    @pytest.mark.covers("mcp.responses.api_key.auto_executes_tools")
    def test_responses_lists_calls_and_executes_mcp_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd = register_datadog_mcp(client, resources)
        client.await_registered(dd.server_id)

        marker = f"e2e-mcp-resp-nohit-{unique_marker()}"
        key = client.generate_key(
            user_id=f"e2e-mcp-resp-{unique_marker()}",
            mcp_servers=[dd.server_id],
            models=[CHEAP_ANTHROPIC_MODEL],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        body = ResponsesMcpBody(
            model=CHEAP_ANTHROPIC_MODEL,
            input=[
                ResponsesMcpInputMessage(
                    content=(
                        "Use the search_datadog_logs tool to search for logs "
                        f"with query '{marker}' from now-30m to now with "
                        "max_tokens 100. After you get results, reply with ok only."
                    )
                )
            ],
            instructions="You are a helpful assistant.",
            tools=[
                ResponsesMcpTool(
                    server_label="datadog",
                    server_url=f"litellm_proxy/mcp/{dd.alias}",
                    require_approval="never",
                )
            ],
        )
        result = unwrap(client.responses_with_mcp(key, body))

        fetched = result.mcp_tools_fetched
        assert fetched is not None, (
            "response.output has no mcp_tools_fetched item; the gateway did not list "
            f"the Datadog server's tools through the responses bridge: {result.output}"
        )
        assert fetched.content, (
            "mcp_tools_fetched item has no content; the tool list was empty"
        )

        executed = result.tool_execution_results
        assert executed is not None, (
            "response.output has no tool_execution_results item; the gateway did not "
            f"execute any MCP tool through the responses bridge: {result.output}"
        )
        assert executed.content, (
            "tool_execution_results item has no content; the tool call returned nothing"
        )
