"""Live e2e: a guardrail on the MCP tool-call path blocks banned content in the
tool arguments before the call reaches the upstream MCP server.

A general litellm_content_filter guardrail is configured with mode=pre_mcp_call
(the event type the proxy rewrites pre_call to for a call_mcp_tool) and default_on
(per-key/request guardrail selection is dropped from the synthetic MCP request the
hook sees, so default_on is how it attaches to tools/call). The banned keyword is
unique per run, so default_on only ever intercepts this test's own banned call.

Against the real Datadog MCP server, calling search_datadog_logs with the banned
keyword in the query is blocked with HTTP 400 attributed to the pre_mcp_call hook,
and the tool never runs; the same guardrail lets a clean query through to Datadog.
This is the enforced half (the block) plus the pass-through half in one spec.
"""

from __future__ import annotations

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import Result, Success, UnknownApiError, unwrap
from lifecycle import ResourceManager
from mcp_client import McpCallToolResponse, McpClient, McpToolArguments

pytestmark = pytest.mark.e2e


class TestMcpToolCallGuardrail:
    @pytest.mark.covers(
        "guardrail.litellm_content_filter.pre_mcp_call.blocks",
        exercised_on=["mcp_operations"],
    )
    def test_content_filter_blocks_banned_keyword_in_tool_args(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        assert_dd_mcp_creds()
        marker = unique_marker()
        banned_keyword = f"e2eblocked{marker}"

        guardrail_id = client.register_mcp_content_filter(
            name=f"e2e-mcp-cf-{marker}", blocked_keyword=banned_keyword
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        server_id = register_datadog_mcp(client, resources)
        key = client.generate_key(user_id=f"e2e-mcp-guard-{marker}", mcp_servers=[server_id])
        resources.defer(lambda: client.proxy.delete_key(key))

        tools = unwrap(client.list_tools(key))
        tool_name = tools.tool_name_containing(server_id, SEARCH_LOGS_TOOL)
        assert tool_name is not None, (
            f"granted key never saw {SEARCH_LOGS_TOOL} on server {server_id}; "
            f"tools={tools.tool_names_for_server(server_id)}"
        )

        def search(query: str) -> Result[McpCallToolResponse]:
            arguments: McpToolArguments = {
                "query": query,
                "from": DD_SEARCH_FROM,
                "to": "now",
                "max_tokens": 500,
                "telemetry": {"intent": "e2e mcp guardrail check"},
            }
            return client.call_tool(key, server_id=server_id, name=tool_name, arguments=arguments)

        blocked = search(f"tell me about {banned_keyword}")
        match blocked:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, (
                    f"a banned keyword in the tool arguments must block the MCP tool call with "
                    f"400, got {status}: {body[:300]}"
                )
                assert banned_keyword in body or "content blocked" in body.lower(), (
                    f"the block must name the content-filter reason, got: {body[:300]}"
                )
                assert "pre_mcp_call" in body, (
                    f"the block must be attributed to the MCP tool-call hook (pre_mcp_call), got: {body[:300]}"
                )
            case _:
                pytest.fail(
                    f"content_filter did not block a banned keyword in an MCP tool call; got {blocked}"
                )

        allowed = search(f"e2e-clean-{marker}")
        match allowed:
            case Success(data=result):
                assert result.is_error is not True, (
                    f"a clean MCP tool call must reach the server and not error, got: {result}"
                )
            case _:
                pytest.fail(
                    f"a clean MCP tool call must pass the guardrail and reach the server; got {allowed}"
                )
