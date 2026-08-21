"""Live e2e: /chat/completions expands a gateway-registered MCP server and
auto-executes its tools in one agentic turn.

Registers the real Datadog remote MCP server, grants a key access to it, then
sends a /chat/completions request whose ``tools`` array carries an
``{type: "mcp", server_url: "litellm_proxy/mcp/<alias>"}`` reference. The gateway
lists the server's tools, feeds them to the model, the model calls
search_datadog_logs, the gateway executes the call upstream, and folds the
result back into a follow-up completion. The response must carry
provider_specific_fields.mcp_list_tools (the gateway listed tools),
mcp_tool_calls (the model called one), and mcp_call_results (the gateway
executed it), proving the full bridge loop ran end to end against a real MCP
server and a real LLM.
"""

from __future__ import annotations

import pytest

from datadog_mcp import assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import ChatBody, ChatMessage, McpChatTool

pytestmark = pytest.mark.e2e


class TestChatCompletionMcpAutoExecute:
    @pytest.mark.covers("mcp.chat_completion.api_key.auto_executes_tools")
    def test_chat_completion_lists_calls_and_executes_mcp_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd = register_datadog_mcp(client, resources)
        client.await_registered(dd.server_id)

        marker = f"e2e-mcp-chat-nohit-{unique_marker()}"
        key = client.generate_key(
            user_id=f"e2e-mcp-chat-{unique_marker()}",
            mcp_servers=[dd.server_id],
            models=[CHEAP_ANTHROPIC_MODEL],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        body = ChatBody(
            model=CHEAP_ANTHROPIC_MODEL,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Use the search_datadog_logs tool to search for logs "
                        f"with query '{marker}' from now-30m to now with "
                        "max_tokens 100. After you get results, reply with ok only."
                    ),
                )
            ],
            max_tokens=1024,
            tools=[
                McpChatTool(
                    type="mcp",
                    server_url=f"litellm_proxy/mcp/{dd.alias}",
                    server_label="datadog",
                    require_approval="never",
                )
            ],
        )
        response = unwrap(client.chat_with_mcp(key, body))

        assert response.choices, f"chat completion returned no choices: {response}"
        message = response.choices[0].message
        assert message is not None, f"choice had no message: {response}"

        psf = message.provider_specific_fields
        assert psf is not None, (
            "provider_specific_fields missing; the gateway did not attach MCP metadata "
            f"(mcp_list_tools / mcp_tool_calls / mcp_call_results): {message}"
        )

        assert psf.mcp_list_tools, (
            "mcp_list_tools is empty; the gateway never listed the Datadog server's tools "
            "through the chat bridge"
        )
        assert psf.mcp_tool_calls, (
            "mcp_tool_calls is empty; the model did not call any MCP tool "
            "(it may not have seen the expanded tools)"
        )
        assert psf.mcp_call_results, (
            "mcp_call_results is empty; the gateway did not execute the tool call upstream"
        )

        listed_names = {
            t.function.name
            for t in psf.mcp_list_tools
            if t.function is not None and t.function.name
        }
        called_names = {
            c.function.name
            for c in psf.mcp_tool_calls
            if c.function is not None and c.function.name
        }
        assert listed_names, f"mcp_list_tools entries had no function names: {psf.mcp_list_tools}"
        assert called_names, f"mcp_tool_calls entries had no function names: {psf.mcp_tool_calls}"
        assert called_names & listed_names, (
            f"model called {sorted(called_names)} but none appear in "
            f"mcp_list_tools {sorted(listed_names)}; the chat bridge listed and "
            f"called disjoint tool sets"
        )
        assert any("search_datadog_logs" in n for n in called_names), (
            f"expected a Datadog log-search tool call, got {sorted(called_names)}"
        )

        result_text = next(
            (r.result for r in psf.mcp_call_results if r.result), None
        )
        assert result_text, (
            "mcp_call_results has no result text; the tool call returned nothing"
        )
