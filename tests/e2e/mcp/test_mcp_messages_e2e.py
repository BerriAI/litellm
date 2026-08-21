"""Live e2e: /v1/messages expands a gateway-registered MCP server and
auto-executes its tools, running the tool_use loop internally and returning a
final Anthropic response (the Claude Code path).

Registers the real Datadog remote MCP server, grants a key access to it, then
sends a /v1/messages request whose ``tools`` array carries an
``{type: "mcp", server_url: "litellm_proxy/mcp/<alias>"}`` reference. The
gateway intercepts the litellm_proxy reference (which Anthropic cannot reach),
expands it into native Anthropic custom tools under the caller's credentials,
runs the tool_use loop (model calls search_datadog_logs, gateway executes it,
feeds the result back as a tool_result), and returns the final answer. The
response has no MCP-specific metadata; the proof is the final text answer,
meaning the loop completed and the model used the tool result.

The streaming variant exercises the same loop but with stream=True, where the
gateway runs the tool_use loop non-streaming internally and then fakes a stream
of the final answer as Anthropic SSE events.
"""

from __future__ import annotations

import pytest

from datadog_mcp import assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import require_successful_call, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import AnthropicMcpTool, AnthropicMessagesBody, ChatMessage

pytestmark = pytest.mark.e2e

def _tool_prompt(marker: str) -> str:
    return (
        "Use the search_datadog_logs tool to search for logs "
        f"with query '{marker}' from now-30m to now with "
        "max_tokens 100. After you get results, reply with ok only."
    )


def _messages_body(model: str, alias: str, marker: str) -> AnthropicMessagesBody:
    return AnthropicMessagesBody(
        model=model,
        max_tokens=1024,
        messages=[ChatMessage(role="user", content=_tool_prompt(marker))],
        tools=[
            AnthropicMcpTool(
                server_label="datadog",
                server_url=f"litellm_proxy/mcp/{alias}",
                require_approval="never",
            )
        ],
    )


class TestMessagesMcpAutoExecute:
    @pytest.mark.covers("mcp.messages.api_key.auto_executes_tools")
    def test_messages_runs_tool_loop_and_returns_final_answer(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd = register_datadog_mcp(client, resources)
        client.await_registered(dd.server_id)

        marker = f"e2e-mcp-msg-nohit-{unique_marker()}"
        key = client.generate_key(
            user_id=f"e2e-mcp-msg-{unique_marker()}",
            mcp_servers=[dd.server_id],
            models=[CHEAP_ANTHROPIC_MODEL],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        response = unwrap(
            client.messages_with_mcp(
                key, _messages_body(CHEAP_ANTHROPIC_MODEL, dd.alias, marker)
            )
        )

        assert response.content, f"/v1/messages returned no content blocks: {response}"
        text = "".join(block.text or "" for block in response.content)
        assert text.strip(), (
            f"/v1/messages returned no text after the MCP tool loop; the gateway "
            f"may not have completed the tool_use loop: {response}"
        )

    @pytest.mark.covers("mcp.messages.api_key.stream_auto_executes_tools")
    def test_messages_stream_runs_tool_loop_and_returns_final_answer(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd = register_datadog_mcp(client, resources)
        client.await_registered(dd.server_id)

        marker = f"e2e-mcp-msg-stream-nohit-{unique_marker()}"
        key = client.generate_key(
            user_id=f"e2e-mcp-msg-stream-{unique_marker()}",
            mcp_servers=[dd.server_id],
            models=[CHEAP_ANTHROPIC_MODEL],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        body = _messages_body(CHEAP_ANTHROPIC_MODEL, dd.alias, marker)
        body.stream = True

        result = client.messages_stream_with_mcp(key, body)
        require_successful_call(result)
        assert result.is_streaming, f"response was not streamed: {result.headers}"
        assert not result.stream_error, f"stream errored: {result.stream_error}"
        assert result.stream_events, "stream produced no SSE events"
        assert any("content_block_delta" in event for event in result.stream_events), (
            "stream carried no content deltas"
        )
        assert any("message_stop" in event for event in result.stream_events), (
            "stream never reached message_stop"
        )
