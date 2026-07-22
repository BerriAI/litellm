"""Live e2e: the proxy brokers the real Datadog remote MCP server.

Seeds a chat completion whose prompt carries a unique `e2e-datadog-mcp-*`
marker so the proxy's DataDogLogger ships a StandardLoggingPayload the org can
search. Registers the regional Datadog MCP endpoint with DD_API_KEY /
DD_APP_KEY as static headers (Datadog's documented CI/header auth). A key
granted that server lists tools, calls search_datadog_logs for the marker, and
the response must contain it. The dual read via datadog_reader proves the log
is also in the Logs Search API. The MCP server row is deleted on teardown.
"""

from __future__ import annotations

import pytest

from conftest import DdLogsReader
from datadog_mcp import SEARCH_LOGS_TOOL, assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import CHEAP_ANTHROPIC_MODEL, DD_SEARCH_FROM, unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import ChatBody, ChatMessage
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

DD_LOGGER_NAME = "DataDogLogger"
MARKER_PREFIX = "e2e-datadog-mcp-"


def _assert_datadog_logger_active(proxy: ProxyClient) -> None:
    result = proxy.probe("/health/readiness/details", params=NoBody())
    assert result.status_code == 200, (
        f"/health/readiness/details must answer 200, got {result.status_code}: {result.body[:300]}"
    )
    assert DD_LOGGER_NAME in result.body, (
        f"the proxy must report the {DD_LOGGER_NAME} callback active "
        f"(callbacks + DD_* env); got: {result.body[:400]}"
    )


def _seed_completion(proxy: ProxyClient, *, key: str, marker: str) -> None:
    body = ChatBody(
        model=CHEAP_ANTHROPIC_MODEL,
        messages=[ChatMessage(role="user", content=f"reply with one word {marker}")],
        max_tokens=16,
    )
    unwrap(proxy.chat(key, body))


class TestDatadogMcpRoundTrip:
    @pytest.mark.covers("mcp.list_tools.api_key.succeeds", "mcp.call_tool.api_key.succeeds")
    def test_search_logs_finds_seeded_completion(
        self,
        client: McpClient,
        dd_logs: DdLogsReader,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        _assert_datadog_logger_active(client.proxy)

        server_id = register_datadog_mcp(client, resources)
        marker = f"{MARKER_PREFIX}{unique_marker()}"

        key = client.generate_key(
            user_id=f"e2e-dd-mcp-{unique_marker()}",
            mcp_servers=[server_id],
            models=[CHEAP_ANTHROPIC_MODEL],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        _seed_completion(client.proxy, key=key, marker=marker)

        shipped = dd_logs.poll_events_for_marker(marker)
        assert shipped, (
            f"proxy DataDogLogger never shipped a log containing {marker!r} "
            "within the poll deadline; MCP search would have nothing to find"
        )

        tools = unwrap(client.list_tools(key))
        tool_name = tools.tool_name_containing(server_id, SEARCH_LOGS_TOOL)
        assert tool_name is not None, (
            f"granted key never saw {SEARCH_LOGS_TOOL} on server {server_id}; "
            f"tools={tools.tool_names_for_server(server_id)}"
        )

        call = unwrap(
            client.call_tool(
                key,
                server_id=server_id,
                name=tool_name,
                arguments={
                    "query": marker,
                    "from": DD_SEARCH_FROM,
                    "to": "now",
                    "max_tokens": 5000,
                    "telemetry": {
                        "intent": "e2e assert seeded litellm completion log is searchable via MCP"
                    },
                },
            )
        )
        assert call.is_error is not True, f"search_datadog_logs errored: {call}"
        body = call.all_text
        assert marker in body, (
            f"search_datadog_logs response must include the seeded marker {marker!r}; "
            f"got: {body[:800]!r}"
        )
