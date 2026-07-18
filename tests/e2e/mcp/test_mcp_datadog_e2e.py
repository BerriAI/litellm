"""Live e2e: the proxy brokers the real Datadog remote MCP server.

Seeds a chat completion whose prompt carries a unique `e2e-datadog-mcp-*`
marker so the proxy's DataDogLogger ships a StandardLoggingPayload the org can
search. Registers https://mcp.<site>/v1/mcp with the stage DD_API_KEY /
DD_APP_KEY as static headers (Datadog's documented CI/header auth; the browser
OAuth authorize/token flow is not headless-automatable). A key granted that
server lists tools, calls search_datadog_logs for the marker, and the response
must contain it. The dual read via datadog_reader proves the log is also in
the Logs Search API. The MCP server row is deleted on teardown.
"""

from __future__ import annotations

import pytest

from e2e_config import (
    CHEAP_ANTHROPIC_MODEL,
    DD_API_KEY,
    DD_APP_KEY,
    DD_SEARCH_FROM,
    datadog_mcp_url,
    unique_marker,
)
from e2e_http import NoBody, unwrap
from e2e_gateway import Gateway
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage
from mcp_client import McpClient
from conftest import DdLogsReader

pytestmark = pytest.mark.e2e

DD_LOGGER_NAME = "DataDogLogger"
SEARCH_LOGS_TOOL = "search_datadog_logs"
MARKER_PREFIX = "e2e-datadog-mcp-"


def _assert_datadog_logger_active(gateway: Gateway) -> None:
    result = gateway.probe("/health/readiness/details", params=NoBody())
    assert result.status_code == 200, (
        f"/health/readiness/details must answer 200, got {result.status_code}: {result.body[:300]}"
    )
    assert DD_LOGGER_NAME in result.body, (
        f"the proxy must report the {DD_LOGGER_NAME} callback active "
        f"(callbacks + DD_* env); got: {result.body[:400]}"
    )


def _assert_dd_mcp_creds() -> None:
    if not DD_API_KEY or not DD_APP_KEY:
        pytest.fail(
            "Datadog MCP e2e requires DD_API_KEY and DD_APP_KEY "
            "(header auth to mcp.<site>/v1/mcp; on the cluster the secret manager "
            "injects them, locally tests/e2e/.env)"
        )


def _seed_completion(gateway: Gateway, *, key: str, marker: str) -> None:
    body = ChatBody(
        model=CHEAP_ANTHROPIC_MODEL,
        messages=[ChatMessage(role="user", content=f"reply with one word {marker}")],
        max_tokens=16,
    )
    unwrap(gateway.chat(key, body))


def _register_datadog_mcp(client: McpClient, resources: ResourceManager) -> str:
    name = f"e2e_dd_mcp_{unique_marker()}"
    server_id = client.register_server(
        server_name=name,
        alias=name,
        url=datadog_mcp_url(toolsets="core"),
        transport="http",
        static_headers={
            "DD-API-KEY": DD_API_KEY,
            "DD-APPLICATION-KEY": DD_APP_KEY,
        },
        allowed_tools=[SEARCH_LOGS_TOOL],
    )
    resources.defer(lambda: client.delete_server(server_id))
    return server_id


class TestDatadogMcpRoundTrip:
    @pytest.mark.covers("mcp.list_tools.api_key.succeeds", "mcp.call_tool.api_key.succeeds")
    def test_search_logs_finds_seeded_completion(
        self,
        client: McpClient,
        dd_logs: DdLogsReader,
        resources: ResourceManager,
    ) -> None:
        _assert_dd_mcp_creds()
        _assert_datadog_logger_active(client.gateway)

        server_id = _register_datadog_mcp(client, resources)
        marker = f"{MARKER_PREFIX}{unique_marker()}"

        key = client.generate_key(
            user_id=f"e2e-dd-mcp-{unique_marker()}",
            mcp_servers=[server_id],
            models=[CHEAP_ANTHROPIC_MODEL],
        )
        resources.defer(lambda: client.gateway.delete_key(key))

        _seed_completion(client.gateway, key=key, marker=marker)

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
