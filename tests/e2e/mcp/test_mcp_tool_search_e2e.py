from __future__ import annotations

from typing import Final

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import provider_edge_base, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient, McpToolSearchSettings
from models import LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

MCP_TOOL_SEARCH_TOOL_NAME: Final = "mcp_tool_search"
OVERSIZED_DESCRIPTION: Final = f"OVERSIZED {'weather ' * 12_000}"


class TestMcpToolSearch:
    @pytest.mark.covers("mcp.tool_search.api_key.oversized_description_omitted")
    def test_native_tool_search_omits_oversized_description(
        self, proxy: ProxyClient, client: McpClient, resources: ResourceManager
    ) -> None:
        """Exercise shared ranking through the native virtual tool because proxy mode does not apply description overrides."""
        model_name: Final = f"e2e-mcp-tool-search-{unique_marker()}"
        embedding_api_base: Final = provider_edge_base("openai")
        model_id: Final = proxy.create_model(
            model_name,
            LiteLLMParamsBody(
                model="openai/text-embedding-3-large",
                api_key="os.environ/OPENAI_API_KEY",
                api_base=None if embedding_api_base is None else f"{embedding_api_base}/v1",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        previous_settings: Final = client.get_mcp_tool_search_settings()
        resources.defer(lambda: client.update_mcp_tool_search_settings(previous_settings))
        client.update_mcp_tool_search_settings(
            McpToolSearchSettings(
                embedding_model=model_name,
                top_k=100,
                similarity_threshold=0.0,
                core_tools=previous_settings.core_tools,
            )
        )

        server_id: Final = register_datadog_mcp(
            client,
            resources,
            allowed_tools=None,
            tool_name_to_description={SEARCH_LOGS_TOOL: OVERSIZED_DESCRIPTION},
        )
        client.await_registered(server_id)
        key: Final = client.generate_key(
            user_id=f"e2e-mcp-tool-search-{unique_marker()}",
            mcp_servers=[server_id],
            models=[model_name],
            mcp_tool_search_enabled=True,
        )
        resources.defer(lambda: proxy.delete_key(key))

        tools: Final = unwrap(client.list_tools(key, server_id=server_id))
        assert len(tools.tool_names_for_server(server_id)) >= 2
        oversized_tool_name: Final = tools.tool_name_containing(server_id, SEARCH_LOGS_TOOL)
        assert oversized_tool_name is not None
        oversized_tool: Final = next((tool for tool in tools.tools if tool.name == oversized_tool_name), None)
        assert oversized_tool is not None
        assert oversized_tool.description == OVERSIZED_DESCRIPTION
        normal_tool_name: Final = next(
            (name for name in tools.tool_names_for_server(server_id) if name != oversized_tool_name), None
        )
        assert normal_tool_name is not None

        result: Final = unwrap(
            client.call_virtual_tool(
                key,
                name=MCP_TOOL_SEARCH_TOOL_NAME,
                arguments={"query": normal_tool_name, "top_k": 100},
            )
        )

        assert result.is_error is not True, result.all_text
        assert normal_tool_name in result.all_text
        assert oversized_tool_name not in result.all_text
