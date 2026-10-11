"""Live e2e: the MCP semantic tool filter narrows a Datadog catalog to the tools the prompt needs.

An admin registers the real Datadog remote MCP server with its whole core toolset, adds an
embedding deployment, and turns on `litellm_settings.mcp_semantic_tool_filter` through
/config/update. A chat completion that attaches the server and asks to search logs must report,
in the x-litellm-semantic-filter headers, that the catalog was cut to at most top_k tools and
that `search_datadog_logs` survived the cut. Requires DD_API_KEY + DD_APP_KEY and the proxy's
OPENAI_API_KEY for the embedding deployment.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import CHEAP_OPENAI_MODEL, POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import StreamingResponse, unwrap
from e2e_metadata import Domain, Route, Subject, meta
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import ChatBody, ChatMessage, LiteLLMParamsBody, McpChatTool
from pydantic import BaseModel

pytestmark = [pytest.mark.e2e, pytest.mark.quiet_stack]

TOP_K: Final = 3


class SemanticToolFilterSettings(BaseModel):
    enabled: bool
    embedding_model: str | None = None
    top_k: int | None = None
    similarity_threshold: float | None = None


class SemanticFilterLitellmSettings(BaseModel):
    mcp_semantic_tool_filter: SemanticToolFilterSettings


class SemanticFilterConfigUpdateBody(BaseModel):
    litellm_settings: SemanticFilterLitellmSettings


class ConfigUpdateResponse(BaseModel):
    message: str


def _write_filter(client: McpClient, settings: SemanticToolFilterSettings) -> None:
    response: Final = unwrap(
        client.proxy.transport.post(
            "/config/update",
            headers=client.proxy.transport.master,
            json=SemanticFilterConfigUpdateBody(
                litellm_settings=SemanticFilterLitellmSettings(mcp_semantic_tool_filter=settings)
            ),
            response_type=ConfigUpdateResponse,
        )
    )
    assert "success" in response.message.lower(), f"/config/update reported {response.message!r}"


def _filtered_chat(client: McpClient, key: str, alias: str) -> StreamingResponse:
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    body: Final = ChatBody(
        model=CHEAP_OPENAI_MODEL,
        messages=[ChatMessage(role="user", content=f"Search the Datadog logs for the marker {unique_marker()}")],
        tools=[McpChatTool(server_url=f"litellm_proxy/mcp/{alias}", server_label=alias, require_approval="never")],
    )
    while True:
        response: StreamingResponse = client.proxy.chat_stream(key, body)  # rebind-ok: poll until hot reload lands
        assert response.ok, (
            f"chat completion with the Datadog MCP server failed: {response.status_code} {response.body}"
        )
        if "x-litellm-semantic-filter" in response.headers or time.monotonic() > deadline:
            return response
        time.sleep(POLL_INTERVAL)


class TestMcpSemanticToolFilter:
    @pytest.mark.covers("mcp.list_tools.api_key.semantic_filtered")
    @meta(Subject(domain=Domain.MCP, route=Route.MCP))
    def test_filter_keeps_the_tool_the_prompt_needs(self, client: McpClient, resources: ResourceManager) -> None:
        server_id: Final = register_datadog_mcp(client, resources, allowed_tools=None)
        row: Final = client.await_registered(server_id)
        alias: Final = row.server_name
        assert alias, f"the registered Datadog server has no alias to attach it by: {row}"

        embedding_model: Final = f"e2e-semantic-filter-embedding-{unique_marker()}"
        model_id: Final = client.proxy.create_model(
            embedding_model,
            LiteLLMParamsBody(model="openai/text-embedding-3-small", api_key="os.environ/OPENAI_API_KEY"),
            provider_live=True,
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        key: Final = client.generate_key(user_id=f"e2e-mcp-semantic-{unique_marker()}", mcp_servers=[server_id])
        resources.defer(lambda: client.proxy.delete_key(key))
        catalog: Final = unwrap(client.list_tools(key)).tool_names_for_server(server_id)
        assert len(catalog) > TOP_K, (
            f"the Datadog core toolset must serve more than {TOP_K} tools, got {sorted(catalog)}"
        )

        resources.defer(lambda: _write_filter(client, SemanticToolFilterSettings(enabled=False)))
        _write_filter(
            client,
            SemanticToolFilterSettings(
                enabled=True, embedding_model=embedding_model, top_k=TOP_K, similarity_threshold=0.0
            ),
        )

        response: Final = _filtered_chat(client, key, alias)
        stats: Final = response.headers.get("x-litellm-semantic-filter")
        assert stats is not None, f"no x-litellm-semantic-filter header within {POLL_TIMEOUT}s: {response.headers}"
        before, _, after = stats.partition("->")
        assert int(before) == len(catalog), f"the filter saw {before} MCP tools, the key lists {len(catalog)}"
        assert 0 < int(after) <= TOP_K, f"the filter kept {after} tools, expected between 1 and {TOP_K}"
        kept: Final = frozenset(response.headers.get("x-litellm-semantic-filter-tools", "").split(","))
        assert any(name.endswith(SEARCH_LOGS_TOOL) for name in kept), (
            f"a prompt asking to search logs must keep {SEARCH_LOGS_TOOL}, the filter kept {sorted(kept)}"
        )
