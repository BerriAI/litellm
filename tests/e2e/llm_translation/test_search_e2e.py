"""Vendor §9.14: POST /v1/search through a registered search tool (LIT-4778).

Registers a Perplexity-backed search tool at runtime, runs a basic search, and
pins missing/empty/invalid query handling.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from proxy_client import ProxyClient
from vendor_contract import assert_client_error, assert_error_or_server_known

pytestmark = pytest.mark.e2e


class SearchToolLiteLLMParams(BaseModel):
    search_provider: str
    api_key: str


class SearchToolBody(BaseModel):
    search_tool_name: str
    litellm_params: SearchToolLiteLLMParams
    search_tool_info: dict[str, str] | None = None


class CreateSearchToolRequest(BaseModel):
    search_tool: SearchToolBody


class SearchToolResponse(BaseModel):
    search_tool_id: str | None = None
    search_tool_name: str | None = None


class SearchRequest(BaseModel):
    search_tool_name: str | None = None
    query: str | None = None
    max_results: int | None = None
    country: str | None = None


class SearchResultItem(BaseModel):
    title: str | None = None
    url: str | None = None


class SearchResponse(BaseModel):
    object: str | None = None
    results: list[SearchResultItem] = []


def _register_search_tool(proxy: ProxyClient, resources: ResourceManager) -> str:
    api_key = os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("TAVILY_API_KEY")
    provider = "perplexity" if os.environ.get("PERPLEXITY_API_KEY") else "tavily"
    if not api_key:
        pytest.fail(
            "set PERPLEXITY_API_KEY or TAVILY_API_KEY for /v1/search e2e coverage"
        )
    name = f"e2e-search-{unique_marker()}"
    created = unwrap(
        proxy.transport.post(
            "/search_tools",
            headers=proxy.transport.master,
            json=CreateSearchToolRequest(
                search_tool=SearchToolBody(
                    search_tool_name=name,
                    litellm_params=SearchToolLiteLLMParams(
                        search_provider=provider, api_key=api_key
                    ),
                    search_tool_info={"description": "e2e search tool"},
                )
            ),
            response_type=SearchToolResponse,
        )
    )
    tool_id = created.search_tool_id
    if tool_id is not None:
        def _delete_tool() -> None:
            _ = proxy.transport.delete(
                f"/search_tools/{tool_id}",
                headers=proxy.transport.master,
                json=NoBody(),
                response_type=NoBody,
            )

        resources.defer(_delete_tool)
    return name


class TestSearch:
    @pytest.mark.covers("llm.search.openai.basic.nonstream.works")
    def test_basic_search_returns_results(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        tool = _register_search_tool(proxy, resources)
        key = resources.key()
        result = unwrap(
            proxy.transport.post(
                "/v1/search",
                headers=proxy.transport.bearer(key),
                json=SearchRequest(
                    search_tool_name=tool,
                    query="latest AI news",
                    max_results=3,
                    country="US",
                ),
                response_type=SearchResponse,
            )
        )
        assert result.object in (None, "search")
        assert isinstance(result.results, list), f"expected results array: {result}"

    @pytest.mark.covers("llm.search.openai.basic.nonstream.works")
    @pytest.mark.parametrize("max_results", [1, 5, 10])
    def test_max_results_boundaries(
        self, proxy: ProxyClient, resources: ResourceManager, max_results: int
    ) -> None:
        tool = _register_search_tool(proxy, resources)
        key = resources.key()
        result = unwrap(
            proxy.transport.post(
                "/v1/search",
                headers=proxy.transport.bearer(key),
                json=SearchRequest(
                    search_tool_name=tool, query="weather forecast", max_results=max_results
                ),
                response_type=SearchResponse,
            )
        )
        assert isinstance(result.results, list)

    @pytest.mark.covers("llm.search.openai.input_validation.nonstream.works")
    def test_missing_query_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        tool = _register_search_tool(proxy, resources)
        key = resources.key()
        result = proxy.transport.send(
            "/v1/search",
            headers=proxy.transport.bearer(key),
            json=SearchRequest(search_tool_name=tool, max_results=3),
        )
        assert_error_or_server_known(result, "search missing query")

    @pytest.mark.covers("llm.search.openai.input_validation.nonstream.works")
    def test_empty_query_returns_client_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        tool = _register_search_tool(proxy, resources)
        key = resources.key()
        result = proxy.transport.send(
            "/v1/search",
            headers=proxy.transport.bearer(key),
            json=SearchRequest(search_tool_name=tool, query=""),
        )
        assert_client_error(result, "search empty query")

    @pytest.mark.covers("llm.search.openai.input_validation.nonstream.works")
    def test_invalid_max_results_returns_client_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        tool = _register_search_tool(proxy, resources)
        key = resources.key()
        result = proxy.transport.send(
            "/v1/search",
            headers=proxy.transport.bearer(key),
            json=SearchRequest(search_tool_name=tool, query="tech trends", max_results=-1),
        )
        assert_client_error(result, "search invalid max_results")
