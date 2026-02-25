"""
E2E tests for Search API endpoints on the LiteLLM Proxy.

Tests the /v1/search endpoint with Perplexity as the search provider,
validating the full request flow through the proxy server.
"""

import os
import pytest
import httpx

PROXY_BASE_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")
SEARCH_TOOL_NAME = "perplexity-search"


@pytest.fixture(scope="module")
def client():
    return httpx.Client(
        base_url=PROXY_BASE_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


@pytest.fixture(scope="module")
def async_client():
    return httpx.AsyncClient(
        base_url=PROXY_BASE_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


class TestSearchAPIHealthAndTools:
    """Verify the proxy is up and search tools are registered."""

    def test_proxy_health(self, client):
        resp = client.get("/health/liveliness")
        assert resp.status_code == 200

    def test_list_search_tools(self, client):
        resp = client.get("/v1/search/tools")
        assert resp.status_code == 200

        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0

        tool_names = [t["search_tool_name"] for t in body["data"]]
        assert SEARCH_TOOL_NAME in tool_names, (
            f"Expected '{SEARCH_TOOL_NAME}' in search tools, got {tool_names}"
        )

        tool = next(t for t in body["data"] if t["search_tool_name"] == SEARCH_TOOL_NAME)
        assert tool["search_provider"] == "perplexity"


class TestSearchEndpointWithPathParam:
    """Test /v1/search/{search_tool_name} (tool name in URL path)."""

    def test_basic_search(self, client):
        """Search with a simple query and validate response structure."""
        resp = client.post(
            f"/v1/search/{SEARCH_TOOL_NAME}",
            json={"query": "What is LiteLLM?"},
        )
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"

        body = resp.json()
        assert body["object"] == "search"
        assert isinstance(body["results"], list)
        assert len(body["results"]) > 0

        first = body["results"][0]
        assert "title" in first
        assert "url" in first
        assert "snippet" in first
        assert len(first["title"]) > 0
        assert len(first["url"]) > 0
        assert len(first["snippet"]) > 0

    def test_search_with_max_results(self, client):
        """Verify max_results parameter is respected."""
        resp = client.post(
            f"/v1/search/{SEARCH_TOOL_NAME}",
            json={
                "query": "latest developments in artificial intelligence",
                "max_results": 3,
            },
        )
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"

        body = resp.json()
        assert body["object"] == "search"
        assert isinstance(body["results"], list)
        assert len(body["results"]) > 0
        assert len(body["results"]) <= 3


class TestSearchEndpointWithBodyParam:
    """Test /v1/search (tool name in request body)."""

    def test_search_with_tool_name_in_body(self, client):
        """Pass search_tool_name inside the JSON body instead of URL."""
        resp = client.post(
            "/v1/search",
            json={
                "search_tool_name": SEARCH_TOOL_NAME,
                "query": "What is a large language model?",
            },
        )
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"

        body = resp.json()
        assert body["object"] == "search"
        assert isinstance(body["results"], list)
        assert len(body["results"]) > 0


class TestSearchEndpointAsync:
    """Async variants to validate concurrency path."""

    @pytest.mark.asyncio
    async def test_async_search(self, async_client):
        resp = await async_client.post(
            f"/v1/search/{SEARCH_TOOL_NAME}",
            json={"query": "OpenAI GPT models"},
        )
        assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"

        body = resp.json()
        assert body["object"] == "search"
        assert len(body["results"]) > 0


class TestSearchEndpointErrorCases:
    """Negative / edge-case tests."""

    def test_missing_query(self, client):
        """Omitting the required query field should return an error."""
        resp = client.post(
            f"/v1/search/{SEARCH_TOOL_NAME}",
            json={},
        )
        assert resp.status_code >= 400

    def test_unauthenticated_request(self):
        """Request without auth header should be rejected."""
        unauthed = httpx.Client(base_url=PROXY_BASE_URL, timeout=10.0)
        resp = unauthed.post(
            f"/v1/search/{SEARCH_TOOL_NAME}",
            json={"query": "test"},
        )
        assert resp.status_code == 401
