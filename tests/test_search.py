"""
Unit and integration tests for Webz.io Search Provider (Issue #49565).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router import SearchAPIRouter
from litellm.search.webz import async_search as webz_async_search
from litellm.types.search import SearchProvider, SearchProviders

MOCK_WEBZ_RESPONSE = {
    "results": [
        {
            "article": {
                "title": "Global Leaders Gather for Climate Summit",
                "url": "https://news.example.com/climate-summit",
                "published_at": "2026-10-31T14:30:00Z",
            },
            "chunk": {
                "text": "World leaders met today to discuss ambitious carbon reduction targets and sustainable finance initiatives.",
            },
            "score": 0.94,
        },
        {
            "article": {
                "title": "Tech Innovations in Renewable Energy",
                "url": "https://news.example.com/tech-renewable",
                "published_at": "2026-10-30T09:15:00Z",
            },
            "chunk": {
                "text": "New breakthroughs in solar storage capacity promise cleaner grids worldwide.",
            },
            "score": 0.88,
        },
    ]
}


def test_webz_search_provider_enum():
    """Verify WEBZ is registered as a first-class SearchProviders enum member."""
    assert SearchProviders.WEBZ == "webz"
    assert SearchProviders("webz") == SearchProviders.WEBZ
    assert SearchProvider.WEBZ == "webz"
    assert "webz" in [p.value for p in SearchProviders]


def test_webz_search_request_transformation_and_headers():
    """Verify request transformation, endpoint, and Bearer token header generation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_WEBZ_RESPONSE

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = litellm.search(
            query="climate change summit",
            search_provider="webz",
            api_key="test-webz-token-123",
            max_results=7,
            country="US",
            language="en",
            domain="bbc.com",
        )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs

        # Verify URL
        assert call_kwargs["url"] == "https://api.webz.io/api/news/context"

        # Verify Headers
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-webz-token-123"
        assert headers.get("Content-Type") == "application/json"

        # Verify Payload mapping
        json_data = call_kwargs.get("json", {})
        assert json_data.get("query") == "climate change summit"
        assert json_data.get("k") == 7
        assert json_data.get("country") == "US"
        assert json_data.get("language") == "en"
        assert json_data.get("domain") == "bbc.com"

        # Verify response structure
        assert response.object == "search"
        assert len(response.results) == 2


def test_webz_search_response_mapping():
    """Verify correct response mapping to title, url, snippet, date, and score."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_WEBZ_RESPONSE

    with (
        patch.dict(os.environ, {"WEBZ_API_TOKEN": "token-from-env"}),
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ),
    ):
        response = litellm.search(
            query="renewable energy breakthroughs",
            search_provider="webz",
        )

        assert response.object == "search"
        assert len(response.results) == 2

        # Result 1 mapping
        r1 = response.results[0]
        assert r1.title == "Global Leaders Gather for Climate Summit"
        assert r1.url == "https://news.example.com/climate-summit"
        assert r1.snippet == "World leaders met today to discuss ambitious carbon reduction targets and sustainable finance initiatives."
        assert r1.date == "2026-10-31T14:30:00Z"
        assert getattr(r1, "score", None) == 0.94

        # Result 2 mapping
        r2 = response.results[1]
        assert r2.title == "Tech Innovations in Renewable Energy"
        assert r2.url == "https://news.example.com/tech-renewable"
        assert r2.snippet == "New breakthroughs in solar storage capacity promise cleaner grids worldwide."
        assert r2.date == "2026-10-30T09:15:00Z"
        assert getattr(r2, "score", None) == 0.88


def test_webz_search_env_token_resolution():
    """Verify resolution of WEBZ_API_TOKEN from environment when api_key is not passed."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}

    with patch.dict(os.environ, {"WEBZ_API_TOKEN": "secret-env-token-xyz"}), patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        litellm.search(
            query="latest headlines",
            search_provider="webz",
        )

        headers = mock_post.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer secret-env-token-xyz"


@pytest.mark.asyncio
async def test_webz_asearch_async_call():
    """Verify litellm.asearch with search_provider='webz'."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_WEBZ_RESPONSE

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = mock_response

        response = await litellm.asearch(
            query="quantum computing breakthroughs",
            search_provider="webz",
            api_key="async-test-key",
            max_results=5,
        )

        assert mock_post.called
        headers = mock_post.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer async-test-key"

        payload = mock_post.call_args.kwargs.get("json", {})
        assert payload.get("query") == "quantum computing breakthroughs"
        assert payload.get("k") == 5

        assert len(response.results) == 2
        assert response.results[0].title == "Global Leaders Gather for Climate Summit"


@pytest.mark.asyncio
async def test_webz_direct_async_search_handler():
    """Verify litellm.search.webz.async_search direct function."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_WEBZ_RESPONSE

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = mock_response

        response = await webz_async_search(
            query="breaking technology",
            api_key="direct-handler-key",
            max_results=3,
            country="GB",
            language="en",
        )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["url"] == "https://api.webz.io/api/news/context"
        assert call_kwargs["headers"]["Authorization"] == "Bearer direct-handler-key"
        assert call_kwargs["json"]["query"] == "breaking technology"
        assert call_kwargs["json"]["k"] == 3
        assert call_kwargs["json"]["country"] == "GB"
        assert call_kwargs["json"]["language"] == "en"

        assert len(response.results) == 2
        assert response.results[0].snippet == "World leaders met today to discuss ambitious carbon reduction targets and sustainable finance initiatives."


def test_search_api_router_credentials_resolution():
    """Verify SearchAPIRouter credential resolution for Webz provider."""
    with patch.dict(os.environ, {"WEBZ_API_TOKEN": "router-token-456"}):
        api_key, _ = SearchAPIRouter._resolve_search_provider_credentials(
            tool_litellm_params={"search_provider": "webz"},  # type: ignore
        )
        assert api_key == "router-token-456"

    # Explicit api_key in params takes precedence over env var
    with patch.dict(os.environ, {"WEBZ_API_TOKEN": "router-token-456"}):
        api_key, _ = SearchAPIRouter._resolve_search_provider_credentials(
            tool_litellm_params={"search_provider": "webz", "api_key": "param-token-789"},  # type: ignore
        )
        assert api_key == "param-token-789"
