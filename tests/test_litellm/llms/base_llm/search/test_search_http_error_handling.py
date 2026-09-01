"""
Tests ensuring BaseLLMHTTPHandler.search, BaseLLMHTTPHandler.async_search,
litellm.search, and litellm.asearch properly raise exceptions on non-2xx HTTP responses from search providers.
"""

from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

import litellm
from litellm.exceptions import (
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig
from litellm.llms.searxng.search.transformation import SearXNGSearchConfig
from litellm.llms.serper.search.transformation import SerperSearchConfig
from litellm.llms.tavily.search.transformation import TavilySearchConfig


class TestSearchHTTPErrorHandling:
    """Test suite verifying HTTP error handling for search providers."""

    @pytest.mark.parametrize(
        "status_code,error_body,provider,config_cls",
        [
            (401, {"error": "Invalid API Key"}, "tavily", TavilySearchConfig),
            (429, {"error": "Rate limit exceeded"}, "perplexity", PerplexitySearchConfig),
            (500, {"error": "Internal server error"}, "searxng", SearXNGSearchConfig),
            (503, {"error": "Service unavailable"}, "serper", SerperSearchConfig),
        ],
    )
    def test_sync_search_raises_on_http_error(
        self, status_code: int, error_body: dict, provider: str, config_cls
    ):
        """Verify sync search raises an exception with the corresponding status code on HTTP errors."""
        handler = BaseLLMHTTPHandler()
        config = config_cls()

        mock_client = MagicMock(spec=HTTPHandler)
        mock_response = httpx.Response(
            status_code=status_code,
            json=error_body,
            request=httpx.Request("POST", "https://example.com/search"),
            headers={"content-type": "application/json"},
        )
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_response

        logging_obj = MagicMock()

        with pytest.raises(Exception) as exc_info:
            handler.search(
                query="test query",
                optional_params={},
                timeout=10,
                logging_obj=logging_obj,
                api_key="test-key",
                api_base="https://example.com",
                custom_llm_provider=provider,
                client=mock_client,
                provider_config=config,
            )

        exc = exc_info.value
        assert (
            getattr(exc, "status_code", None) == status_code
            or getattr(exc, "code", None) == status_code
            or str(status_code) in str(exc)
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,error_body,provider,config_cls",
        [
            (401, {"error": "Unauthorized access"}, "tavily", TavilySearchConfig),
            (429, {"error": "Rate limited"}, "searxng", SearXNGSearchConfig),
            (500, {"error": "Backend crash"}, "perplexity", PerplexitySearchConfig),
            (503, {"error": "Upstream unavailable"}, "serper", SerperSearchConfig),
        ],
    )
    async def test_async_search_raises_on_http_error(
        self, status_code: int, error_body: dict, provider: str, config_cls
    ):
        """Verify async search raises an exception with the corresponding status code on HTTP errors."""
        handler = BaseLLMHTTPHandler()
        config = config_cls()

        mock_async_client = AsyncMock(spec=AsyncHTTPHandler)
        mock_response = httpx.Response(
            status_code=status_code,
            json=error_body,
            request=httpx.Request("POST", "https://example.com/search"),
            headers={"content-type": "application/json"},
        )
        mock_async_client.post.return_value = mock_response
        mock_async_client.get.return_value = mock_response

        logging_obj = MagicMock()

        with pytest.raises(Exception) as exc_info:
            await handler.async_search(
                query="test query",
                optional_params={},
                timeout=10,
                logging_obj=logging_obj,
                api_key="test-key",
                api_base="https://example.com",
                custom_llm_provider=provider,
                client=mock_async_client,
                provider_config=config,
            )

        exc = exc_info.value
        assert (
            getattr(exc, "status_code", None) == status_code
            or getattr(exc, "code", None) == status_code
            or str(status_code) in str(exc)
        )

    def test_litellm_search_top_level_exception_mapping(self):
        """Verify high-level litellm.search() maps status codes to LiteLLM exception types."""
        mock_client = MagicMock(spec=HTTPHandler)
        mock_response = httpx.Response(
            status_code=401,
            json={"error": "Invalid API Key"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            headers={"content-type": "application/json"},
        )
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_response

        with pytest.raises(AuthenticationError):
            litellm.search(
                query="test",
                search_provider="tavily",
                api_key="invalid-key",
                client=mock_client,
            )

    @pytest.mark.asyncio
    async def test_litellm_asearch_top_level_exception_mapping(self):
        """Verify high-level litellm.asearch() maps status codes to LiteLLM exception types."""
        mock_async_client = AsyncMock(spec=AsyncHTTPHandler)
        mock_response = httpx.Response(
            status_code=429,
            json={"error": "Rate limit exceeded"},
            request=httpx.Request("POST", "https://api.perplexity.ai/search"),
            headers={"content-type": "application/json"},
        )
        mock_async_client.post.return_value = mock_response
        mock_async_client.get.return_value = mock_response

        with pytest.raises(RateLimitError):
            await litellm.asearch(
                query="test",
                search_provider="perplexity",
                api_key="fake-key",
                client=mock_async_client,
            )

    def test_sync_search_success_parsing(self):
        """Verify successful 200 OK responses continue to parse correctly."""
        handler = BaseLLMHTTPHandler()
        config = TavilySearchConfig()

        mock_client = MagicMock(spec=HTTPHandler)
        mock_response = httpx.Response(
            status_code=200,
            json={
                "results": [
                    {
                        "title": "LiteLLM Docs",
                        "url": "https://docs.litellm.ai",
                        "content": "LiteLLM Documentation",
                    }
                ]
            },
            request=httpx.Request("POST", "https://api.tavily.com/search"),
            headers={"content-type": "application/json"},
        )
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_response

        logging_obj = MagicMock()

        response = handler.search(
            query="LiteLLM documentation",
            optional_params={},
            timeout=10,
            logging_obj=logging_obj,
            api_key="test-key",
            api_base="https://api.tavily.com",
            custom_llm_provider="tavily",
            client=mock_client,
            provider_config=config,
        )

        assert response.object == "search"
        assert len(response.results) == 1
        assert response.results[0].title == "LiteLLM Docs"
        assert response.results[0].url == "https://docs.litellm.ai"
