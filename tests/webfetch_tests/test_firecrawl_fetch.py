"""
WebFetch tests for Firecrawl fetch provider.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from litellm.llms.firecrawl.fetch.transformation import FirecrawlFetchConfig
from litellm.llms.base_llm.fetch.transformation import WebFetchResponse


class TestFirecrawlFetchConfig:
    """Test Firecrawl fetch configuration."""

    @pytest.fixture
    def fetch_config(self):
        return FirecrawlFetchConfig()

    def test_validate_environment_with_api_key(self, fetch_config):
        """Test that API key validation works."""
        headers = {}
        result = fetch_config.validate_environment(headers, api_key="test-key")
        assert result["Authorization"] == "Bearer test-key"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_without_api_key(self, fetch_config):
        """Test that missing API key raises ValueError."""
        headers = {}
        with patch(
            "litellm.llms.firecrawl.fetch.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="FIRECRAWL_API_KEY is not set"):
                fetch_config.validate_environment(headers)

    def test_get_complete_url_default(self, fetch_config):
        """Test default API base URL."""
        url = fetch_config.get_complete_url()
        assert url == "https://api.firecrawl.dev/v1/scrape"

    def test_get_complete_url_custom(self, fetch_config):
        """Test custom API base URL."""
        url = fetch_config.get_complete_url(api_base="https://custom.firecrawl.dev")
        assert url == "https://custom.firecrawl.dev/scrape"

    @pytest.mark.asyncio
    async def test_afetch_url_success(self, fetch_config):
        """Test successful fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "markdown": "# Test Content",
                "metadata": {"title": "Test Title"},
            }
        }
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "litellm.llms.firecrawl.fetch.transformation.get_async_httpx_client",
            return_value=mock_client,
        ):
            result = await fetch_config.afetch_url(
                url="https://example.com",
                headers={"Authorization": "Bearer test-key"},
                optional_params={},
            )

        assert isinstance(result, WebFetchResponse)
        assert result.url == "https://example.com"
        assert result.title == "Test Title"
        assert result.content == "# Test Content"

    @pytest.mark.asyncio
    async def test_afetch_url_error(self, fetch_config):
        """Test fetch with error response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "litellm.llms.firecrawl.fetch.transformation.get_async_httpx_client",
            return_value=mock_client,
        ):
            with pytest.raises(Exception, match="Firecrawl fetch failed"):
                await fetch_config.afetch_url(
                    url="https://example.com",
                    headers={"Authorization": "Bearer test-key"},
                    optional_params={},
                )


class TestWebFetchResponse:
    """Test WebFetchResponse data model."""

    def test_basic_response(self):
        """Test creating a basic response."""
        response = WebFetchResponse(
            url="https://example.com",
            content="Test content",
        )
        assert response.url == "https://example.com"
        assert response.content == "Test content"
        assert response.title is None
        assert response.metadata is None

    def test_full_response(self):
        """Test creating a response with all fields."""
        response = WebFetchResponse(
            url="https://example.com",
            title="Test Title",
            content="Test content",
            metadata={"author": "Test"},
        )
        assert response.title == "Test Title"
        assert response.metadata == {"author": "Test"}
