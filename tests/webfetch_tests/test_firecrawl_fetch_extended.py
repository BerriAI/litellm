"""Tests for Firecrawl Fetch Transformation.

Covers FirecrawlFetchConfig scrape/fetch logic and self-hosted support.
All network calls are mocked with httpx + respx.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from litellm.llms.firecrawl.fetch.transformation import FirecrawlFetchConfig
from litellm.llms.base_llm.fetch.transformation import WebFetchResponse


class TestFirecrawlFetchConfig:
    """Test FirecrawlFetchConfig initialization and methods."""

    def test_ui_friendly_name(self):
        """Test ui_friendly_name returns correct value."""
        config = FirecrawlFetchConfig()
        assert config.ui_friendly_name() == "Firecrawl"

    def test_default_api_base(self):
        """Test default API base is Firecrawl Cloud."""
        config = FirecrawlFetchConfig()
        assert "api.firecrawl.dev" in config.api_base

    def test_custom_api_base(self):
        """Test custom api_base overrides default."""
        config = FirecrawlFetchConfig(api_base="http://localhost:3002")
        assert config.api_base == "http://localhost:3002"

    def test_validate_environment_no_api_key(self):
        """Test validation without API key raises ValueError."""
        config = FirecrawlFetchConfig()
        with pytest.raises(ValueError, match="api_key"):
            config.validate_environment(headers={})

    def test_validate_environment_with_api_key(self):
        """Test validation with API key succeeds."""
        config = FirecrawlFetchConfig()
        headers = {"Authorization": "Bearer fc-test"}
        result = config.validate_environment(headers=headers, api_key="fc-test")
        assert "Authorization" in result

    def test_scrape_url_sync_success(self):
        """Test synchronous scrape URL success."""
        config = FirecrawlFetchConfig(api_key="fc-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "# Hello\nWorld",
                "metadata": {
                    "title": "Hello Page",
                    "sourceURL": "https://example.com",
                }
            }
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        result = config.scrape_url(
            url="https://example.com",
            headers={},
            client=mock_client,
        )

        assert isinstance(result, WebFetchResponse)
        assert result.url == "https://example.com"
        assert "Hello" in result.title

    def test_scrape_url_sync_error(self):
        """Test synchronous scrape URL error handling."""
        config = FirecrawlFetchConfig(api_key="fc-test")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with pytest.raises(ValueError, match="scrape URL"):
            config.scrape_url(
                url="https://example.com",
                headers={},
                client=mock_client,
            )

    def test_scrape_url_sync_no_markdown(self):
        """Test scrape URL with no markdown in response uses html."""
        config = FirecrawlFetchConfig(api_key="fc-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "html": "<html><body>Hello</body></html>",
                "metadata": {
                    "title": "Hello",
                    "sourceURL": "https://example.com",
                }
            }
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        result = config.scrape_url(
            url="https://example.com",
            headers={},
            client=mock_client,
        )

        assert "Hello" in result.content


class TestFetchUrlAsync:
    """Test async fetch_url method."""

    @pytest.mark.asyncio
    async def test_afetch_url_success(self):
        """Test async fetch URL success."""
        config = FirecrawlFetchConfig(api_key="fc-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value={
            "success": True,
            "data": {
                "markdown": "Async Hello",
                "metadata": {
                    "title": "Async Hello",
                    "sourceURL": "https://example.com",
                }
            }
        })

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await config.afetch_url(
            url="https://example.com",
            headers={},
            client=mock_client,
        )

        assert isinstance(result, WebFetchResponse)
        assert result.content == "Async Hello"

    @pytest.mark.asyncio
    async def test_afetch_url_error(self):
        """Test async fetch URL error."""
        config = FirecrawlFetchConfig(api_key="fc-test")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="scrape URL"):
            await config.afetch_url(
                url="https://example.com",
                headers={},
                client=mock_client,
            )


class TestSelfHosted:
    """Test self-hosted Firecrawl instance support."""

    def test_self_hosted_api_base(self):
        """Test setting custom api_base for self-hosted instance."""
        config = FirecrawlFetchConfig(
            api_base="http://localhost:3002",
            api_key="local-key",
        )
        assert "localhost:3002" in config.api_base

    def test_self_hosted_request_url(self):
        """Test request URL uses custom api_base."""
        config = FirecrawlFetchConfig(
            api_base="http://my-firecrawl.internal:3002",
            api_key="local-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "Local content",
                "metadata": {
                    "title": "Local",
                    "sourceURL": "https://example.com",
                }
            }
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        result = config.scrape_url(
            url="https://example.com",
            headers={},
            client=mock_client,
        )

        # Verify the URL used in the request
        call_args = mock_client.post.call_args
        assert "my-firecrawl.internal:3002" in call_args[0][0] or \
               "my-firecrawl.internal:3002" in call_args[1].get("url", "")
