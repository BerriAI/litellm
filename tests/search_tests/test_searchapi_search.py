"""
Tests for SearchAPI.io (Google Search) integration.

Tests the SearchAPI.io search provider implementation including:
- Request transformation
- Response transformation
- Parameter mapping
- Error handling
"""

import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.searchapi.search.transformation import SearchAPIConfig
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult


class TestSearchAPIConfig:
    """Test SearchAPI.io configuration and transformations."""

    def test_ui_friendly_name(self):
        """Test that UI friendly name is returned correctly."""
        config = SearchAPIConfig()
        assert config.ui_friendly_name() == "SearchAPI.io (Google Search)"

    def test_get_http_method(self):
        """Test that HTTP method is GET."""
        config = SearchAPIConfig()
        assert config.get_http_method() == "GET"

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        """Test environment validation with API key."""
        mock_get_secret.return_value = "test_api_key"
        config = SearchAPIConfig()
        headers = {}

        result = config.validate_environment(headers, api_key="test_api_key")

        assert result["Content-Type"] == "application/json"

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_validate_environment_without_api_key(self, mock_get_secret):
        """Test environment validation without API key raises error."""
        mock_get_secret.return_value = None
        config = SearchAPIConfig()
        headers = {}

        with pytest.raises(ValueError, match="SEARCHAPI_API_KEY is not set"):
            config.validate_environment(headers)

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_transform_search_request_basic(self, mock_get_secret):
        """Test basic search request transformation."""
        mock_get_secret.return_value = "test_api_key"
        config = SearchAPIConfig()

        result = config.transform_search_request(
            query="test query", optional_params={}, api_key="test_api_key"
        )

        assert "_searchapi_params" in result
        params = result["_searchapi_params"]
        assert params["engine"] == "google"
        assert params["q"] == "test query"
        assert params["api_key"] == "test_api_key"

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_max_results(self, mock_get_secret):
        """Test search request transformation with max_results parameter."""
        mock_get_secret.return_value = "test_api_key"
        config = SearchAPIConfig()

        result = config.transform_search_request(
            query="test query",
            optional_params={"max_results": 5},
            api_key="test_api_key",
        )

        params = result["_searchapi_params"]
        assert params["num"] == 5

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_country(self, mock_get_secret):
        """Test search request transformation with country parameter."""
        mock_get_secret.return_value = "test_api_key"
        config = SearchAPIConfig()

        result = config.transform_search_request(
            query="test query",
            optional_params={"country": "US"},
            api_key="test_api_key",
        )

        params = result["_searchapi_params"]
        assert params["gl"] == "us"

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_domain_filter(self, mock_get_secret):
        """Test search request transformation with domain filter."""
        mock_get_secret.return_value = "test_api_key"
        config = SearchAPIConfig()

        result = config.transform_search_request(
            query="test query",
            optional_params={"search_domain_filter": ["example.com", "test.com"]},
            api_key="test_api_key",
        )

        params = result["_searchapi_params"]
        assert "site:example.com" in params["q"]
        assert "site:test.com" in params["q"]

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_transform_search_request_with_list_query(self, mock_get_secret):
        """Test search request transformation with list query."""
        mock_get_secret.return_value = "test_api_key"
        config = SearchAPIConfig()

        result = config.transform_search_request(
            query=["test", "query"], optional_params={}, api_key="test_api_key"
        )

        params = result["_searchapi_params"]
        assert params["q"] == "test query"

    @patch("litellm.llms.searchapi.search.transformation.get_secret_str")
    def test_get_complete_url(self, mock_get_secret):
        """Test URL construction with query parameters."""
        mock_get_secret.return_value = None
        config = SearchAPIConfig()

        data = {
            "_searchapi_params": {
                "engine": "google",
                "q": "test query",
                "api_key": "test_key",
            }
        }

        url = config.get_complete_url(api_base=None, optional_params={}, data=data)

        assert "https://www.searchapi.io/api/v1/search?" in url
        assert "engine=google" in url
        assert "q=test+query" in url
        assert "api_key=test_key" in url

    def test_transform_search_response(self):
        """Test search response transformation."""
        config = SearchAPIConfig()

        # Mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "title": "Test Result 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a test snippet 1",
                    "date": "2024-01-01",
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "This is a test snippet 2",
                },
            ]
        }

        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )

        assert isinstance(result, SearchResponse)
        assert result.object == "search"
        assert len(result.results) == 2

        # Check first result
        assert result.results[0].title == "Test Result 1"
        assert result.results[0].url == "https://example.com/1"
        assert result.results[0].snippet == "This is a test snippet 1"
        assert result.results[0].date == "2024-01-01"
        assert result.results[0].last_updated is None

        # Check second result
        assert result.results[1].title == "Test Result 2"
        assert result.results[1].url == "https://example.com/2"
        assert result.results[1].snippet == "This is a test snippet 2"
        assert result.results[1].date is None

    def test_transform_search_response_empty(self):
        """Test search response transformation with no results."""
        config = SearchAPIConfig()

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"organic_results": []}

        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0

    def test_append_domain_filters(self):
        """Test domain filter appending logic."""
        config = SearchAPIConfig()

        query = "test query"
        domains = ["example.com", "test.com"]

        result = config._append_domain_filters(query, domains)

        assert "(test query)" in result
        assert "site:example.com" in result
        assert "site:test.com" in result
        assert "OR" in result
        assert "AND" in result


@pytest.mark.skipif(
    os.environ.get("SEARCHAPI_API_KEY") is None,
    reason="SEARCHAPI_API_KEY not set in environment",
)
class TestSearchAPIIntegration:
    """Integration tests for SearchAPI.io (requires API key)."""

    def test_real_search_request(self):
        """
        Test a real search request to SearchAPI.io.
        This test is skipped if SEARCHAPI_API_KEY is not set.
        """
        import litellm

        response = litellm.search(
            query="Python programming", search_provider="searchapi", max_results=5
        )

        assert response is not None
        assert hasattr(response, "results")
        assert len(response.results) > 0
        assert all(hasattr(r, "title") for r in response.results)
        assert all(hasattr(r, "url") for r in response.results)
        assert all(hasattr(r, "snippet") for r in response.results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
