"""
Unit tests for SearXNG Search request/response transformation.

These tests validate the request payload and response parsing without
requiring a live SearXNG instance.
"""

import json
import os
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from litellm.llms.searxng.search.transformation import SearXNGSearchConfig


class TestSearXNGSearchRequestTransformation:
    """
    Tests that SearXNG search requests are transformed into the expected payload.
    """

    def setup_method(self):
        self.config = SearXNGSearchConfig()

    def test_basic_query_request(self):
        """Test that a basic query produces the expected SearXNG request params."""
        result = self.config.transform_search_request(
            query="artificial intelligence recent news",
            optional_params={},
        )

        assert "_searxng_params" in result
        params = result["_searxng_params"]
        assert params["q"] == "artificial intelligence recent news"
        assert params["format"] == "json"

    def test_list_query_joined(self):
        """Test that a list query is joined into a single string."""
        result = self.config.transform_search_request(
            query=["artificial intelligence", "recent news"],
            optional_params={},
        )

        params = result["_searxng_params"]
        assert params["q"] == "artificial intelligence recent news"
        assert params["format"] == "json"

    def test_country_to_language_mapping(self):
        """Test that country codes are mapped to SearXNG language params."""
        test_cases = {
            "us": "en",
            "uk": "en",
            "de": "de",
            "fr": "fr",
            "es": "es",
            "jp": "ja",
            "br": "br",  # unmapped country passed through as-is
        }
        for country, expected_language in test_cases.items():
            result = self.config.transform_search_request(
                query="test",
                optional_params={"country": country},
            )
            params = result["_searxng_params"]
            assert (
                params["language"] == expected_language
            ), f"country={country} should map to language={expected_language}"

    def test_max_results_ignored(self):
        """Test that max_results is accepted but doesn't add extra params."""
        result = self.config.transform_search_request(
            query="test",
            optional_params={"max_results": 5},
        )

        params = result["_searxng_params"]
        assert params["q"] == "test"
        assert params["format"] == "json"
        # max_results should not appear in the SearXNG params
        assert "max_results" not in params

    def test_searxng_specific_params_passthrough(self):
        """Test that SearXNG-specific params are passed through as-is."""
        result = self.config.transform_search_request(
            query="test",
            optional_params={
                "categories": "general,news",
                "engines": "google,bing",
                "time_range": "month",
            },
        )

        params = result["_searxng_params"]
        assert params["q"] == "test"
        assert params["format"] == "json"
        assert params["categories"] == "general,news"
        assert params["engines"] == "google,bing"
        assert params["time_range"] == "month"


class TestSearXNGSearchURLConstruction:
    """
    Tests that the complete URL is built correctly from api_base and request params.
    """

    def setup_method(self):
        self.config = SearXNGSearchConfig()

    def test_url_with_search_suffix(self):
        """Test URL construction appends /search."""
        data = {"_searxng_params": {"q": "test query", "format": "json"}}
        url = self.config.get_complete_url(
            api_base="https://searxng.example.com",
            optional_params={},
            data=data,
        )

        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "searxng.example.com"
        assert parsed.path == "/search"
        query_params = parse_qs(parsed.query)
        assert query_params["q"] == ["test query"]
        assert query_params["format"] == ["json"]

    def test_url_already_has_search_suffix(self):
        """Test URL construction doesn't double-append /search."""
        data = {"_searxng_params": {"q": "test", "format": "json"}}
        url = self.config.get_complete_url(
            api_base="https://searxng.example.com/search",
            optional_params={},
            data=data,
        )

        parsed = urlparse(url)
        assert parsed.path == "/search"
        assert "/search/search" not in url

    def test_url_with_trailing_slash(self):
        """Test URL construction with trailing slash on api_base."""
        data = {"_searxng_params": {"q": "test", "format": "json"}}
        url = self.config.get_complete_url(
            api_base="https://searxng.example.com/",
            optional_params={},
            data=data,
        )

        parsed = urlparse(url)
        assert parsed.path == "/search"

    def test_url_from_env_variable(self):
        """Test URL construction falls back to SEARXNG_API_BASE env var."""
        data = {"_searxng_params": {"q": "test", "format": "json"}}
        with patch(
            "litellm.llms.searxng.search.transformation.get_secret_str",
            return_value="https://env-searxng.example.com",
        ):
            url = self.config.get_complete_url(
                api_base=None,
                optional_params={},
                data=data,
            )

        assert url.startswith("https://env-searxng.example.com/search?")

    def test_url_missing_api_base_raises(self):
        """Test that missing api_base and env var raises ValueError."""
        with patch(
            "litellm.llms.searxng.search.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="SEARXNG_API_BASE is not set"):
                self.config.get_complete_url(
                    api_base=None,
                    optional_params={},
                    data={"_searxng_params": {"q": "test"}},
                )

    def test_url_without_data_returns_base(self):
        """Test URL construction without data returns just the api_base/search."""
        url = self.config.get_complete_url(
            api_base="https://searxng.example.com",
            optional_params={},
            data=None,
        )

        assert url == "https://searxng.example.com/search"


class TestSearXNGSearchResponseTransformation:
    """
    Tests that SearXNG API responses are correctly transformed to SearchResponse.
    """

    def setup_method(self):
        self.config = SearXNGSearchConfig()
        self.logging_obj = MagicMock()

    def _make_mock_response(self, json_data: dict) -> httpx.Response:
        response = httpx.Response(
            status_code=200,
            json=json_data,
            request=httpx.Request("GET", "https://searxng.example.com/search"),
        )
        return response

    def test_response_with_results(self):
        """Test transforming a typical SearXNG response with results."""
        raw = self._make_mock_response(
            {
                "results": [
                    {
                        "title": "AI News Article",
                        "url": "https://example.com/ai-news",
                        "content": "Latest developments in artificial intelligence.",
                        "publishedDate": "2025-01-15",
                    },
                    {
                        "title": "ML Research Paper",
                        "url": "https://example.com/ml-paper",
                        "content": "New machine learning research findings.",
                        "pubdate": "2025-01-10",
                    },
                ]
            }
        )

        response = self.config.transform_search_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert response.object == "search"
        assert len(response.results) == 2

        first = response.results[0]
        assert first.title == "AI News Article"
        assert first.url == "https://example.com/ai-news"
        assert first.snippet == "Latest developments in artificial intelligence."
        assert first.date == "2025-01-15"
        assert first.last_updated is None

        second = response.results[1]
        assert second.title == "ML Research Paper"
        assert second.date == "2025-01-10"  # from pubdate field

    def test_response_empty_results(self):
        """Test transforming a response with no results."""
        raw = self._make_mock_response({"results": []})

        response = self.config.transform_search_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert response.object == "search"
        assert response.results == []

    def test_response_missing_results_key(self):
        """Test transforming a response that has no 'results' key."""
        raw = self._make_mock_response({"query": "test"})

        response = self.config.transform_search_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert response.object == "search"
        assert response.results == []

    def test_response_missing_optional_fields(self):
        """Test transforming results with missing optional fields."""
        raw = self._make_mock_response(
            {
                "results": [
                    {
                        "title": "Minimal Result",
                        "url": "https://example.com",
                    }
                ]
            }
        )

        response = self.config.transform_search_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        result = response.results[0]
        assert result.title == "Minimal Result"
        assert result.url == "https://example.com"
        assert result.snippet == ""  # defaults to empty string
        assert result.date is None
        assert result.last_updated is None


class TestSearXNGSearchHeaders:
    """
    Tests for header/environment validation.
    """

    def setup_method(self):
        self.config = SearXNGSearchConfig()

    def test_headers_without_api_key(self):
        """Test that headers are set correctly without an API key."""
        with patch(
            "litellm.llms.searxng.search.transformation.get_secret_str",
            return_value=None,
        ):
            headers = self.config.validate_environment(headers={})

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_headers_with_api_key(self):
        """Test that headers include Authorization when API key is provided."""
        headers = self.config.validate_environment(headers={}, api_key="test-key-123")

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_headers_with_env_api_key(self):
        """Test that headers use SEARXNG_API_KEY from env."""
        with patch(
            "litellm.llms.searxng.search.transformation.get_secret_str",
            return_value="env-key-456",
        ):
            headers = self.config.validate_environment(headers={})

        assert headers["Authorization"] == "Bearer env-key-456"

    def test_http_method_is_get(self):
        """Test that the HTTP method is GET."""
        assert self.config.get_http_method() == "GET"
