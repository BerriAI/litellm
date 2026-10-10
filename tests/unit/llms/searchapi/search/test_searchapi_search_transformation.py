from collections.abc import Iterator
from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.searchapi.search.transformation import SearchAPIConfig


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("SEARCHAPI_API_KEY", "test-searchapi-key")
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


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

    def test_validate_environment_without_api_key(self, monkeypatch):
        """Test environment validation without API key raises error."""
        monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
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


def test_search_request_and_result_mapping(
    respx_mock: respx.MockRouter, httpx_transport: None
) -> None:
    route: Final = respx_mock.get("https://www.searchapi.io/api/v1/search").respond(
        json={
            "organic_results": [
                {
                    "title": "Search result",
                    "link": "https://example.com/result",
                    "snippet": "A scripted SearchAPI result.",
                }
            ]
        }
    )

    response: Final = litellm.search(
        query="how does LiteLLM search",
        search_provider="searchapi",
        api_key="test-searchapi-key",
    )

    assert dict(route.calls.last.request.url.params) == {
        "engine": "google",
        "q": "how does LiteLLM search",
        "api_key": "test-searchapi-key",
    }
    assert [(item.title, item.url, item.snippet) for item in response.results] == [
        ("Search result", "https://example.com/result", "A scripted SearchAPI result.")
    ]
