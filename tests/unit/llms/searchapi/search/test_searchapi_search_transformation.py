from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.searchapi.search.transformation import SearchAPIConfig


class TestSearchAPIConfig:
    """Test SearchAPI.io configuration and transformations."""

    def test_ui_friendly_name(self) -> None:
        """Test that UI friendly name is returned correctly."""
        config: Final = SearchAPIConfig()
        assert config.ui_friendly_name() == "SearchAPI.io (Google Search)"

    def test_get_http_method(self) -> None:
        """Test that HTTP method is GET."""
        config: Final = SearchAPIConfig()
        assert config.get_http_method() == "GET"

    def test_validate_environment_with_api_key(self) -> None:
        """Test environment validation with API key."""
        config: Final = SearchAPIConfig()

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value="test_api_key",
        ):
            result: Final = config.validate_environment({}, api_key="test_api_key")

        assert result["Content-Type"] == "application/json"

    def test_validate_environment_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment validation without API key raises error."""
        monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
        config: Final = SearchAPIConfig()

        with pytest.raises(ValueError, match="SEARCHAPI_API_KEY is not set"):
            config.validate_environment({})

    def test_transform_search_request_basic(self) -> None:
        """Test basic search request transformation."""
        config: Final = SearchAPIConfig()

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value="test_api_key",
        ):
            result: Final = config.transform_search_request(
                query="test query", optional_params={}, api_key="test_api_key"
            )

        assert "_searchapi_params" in result
        params: Final = result["_searchapi_params"]
        assert params["engine"] == "google"
        assert params["q"] == "test query"
        assert params["api_key"] == "test_api_key"

    def test_transform_search_request_with_max_results(self) -> None:
        """Test search request transformation with max_results parameter."""
        config: Final = SearchAPIConfig()

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value="test_api_key",
        ):
            result: Final = config.transform_search_request(
                query="test query",
                optional_params={"max_results": 5},
                api_key="test_api_key",
            )

        params: Final = result["_searchapi_params"]
        assert params["num"] == 5

    def test_transform_search_request_with_country(self) -> None:
        """Test search request transformation with country parameter."""
        config: Final = SearchAPIConfig()

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value="test_api_key",
        ):
            result: Final = config.transform_search_request(
                query="test query",
                optional_params={"country": "US"},
                api_key="test_api_key",
            )

        params: Final = result["_searchapi_params"]
        assert params["gl"] == "us"

    def test_transform_search_request_with_domain_filter(self) -> None:
        """Test search request transformation with domain filter."""
        config: Final = SearchAPIConfig()

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value="test_api_key",
        ):
            result: Final = config.transform_search_request(
                query="test query",
                optional_params={"search_domain_filter": ["example.com", "test.com"]},
                api_key="test_api_key",
            )

        params: Final = result["_searchapi_params"]
        assert "site:example.com" in params["q"]
        assert "site:test.com" in params["q"]

    def test_transform_search_request_with_list_query(self) -> None:
        """Test search request transformation with list query."""
        config: Final = SearchAPIConfig()

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value="test_api_key",
        ):
            result: Final = config.transform_search_request(
                query=["test", "query"], optional_params={}, api_key="test_api_key"
            )

        params: Final = result["_searchapi_params"]
        assert params["q"] == "test query"

    def test_get_complete_url(self) -> None:
        """Test URL construction with query parameters."""
        config: Final = SearchAPIConfig()
        data: Final = {
            "_searchapi_params": {
                "engine": "google",
                "q": "test query",
                "api_key": "test_key",
            }
        }

        with patch(
            "litellm.llms.searchapi.search.transformation.get_secret_str",
            return_value=None,
        ):
            url: Final = config.get_complete_url(api_base=None, optional_params={}, data=data)

        assert "https://www.searchapi.io/api/v1/search?" in url
        assert "engine=google" in url
        assert "q=test+query" in url
        assert "api_key=test_key" in url

    def test_transform_search_response(self) -> None:
        """Test search response transformation."""
        config: Final = SearchAPIConfig()
        mock_response: Final = Mock(
            spec=httpx.Response,
            json=Mock(
                return_value={
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
            ),
        )

        result: Final = config.transform_search_response(raw_response=mock_response, logging_obj=None)

        assert isinstance(result, SearchResponse)
        assert result.object == "search"
        assert len(result.results) == 2

        first_result: Final = result.results[0]
        assert first_result.title == "Test Result 1"
        assert first_result.url == "https://example.com/1"
        assert first_result.snippet == "This is a test snippet 1"
        assert first_result.date == "2024-01-01"
        assert first_result.last_updated is None

        second_result: Final = result.results[1]
        assert second_result.title == "Test Result 2"
        assert second_result.url == "https://example.com/2"
        assert second_result.snippet == "This is a test snippet 2"
        assert second_result.date is None

    def test_transform_search_response_empty(self) -> None:
        """Test search response transformation with no results."""
        config: Final = SearchAPIConfig()
        mock_response: Final = Mock(
            spec=httpx.Response,
            json=Mock(return_value={"organic_results": []}),
        )

        result: Final = config.transform_search_response(raw_response=mock_response, logging_obj=None)

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 0

    def test_append_domain_filters(self) -> None:
        """Test domain filter appending logic."""
        config: Final = SearchAPIConfig()
        query: Final = "test query"

        result: Final = config._append_domain_filters(query, ["example.com", "test.com"])

        assert "(test query)" in result
        assert "site:example.com" in result
        assert "site:test.com" in result
        assert "OR" in result
        assert "AND" in result
