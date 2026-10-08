from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.searchapi.search.transformation import SearchAPIConfig


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


def _transform(payload: object) -> SearchResponse:
    return SearchAPIConfig().transform_search_response(raw_response=httpx.Response(200, json=payload), logging_obj=None)


def test_searchapi_organic_results_keep_provider_order_fields_and_dates():
    response = _transform(
        {
            "search_metadata": {"id": "search_1", "status": "Success"},
            "search_information": {"total_results": 2},
            "organic_results": [
                {
                    "position": 1,
                    "title": "LiteLLM",
                    "link": "https://example.com/litellm",
                    "snippet": "Call every LLM API",
                    "date": "Jan 5, 2024",
                    "sitelinks": {"inline": [{"title": "Docs", "link": "https://example.com/docs"}]},
                },
                {"position": 2, "title": "Docs", "link": "https://example.com/docs", "snippet": "Docs", "date": None},
            ],
            "related_searches": [{"query": "litellm proxy"}],
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(
            title="LiteLLM", url="https://example.com/litellm", snippet="Call every LLM API", date="Jan 5, 2024"
        ),
        SearchResult(title="Docs", url="https://example.com/docs", snippet="Docs", date=None),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"error": "Invalid API key"},
        {"search_metadata": {"status": "Success"}, "organic_results": []},
        {"organic_results": ""},
        {"organic_results": {}},
    ],
)
def test_searchapi_response_without_organic_results_is_empty(payload):
    assert _transform(payload).results == []


def test_searchapi_result_missing_optional_fields_defaults_to_empty_strings_and_no_date():
    assert _transform({"organic_results": [{"position": 1}]}).results == [
        SearchResult(title="", url="", snippet="", date=None, last_updated=None)
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", None),
        ("title", 7),
        ("link", None),
        ("link", ["https://example.com"]),
        ("snippet", None),
        ("snippet", {"text": "snippet"}),
        ("date", 1704412800),
        ("date", ["Jan 5, 2024"]),
    ],
)
def test_searchapi_result_with_field_of_wrong_type_is_rejected(field, value):
    result = {"title": "T", "link": "https://example.com", "snippet": "S", "date": "Jan 5, 2024", field: value}

    with pytest.raises(ValidationError):
        _transform({"organic_results": [result]})


def test_searchapi_result_with_several_fields_of_wrong_type_reports_every_field():
    result = {"title": 7, "link": None, "snippet": ["snippet"], "date": 1704412800}

    with pytest.raises(ValidationError) as exc_info:
        _transform({"organic_results": [result]})

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",), ("date",)]
