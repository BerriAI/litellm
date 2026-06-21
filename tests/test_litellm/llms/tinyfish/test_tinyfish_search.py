"""
Tests for TinyFish Search API integration.
"""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.tinyfish.search.transformation import (
    TinyfishSearchConfig,
    _append_domain_filters,
)

MOCK_TINYFISH_RESPONSE = {
    "query": "web automation tools",
    "results": [
        {
            "position": 1,
            "site_name": "tinyfish.ai",
            "title": "TinyFish - AI Web Automation",
            "snippet": "Automate any website with natural language.",
            "url": "https://tinyfish.ai",
        },
        {
            "position": 2,
            "site_name": "github.com",
            "title": "Top Web Automation Tools",
            "snippet": "A curated list of browser automation frameworks.",
            "url": "https://github.com/example/web-automation",
        },
    ],
    "total_results": 2,
    "page": 0,
}


def _make_mock_response(
    json_data: dict, status_code: int = 200, request_url: str | None = None
) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if request_url:
        mock.request = MagicMock()
        mock.request.url = httpx.URL(request_url)
    else:
        mock.request = None
    return mock


class TestTinyfishSearchConfig:
    def test_ui_friendly_name(self):
        assert TinyfishSearchConfig.ui_friendly_name() == "TinyFish"

    def test_get_http_method(self):
        assert TinyfishSearchConfig().get_http_method() == "GET"

    def test_validate_environment_with_explicit_key(self):
        config = TinyfishSearchConfig()
        headers = config.validate_environment(headers={}, api_key="sk-tinyfish-test")
        assert headers["X-API-Key"] == "sk-tinyfish-test"
        assert headers["Accept"] == "application/json"

    def test_validate_environment_from_env(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value="sk-from-env",
        ):
            headers = config.validate_environment(headers={})
        assert headers["X-API-Key"] == "sk-from-env"

    def test_validate_environment_missing_key(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="TINYFISH_API_KEY"):
                config.validate_environment(headers={})

    def test_validate_environment_uses_api_base_kwarg(self):
        config = TinyfishSearchConfig()
        headers = config.validate_environment(
            headers={},
            api_key="sk-test",
            api_base="https://custom.tinyfish.ai",
        )
        assert headers["X-API-Key"] == "sk-test"


class TestTransformSearchRequest:
    def test_basic_query(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="hello world", optional_params={}
        )
        assert result == {"_tinyfish_params": {"query": "hello world"}}

    def test_list_query_joined(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query=["hello", "world"], optional_params={}
        )
        assert result["_tinyfish_params"]["query"] == "hello world"

    def test_country_maps_to_location(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"country": "US"}
        )
        assert result["_tinyfish_params"]["location"] == "US"

    def test_max_results_clamped_upper(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"max_results": 100}
        )
        assert result["_tinyfish_params"]["max_results"] == 20

    def test_max_results_clamped_lower(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"max_results": 0}
        )
        assert result["_tinyfish_params"]["max_results"] == 1

    def test_max_results_normal(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"max_results": 5}
        )
        assert result["_tinyfish_params"]["max_results"] == 5

    def test_domain_filter_appends_site_operators(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="python tutorials",
            optional_params={"search_domain_filter": ["arxiv.org", "github.com"]},
        )
        query_value = result["_tinyfish_params"]["query"]
        assert "site:arxiv.org" in query_value
        assert "site:github.com" in query_value
        assert "(python tutorials) (site:arxiv.org OR site:github.com)" == query_value

    def test_domain_filter_empty_list_ignored(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"search_domain_filter": []}
        )
        assert result["_tinyfish_params"]["query"] == "test"

    def test_domain_filter_non_list_ignored(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"search_domain_filter": "not-a-list"}
        )
        assert result["_tinyfish_params"]["query"] == "test"

    def test_unknown_params_passed_through(self):
        config = TinyfishSearchConfig()
        result = config.transform_search_request(
            query="test", optional_params={"language": "en", "page": 2}
        )
        params = result["_tinyfish_params"]
        assert params["language"] == "en"
        assert params["page"] == 2

    def test_perplexity_params_not_passed_through(self):
        config = TinyfishSearchConfig()
        supported = config.get_supported_perplexity_optional_params()
        if supported:
            param = next(p for p in supported if p != "max_results" and p != "country")
            result = config.transform_search_request(
                query="test", optional_params={param: "value"}
            )
            assert param not in result["_tinyfish_params"]


class TestGetCompleteUrl:
    def test_default_api_base(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(api_base=None, optional_params={})
        assert url == "https://api.search.tinyfish.ai"

    def test_custom_api_base(self):
        config = TinyfishSearchConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.tinyfish.ai", optional_params={}
        )
        assert url == "https://custom.api.tinyfish.ai"

    def test_env_api_base(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value="https://env.tinyfish.ai",
        ):
            url = config.get_complete_url(api_base=None, optional_params={})
        assert url == "https://env.tinyfish.ai"

    def test_with_tinyfish_params(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(
                api_base=None,
                optional_params={},
                data={"_tinyfish_params": {"query": "hello", "max_results": 5}},
            )
        assert "query=hello" in url
        assert "max_results=5" in url
        assert url.startswith("https://api.search.tinyfish.ai?")

    def test_without_tinyfish_params_key(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(
                api_base=None, optional_params={}, data={"other": "value"}
            )
        assert url == "https://api.search.tinyfish.ai"

    def test_data_none(self):
        config = TinyfishSearchConfig()
        with patch(
            "litellm.llms.tinyfish.search.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(api_base=None, optional_params={}, data=None)
        assert url == "https://api.search.tinyfish.ai"


class TestTransformSearchResponse:
    def test_basic_response(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert result.object == "search"
        assert len(result.results) == 2
        assert result.results[0].title == "TinyFish - AI Web Automation"
        assert result.results[0].url == "https://tinyfish.ai"
        assert (
            result.results[0].snippet == "Automate any website with natural language."
        )

    def test_empty_results(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response({"results": []})
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert result.object == "search"
        assert len(result.results) == 0

    def test_max_results_truncates(self):
        config = TinyfishSearchConfig()
        many_results = {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "snippet": f"Snippet {i}",
                }
                for i in range(10)
            ]
        }
        mock_response = _make_mock_response(
            many_results,
            request_url="https://api.search.tinyfish.ai?query=test&max_results=3",
        )
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 3
        assert result.results[0].title == "Result 0"
        assert result.results[2].title == "Result 2"

    def test_max_results_default_is_20(self):
        config = TinyfishSearchConfig()
        many_results = {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "snippet": f"Snippet {i}",
                }
                for i in range(25)
            ]
        }
        mock_response = _make_mock_response(
            many_results,
            request_url="https://api.search.tinyfish.ai?query=test",
        )
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 20

    def test_missing_fields_default_to_empty_string(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response({"results": [{}]})
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 1
        assert result.results[0].title == ""
        assert result.results[0].url == ""
        assert result.results[0].snippet == ""

    def test_no_request_uses_default_max_results(self):
        config = TinyfishSearchConfig()
        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)
        result = config.transform_search_response(
            raw_response=mock_response, logging_obj=None
        )
        assert len(result.results) == 2


class TestAppendDomainFilters:
    def test_single_domain(self):
        result = _append_domain_filters("test", ["example.com"])
        assert result == "(test) (site:example.com)"

    def test_multiple_domains(self):
        result = _append_domain_filters("query", ["a.com", "b.com", "c.com"])
        assert result == "(query) (site:a.com OR site:b.com OR site:c.com)"
