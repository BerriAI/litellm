"""
Tests for the Autom Search API integration. The network is never called.
"""

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.llms.autom.search.transformation import AUTOM_PARAMS_KEY, AutomSearchConfig
from litellm.llms.base_llm.search.transformation import SearchResponse


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOM_API_KEY", raising=False)
    monkeypatch.delenv("AUTOM_API_BASE", raising=False)


def _query(url: str) -> dict:
    return {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}


class TestAutomSearchConfig:
    def test_ui_friendly_name_and_method(self):
        config = AutomSearchConfig()
        assert config.ui_friendly_name() == "Autom"
        assert config.get_http_method() == "GET"

    def test_validate_environment_sends_the_key_in_a_header(self, monkeypatch):
        monkeypatch.setenv("AUTOM_API_KEY", "autom-test-key")
        headers = AutomSearchConfig().validate_environment(headers={})
        assert headers["x-api-key"] == "autom-test-key"
        assert headers["Accept"] == "application/json"

    def test_validate_environment_without_api_key(self):
        with pytest.raises(ValueError, match="AUTOM_API_KEY"):
            AutomSearchConfig().validate_environment(headers={})

    def test_transform_search_request_basic(self):
        data = AutomSearchConfig().transform_search_request(query="best running shoes", optional_params={})
        assert data == {AUTOM_PARAMS_KEY: {"query": "best running shoes"}}

    def test_transform_search_request_joins_a_list_query(self):
        data = AutomSearchConfig().transform_search_request(query=["running", "shoes"], optional_params={})
        assert data[AUTOM_PARAMS_KEY]["query"] == "running shoes"

    def test_transform_search_request_maps_country_and_drops_max_results(self):
        data = AutomSearchConfig().transform_search_request(
            query="news", optional_params={"country": "DE", "max_results": 5}
        )
        assert data[AUTOM_PARAMS_KEY] == {"query": "news", "gl": "de"}

    def test_transform_search_request_passes_autom_params_through(self):
        data = AutomSearchConfig().transform_search_request(
            query="news", optional_params={"hl": "fr", "tbs": "qdr:d", "page": 2}
        )
        assert data[AUTOM_PARAMS_KEY] == {"query": "news", "hl": "fr", "tbs": "qdr:d", "page": 2}

    def test_domain_filters_include_with_or_and_exclude_with_and(self):
        data = AutomSearchConfig().transform_search_request(
            query="docs", optional_params={"search_domain_filter": ["a.com", "b.com", "-c.com", "-d.com"]}
        )
        assert data[AUTOM_PARAMS_KEY]["query"] == "(docs) (site:a.com OR site:b.com) -site:c.com -site:d.com"

    def test_domain_filter_exclusion_only(self):
        data = AutomSearchConfig().transform_search_request(
            query="docs", optional_params={"search_domain_filter": ["-c.com"]}
        )
        assert data[AUTOM_PARAMS_KEY]["query"] == "(docs) -site:c.com"

    def test_get_complete_url_builds_the_query_string(self):
        config = AutomSearchConfig()
        data = config.transform_search_request(query="a b", optional_params={"country": "US"})
        url = config.get_complete_url(api_base=None, optional_params={}, data=data)
        assert url.startswith("https://api.autom.dev/v1/google/search?")
        assert _query(url) == {"query": "a b", "gl": "us"}
        assert "x-api-key" not in url and "api_key" not in url

    def test_get_complete_url_with_a_custom_base(self, monkeypatch):
        monkeypatch.setenv("AUTOM_API_BASE", "https://autom.internal.corp")
        url = AutomSearchConfig().get_complete_url(api_base=None, optional_params={}, data=None)
        assert url == "https://autom.internal.corp/v1/google/search"

    def test_get_complete_url_keeps_a_full_route(self):
        url = AutomSearchConfig().get_complete_url(
            api_base="https://api.autom.dev/v1/google/search", optional_params={}, data=None
        )
        assert url == "https://api.autom.dev/v1/google/search"

    def test_transform_search_response(self):
        response = MagicMock()
        response.json.return_value = {
            "organic_results": [
                {"position": 1, "title": "Result 1", "link": "https://example.com/1", "snippet": "One", "domain": "example.com"},
                {"position": 2, "title": "No link", "link": "", "snippet": "skipped"},
                {"position": 3, "title": "Result 3", "link": "https://example.com/3", "snippet": None, "date": "2 days ago"},
            ],
            "pagination": {"current": 1, "has_next_page": True},
        }
        result = AutomSearchConfig().transform_search_response(raw_response=response, logging_obj=None)
        assert isinstance(result, SearchResponse)
        assert [r.url for r in result.results] == ["https://example.com/1", "https://example.com/3"]
        assert result.results[0].title == "Result 1"
        assert result.results[0].snippet == "One"
        assert result.results[1].snippet == ""
        assert result.results[1].date == "2 days ago"

    def test_transform_search_response_empty(self):
        response = MagicMock()
        response.json.return_value = {"search_parameters": {"q": "nothing"}}
        result = AutomSearchConfig().transform_search_response(raw_response=response, logging_obj=None)
        assert result.results == []

    def test_provider_is_registered(self):
        from litellm.types.utils import SearchProviders
        from litellm.utils import ProviderConfigManager

        assert SearchProviders.AUTOM.value == "autom"
        config = ProviderConfigManager.get_provider_search_config(provider=SearchProviders.AUTOM)
        assert isinstance(config, AutomSearchConfig)
