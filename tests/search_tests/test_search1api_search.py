"""
Tests for Search1API Search integration.
"""

import json

import pytest

import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest

SEARCH1API_SEARCH_URL = "https://api.search1api.com/search"

MOCK_SEARCH1API_RESPONSE = {
    "searchParameters": {
        "query": "search1api web search",
        "search_service": "google",
        "max_results": 2,
        "crawl_results": 0,
        "image": False,
        "include_sites": ["s1.dev"],
        "exclude_sites": ["spam.example"],
    },
    "results": [
        {
            "title": "Search1API: One API to search, crawl, and ingest",
            "link": "https://s1.dev/",
            "snippet": "Web search, news, crawling and extraction APIs for AI agents.",
        },
        {
            "title": "Search1API Docs: Search",
            "link": "https://s1.dev/docs/basic/search",
            "snippet": "Search the web across multiple engines and return ranked results.",
        },
    ],
}


@pytest.mark.skip(reason="Local only tested search providers")
class TestSearch1APISearch(BaseSearchTest):
    """
    E2E tests for Search1API Search functionality that make real API calls.
    Inherits from BaseSearchTest to run standard search tests.
    """

    def get_search_provider(self) -> str:
        return "search1api"


class TestSearch1APISearchTransformation:
    """
    Full-stack tests through `litellm.search` / `litellm.asearch` with the HTTP boundary faked by respx.
    Transformation details are unit-tested in tests/test_litellm/llms/search1api/search/.
    """

    @pytest.fixture(autouse=True)
    def _server_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SEARCH1API_API_KEY", "test-api-key")
        monkeypatch.delenv("SEARCH1API_KEY", raising=False)
        monkeypatch.delenv("SEARCH1API_API_BASE", raising=False)
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    @pytest.mark.respx()
    def test_search1api_search_request_and_response(self, respx_mock):
        route = respx_mock.post(SEARCH1API_SEARCH_URL).respond(json=MOCK_SEARCH1API_RESPONSE)

        response = litellm.search(
            query="search1api web search",
            search_provider="search1api",
            max_results=2,
            country="us",
            search_domain_filter=["s1.dev", "-spam.example"],
        )

        assert route.called
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer test-api-key"
        assert request.headers["Content-Type"] == "application/json"

        request_body = json.loads(request.content)
        assert request_body["query"] == "search1api web search"
        assert request_body["max_results"] == 2
        assert request_body["include_sites"] == ["s1.dev"]
        assert request_body["exclude_sites"] == ["spam.example"]
        assert "country" not in request_body
        assert "search_domain_filter" not in request_body

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].title == "Search1API: One API to search, crawl, and ingest"
        assert response.results[0].url == "https://s1.dev/"
        assert response.results[0].snippet == "Web search, news, crawling and extraction APIs for AI agents."
        assert response.results[0].date is None

    @pytest.mark.respx()
    def test_provider_specific_params_survive_to_the_wire(self, respx_mock):
        """Search1API-native params must not be eaten by `filter_out_litellm_params`."""
        route = respx_mock.post(SEARCH1API_SEARCH_URL).respond(json=MOCK_SEARCH1API_RESPONSE)

        litellm.search(
            query="test query",
            search_provider="search1api",
            search_service="bing",
            time_range="month",
            language="de",
        )

        request_body = json.loads(route.calls.last.request.content)
        assert request_body["search_service"] == "bing"
        assert request_body["time_range"] == "month"
        assert request_body["language"] == "de"
        assert request_body["max_results"] == 10

    @pytest.mark.respx()
    def test_disabled_crawl_results_is_a_valid_search(self, respx_mock):
        route = respx_mock.post(SEARCH1API_SEARCH_URL).respond(json=MOCK_SEARCH1API_RESPONSE)

        response = litellm.search(query="test query", search_provider="search1api", crawl_results=0, image=False)

        request_body = json.loads(route.calls.last.request.content)
        assert "crawl_results" not in request_body
        assert "image" not in request_body
        assert len(response.results) == 2

    @pytest.mark.respx()
    def test_crawl_results_is_rejected_before_any_request_is_made(self, respx_mock):
        """A rejected param must fail before hitting the wire, so no Search1API credit is spent."""
        with pytest.raises(Exception, match="crawl_results"):
            litellm.search(query="test query", search_provider="search1api", crawl_results=1)

        assert len(respx_mock.calls) == 0

    @pytest.mark.respx()
    def test_search1api_error_envelope_is_surfaced(self, respx_mock):
        """Verbatim shape of a Search1API 402; the caller sees Search1API's message and status, not a generic 500."""
        respx_mock.post(SEARCH1API_SEARCH_URL).respond(
            status_code=402,
            json={"ok": False, "error": "Payment Required", "message": "Insufficient credits"},
        )

        with pytest.raises(Exception, match="Search1API: Insufficient credits") as excinfo:
            litellm.search(query="test query", search_provider="search1api")

        assert excinfo.value.status_code == 402

    @pytest.mark.asyncio
    @pytest.mark.respx()
    async def test_search1api_asearch(self, respx_mock):
        route = respx_mock.post(SEARCH1API_SEARCH_URL).respond(json=MOCK_SEARCH1API_RESPONSE)

        response = await litellm.asearch(
            query="latest ai developments",
            search_provider="search1api",
            search_service="duckduckgo",
        )

        assert json.loads(route.calls.last.request.content)["search_service"] == "duckduckgo"
        assert len(response.results) == 2

    @pytest.mark.respx()
    def test_search1api_search_tracks_cost(self, respx_mock):
        respx_mock.post(SEARCH1API_SEARCH_URL).respond(json=MOCK_SEARCH1API_RESPONSE)

        response = litellm.search(query="pricing check", search_provider="search1api")

        assert response._hidden_params["response_cost"] == pytest.approx(0.001)
