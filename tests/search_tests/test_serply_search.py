"""
Tests for Serply Search API integration.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import pytest


import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest

MOCK_SERPLY_RESPONSE = {
    "results": [
        {
            "title": "Serply Search API",
            "description": "Google SERP results as JSON",
            "link": "https://serply.io/",
            "position": 1,
            "realPosition": 1,
            "result_type": "organic",
            "metadata": {"display_url": "serply.io", "published_time": "Aug 11, 2026"},
        },
        {
            "title": "Serply Docs",
            "description": "Endpoint reference",
            "link": "https://serply.io/docs",
            "position": 2,
            "realPosition": 2,
            "result_type": "organic",
            "metadata": {"display_url": "serply.io › docs"},
        },
    ],
    "ads": [],
    "answers": [],
    "related_searches": {"text": []},
}


def _mock_response() -> Mock:
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = json.dumps(MOCK_SERPLY_RESPONSE).encode()
    response.raise_for_status = Mock()
    return response


def _query_params(mock_get: Mock) -> dict[str, list[str]]:
    return parse_qs(urlsplit(mock_get.call_args.kwargs["url"]).query)


@pytest.mark.skip(reason="Local only tested search providers")
class TestSerplySearch(BaseSearchTest):
    """
    E2E tests for Serply Search functionality that make real API calls.
    Inherits from BaseSearchTest to run standard search tests.
    """

    def get_search_provider(self) -> str:
        return "serply"


class TestSerplySearchTransformation:
    """
    Full-stack tests through `litellm.search` / `litellm.asearch` with the HTTP layer mocked.
    Transformation details are unit-tested in tests/unit/llms/serply/search/.
    """

    @pytest.fixture(autouse=True)
    def _server_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SERPLY_API_KEY", "test-api-key")
        monkeypatch.delenv("SERPLY_API_BASE", raising=False)

    def test_serply_search_request_and_response(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            return_value=_mock_response(),
        ) as mock_get:
            response = litellm.search(
                query="serply search api",
                search_provider="serply",
                max_results=2,
                country="US",
                search_domain_filter=["serply.io", "-spam.example"],
            )

        assert mock_get.called
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["headers"]["X-Api-Key"] == "test-api-key"
        assert call_kwargs["headers"]["User-Agent"] == "litellm"

        assert urlsplit(call_kwargs["url"])._replace(query="").geturl() == "https://api.serply.io/v1/search"
        params = _query_params(mock_get)
        assert params["q"] == ["(serply search api) (site:serply.io) -site:spam.example"]
        assert params["num"] == ["2"]
        assert params["gl"] == ["us"]

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].title == "Serply Search API"
        assert response.results[0].url == "https://serply.io/"
        assert response.results[0].snippet == "Google SERP results as JSON"
        assert response.results[0].date == "Aug 11, 2026"
        assert response.results[1].date is None

    def test_provider_specific_params_survive_to_the_wire(self):
        """Serply-native params must not be eaten by `filter_out_litellm_params`."""
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            return_value=_mock_response(),
        ) as mock_get:
            litellm.search(
                query="test query",
                search_provider="serply",
                tbs="qdr:w",
                hl="de",
                start=10,
            )

        params = _query_params(mock_get)
        assert params["tbs"] == ["qdr:w"]
        assert params["hl"] == ["de"]
        assert params["start"] == ["10"]

    @pytest.mark.asyncio
    async def test_serply_asearch(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new=AsyncMock(return_value=_mock_response()),
        ) as mock_get:
            response = await litellm.asearch(
                query="latest ai developments",
                search_provider="serply",
                tbs="qdr:d",
            )

        assert _query_params(mock_get)["tbs"] == ["qdr:d"]
        assert len(response.results) == 2

    def test_serply_search_tracks_cost(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="serply")

        assert response._hidden_params["response_cost"] == pytest.approx(0.002)
