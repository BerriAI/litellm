"""
Tests for Nimble Search API integration.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest


import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest

MOCK_NIMBLE_RESPONSE = {
    "request_id": "0f8b3a1c-1d2e-4f5a-9b0c-6d7e8f9a0b1c",
    "total_results": 2,
    "results": [
        {
            "title": "Nimble Web API",
            "description": "Short SERP description",
            "url": "https://nimbleway.com/",
            "content": "Full markdown content for the first result",
            "metadata": {"position": 1, "entity_type": "organic", "country": "US", "locale": "en"},
            "additional_data": {"publish_date": "2026-07-15"},
        },
        {
            "title": "Nimble Docs",
            "description": "Only a description here",
            "url": "https://docs.nimbleway.com/",
            "content": "",
            "metadata": {"position": 2, "entity_type": "organic"},
            "additional_data": None,
        },
    ],
    "serp_data": None,
}


def _mock_response():
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = json.dumps(MOCK_NIMBLE_RESPONSE).encode()
    return response


@pytest.mark.skip(reason="Local only tested search providers")
class TestNimbleSearch(BaseSearchTest):
    """
    E2E tests for Nimble Search functionality that make real API calls.
    Inherits from BaseSearchTest to run standard search tests.
    """

    def get_search_provider(self) -> str:
        return "nimble"


class TestNimbleSearchTransformation:
    """
    Full-stack tests through `litellm.search` / `litellm.asearch` with the HTTP layer mocked.
    Transformation details are unit-tested in tests/test_litellm/llms/nimble/search/.
    """

    @pytest.fixture(autouse=True)
    def _server_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NIMBLE_API_KEY", "test-api-key")
        monkeypatch.delenv("NIMBLE_API_BASE", raising=False)

    def test_nimble_search_request_and_response(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            response = litellm.search(
                query="nimble web scraping",
                search_provider="nimble",
                max_results=2,
                country="us",
                search_domain_filter=["nimbleway.com", "-spam.example"],
            )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["url"] == "https://sdk.nimbleway.com/v2/search"
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert call_kwargs["headers"]["X-Client-Source"] == "litellm"

        request_body = call_kwargs["json"]
        assert request_body["query"] == "nimble web scraping"
        assert request_body["max_results"] == 2
        assert request_body["country"] == "US"
        assert request_body["include_domains"] == ("nimbleway.com",)
        assert request_body["exclude_domains"] == ("spam.example",)

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].title == "Nimble Web API"
        assert response.results[0].url == "https://nimbleway.com/"
        assert response.results[0].snippet == "Full markdown content for the first result"
        assert response.results[0].date == "2026-07-15"
        # Second result has no `content`, so the SERP description is the snippet.
        assert response.results[1].snippet == "Only a description here"
        assert response.results[1].date is None

    def test_provider_specific_params_survive_to_the_wire(self):
        """Nimble-native params must not be eaten by `filter_out_litellm_params`."""
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            litellm.search(
                query="test query",
                search_provider="nimble",
                focus="news",
                search_depth="deep",
                time_range="week",
                locale="fr",
                output_format="plain_text",
                max_subagents=5,
            )

        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["focus"] == "news"
        assert request_body["search_depth"] == "deep"
        assert request_body["time_range"] == "week"
        assert request_body["locale"] == "fr"
        assert request_body["output_format"] == "plain_text"
        assert request_body["max_subagents"] == 5

    @pytest.mark.asyncio
    async def test_nimble_asearch(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=AsyncMock(return_value=_mock_response()),
        ) as mock_post:
            response = await litellm.asearch(
                query="latest ai developments",
                search_provider="nimble",
                focus="news",
            )

        assert mock_post.call_args.kwargs["json"]["focus"] == "news"
        assert len(response.results) == 2

    def test_nimble_search_tracks_cost(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="nimble")

        assert response._hidden_params["response_cost"] == pytest.approx(0.005)
