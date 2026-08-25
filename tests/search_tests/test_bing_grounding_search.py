"""
Tests for the Grounding with Bing Search (Microsoft Foundry) integration.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest

PROJECT_ENDPOINT = "https://acct.services.ai.azure.com/api/projects/proj"

_ANSWER_TEXT = (
    "LiteLLM is an open source LLM gateway ([github.com](https://github.com/BerriAI/litellm))\n"
    "The docs live on docs.litellm.ai ([docs.litellm.ai](https://docs.litellm.ai/))"
)


def _annotation(marker: str, url: str, title: str) -> dict:
    start = _ANSWER_TEXT.index(marker)
    return {
        "type": "url_citation",
        "url": url,
        "title": title,
        "start_index": start,
        "end_index": start + len(marker),
    }


MOCK_BING_GROUNDING_RESPONSE = {
    "id": "resp_mock",
    "object": "response",
    "status": "completed",
    "model": "gpt-4.1",
    "output": [
        {"type": "web_search_call", "status": "completed"},
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": _ANSWER_TEXT,
                    "annotations": [
                        _annotation(
                            "([github.com](https://github.com/BerriAI/litellm))",
                            "https://github.com/BerriAI/litellm",
                            "BerriAI/litellm - GitHub",
                        ),
                        _annotation(
                            "([docs.litellm.ai](https://docs.litellm.ai/))",
                            "https://docs.litellm.ai/",
                            "LiteLLM Docs",
                        ),
                    ],
                }
            ],
        },
    ],
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


def _mock_response():
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = json.dumps(MOCK_BING_GROUNDING_RESPONSE).encode()
    return response


@pytest.mark.skip(reason="Local only tested search providers")
class TestBingGroundingSearch(BaseSearchTest):
    """
    E2E tests for Grounding with Bing Search that make real API calls.
    Inherits from BaseSearchTest to run standard search tests.
    """

    def get_search_provider(self) -> str:
        return "bing_grounding"


class TestBingGroundingSearchTransformation:
    """
    Full-stack tests through `litellm.search` / `litellm.asearch` with the HTTP layer mocked.
    Transformation details are unit-tested in tests/test_litellm/llms/azure/search/.
    """

    @pytest.fixture(autouse=True)
    def _server_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BING_GROUNDING_PROJECT_ENDPOINT", PROJECT_ENDPOINT)
        monkeypatch.setenv("BING_GROUNDING_MODEL", "gpt-4.1")
        monkeypatch.setenv("BING_GROUNDING_TOKEN", "test-entra-token")
        monkeypatch.delenv("BING_GROUNDING_CONNECTION_ID", raising=False)

    def test_bing_grounding_search_request_and_response(self):
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            response = litellm.search(
                query="what is litellm",
                search_provider="bing_grounding",
                max_results=5,
                country="us",
            )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["url"] == f"{PROJECT_ENDPOINT}/openai/v1/responses"
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-entra-token"

        request_body = call_kwargs["json"]
        assert request_body["model"] == "gpt-4.1"
        assert request_body["input"] == "what is litellm"
        assert request_body["tools"] == [
            {"type": "web_search", "user_location": {"type": "approximate", "country": "US"}}
        ]

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].url == "https://github.com/BerriAI/litellm"
        assert response.results[0].title == "BerriAI/litellm - GitHub"
        assert response.results[0].snippet == "LiteLLM is an open source LLM gateway"
        assert response.results[1].url == "https://docs.litellm.ai/"
        assert response.results[1].snippet == "The docs live on docs.litellm.ai"

    def test_connection_mode_sends_the_bing_grounding_tool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "BING_GROUNDING_CONNECTION_ID",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices"
            "/accounts/acct/projects/proj/connections/bing-conn",
        )
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ) as mock_post:
            litellm.search(
                query="what is litellm",
                search_provider="bing_grounding",
                max_results=3,
            )

        request_body = mock_post.call_args.kwargs["json"]
        assert request_body["tools"] == [
            {
                "type": "bing_grounding",
                "bing_grounding": {
                    "search_configurations": [
                        {
                            "project_connection_id": (
                                "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices"
                                "/accounts/acct/projects/proj/connections/bing-conn"
                            ),
                            "count": 3,
                        }
                    ]
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_bing_grounding_asearch(self):
        with patch(  # test-quality-ok: litellm.asearch has no client injection seam
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new=AsyncMock(return_value=_mock_response()),
        ) as mock_post:
            response = await litellm.asearch(
                query="what is litellm",
                search_provider="bing_grounding",
            )

        assert mock_post.call_args.kwargs["json"]["tools"] == [{"type": "web_search"}]
        assert len(response.results) == 2

    def test_web_search_mode_is_not_billed_the_g1_price(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="bing_grounding")

        assert response._hidden_params["response_cost"] == 0.0

    def test_connection_mode_tracks_the_g1_cost(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BING_GROUNDING_CONNECTION_ID", "conn-id")
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        with patch(  # test-quality-ok: litellm.search has no client injection seam
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=_mock_response(),
        ):
            response = litellm.search(query="pricing check", search_provider="bing_grounding")

        assert response._hidden_params["response_cost"] == pytest.approx(0.035)
