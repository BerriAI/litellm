import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx

import litellm
from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig

EXA_SEARCH_URL: Final = "https://api.exa.ai/search"


@pytest.fixture
def exa_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.delenv("EXA_API_BASE", raising=False)


@pytest.mark.parametrize(
    ("content_fields", "expected_snippet"),
    [
        ({"text": "full text"}, "full text"),
        ({"highlights": ["first highlight", "second highlight"]}, "first highlight\n\nsecond highlight"),
        ({"summary": "a summary"}, "a summary"),
        ({"text": "full text", "highlights": ["a highlight"], "summary": "a summary"}, "full text"),
        ({"highlights": ["a highlight"], "summary": "a summary"}, "a highlight"),
        ({"text": "", "highlights": ["a highlight"]}, "a highlight"),
        ({"highlights": [], "summary": "a summary"}, "a summary"),
        ({}, ""),
    ],
)
def test_transform_search_response_snippet_falls_back_through_content_modes(
    content_fields: dict[str, str | list[str]], expected_snippet: str
):
    raw_response: Final = httpx.Response(
        200,
        json={"results": [{"title": "Title", "url": "https://example.com", **content_fields}]},
    )

    response: Final = ExaAISearchConfig().transform_search_response(raw_response, logging_obj=Mock())

    assert response.results[0].snippet == expected_snippet


@pytest.mark.usefixtures("exa_api_key")
def test_search_maps_exa_results_to_search_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(EXA_SEARCH_URL).respond(
        json={
            "results": [
                {
                    "title": "AI news roundup",
                    "url": "https://example.com/ai-news",
                    "text": "The latest in artificial intelligence.",
                    "publishedDate": "2026-01-15T00:00:00.000Z",
                },
                {"title": "Second", "url": "https://example.com/second", "text": "Second text."},
            ]
        }
    )

    response: Final = litellm.search(query="artificial intelligence recent news", search_provider="exa_ai")

    assert response.object == "search"
    assert isinstance(response.results, list)
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert first.title == "AI news roundup"
    assert first.url == "https://example.com/ai-news"
    assert first.snippet == "The latest in artificial intelligence."
    assert first.date == "2026-01-15T00:00:00.000Z"
    assert response.results[1].url == "https://example.com/second"


@pytest.mark.usefixtures("exa_api_key")
def test_search_sends_max_results_as_num_results(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(EXA_SEARCH_URL).respond(
        json={"results": [{"title": "ML", "url": "https://example.com/ml", "text": "Machine learning."}]}
    )

    response: Final = litellm.search(query="machine learning", search_provider="exa_ai", max_results=5)

    assert json.loads(route.calls.last.request.content) == {
        "query": "machine learning",
        "numResults": 5,
        "contents": {"text": True},
    }
    assert route.calls.last.request.headers["x-api-key"] == "test-exa-key"
    assert [result.url for result in response.results] == ["https://example.com/ml"]
