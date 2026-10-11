import json
from typing import Final

import pytest
import respx

import litellm

PERPLEXITY_SEARCH_URL: Final = "https://api.perplexity.ai/search"


@pytest.fixture(autouse=True)
def perplexity_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "test-perplexity-key")
    monkeypatch.delenv("PERPLEXITY_API_BASE", raising=False)


def test_search_maps_perplexity_results_to_search_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={
            "results": [
                {
                    "title": "AI news roundup",
                    "url": "https://example.com/ai-news",
                    "snippet": "The latest in artificial intelligence.",
                    "date": "2026-01-15",
                    "last_updated": "2026-01-16",
                },
                {"title": "Second", "url": "https://example.com/second", "snippet": "Second snippet."},
            ]
        }
    )

    response: Final = litellm.search(query="artificial intelligence recent news", search_provider="perplexity")

    assert response.object == "search"
    assert isinstance(response.results, list)
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert first.title == "AI news roundup"
    assert first.url == "https://example.com/ai-news"
    assert first.snippet == "The latest in artificial intelligence."
    assert first.date == "2026-01-15"
    assert first.last_updated == "2026-01-16"
    assert response.results[1].snippet == "Second snippet."


def test_search_sends_max_results_in_request_body(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={"results": [{"title": "ML", "url": "https://example.com/ml", "snippet": "Machine learning."}]}
    )

    response: Final = litellm.search(query="machine learning", search_provider="perplexity", max_results=5)

    assert json.loads(route.calls.last.request.content) == {"query": "machine learning", "max_results": 5}
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-perplexity-key"
    assert [result.url for result in response.results] == ["https://example.com/ml"]
