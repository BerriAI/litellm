import json
from typing import Final

import httpx
import pytest
import respx

import litellm


def test_firecrawl_search_request_body(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-api-key")
    route: Final = respx_mock.post("https://api.firecrawl.dev/v2/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": {"web": [{"title": "Test Title", "url": "https://example.com", "markdown": "Test content"}]},
            },
        )
    )

    response: Final = litellm.search(query="test query", search_provider="firecrawl", max_results=10, country="US")

    assert route.call_count == 1
    sent: Final = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer test-api-key"
    body: Final = json.loads(sent.content)
    assert body["query"] == "test query"
    assert body["limit"] == 10
    assert body["country"] == "US"
    assert [(result.title, result.url) for result in response.results] == [("Test Title", "https://example.com")]
