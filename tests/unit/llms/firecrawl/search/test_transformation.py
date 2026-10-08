import json
from typing import Final

import litellm


def test_firecrawl_search_request_body(respx_mock, monkeypatch):
    import httpx

    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-api-key")
    route: Final = respx_mock.post(url__regex=r".*firecrawl.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": [
                    {
                        "title": "Test Title",
                        "url": "https://example.com",
                        "markdown": "Test content",
                    }
                ],
            },
        )
    )
    response: Final = litellm.search(
        query="test query", search_provider="firecrawl", max_results=10, country="US"
    )
    assert route.called
    parsed: Final = json.loads(route.calls[0].request.read())
    assert parsed["query"] == "test query"
    assert parsed["limit"] == 10
    assert parsed["country"] == "US"
    assert response.results[0].title == "Test Title"
