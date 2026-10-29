from typing import Final

import pytest
import respx

import litellm

PERPLEXITY_SEARCH_URL: Final = "https://api.perplexity.ai/search"


@pytest.fixture
def perplexity_search(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> respx.Route:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "test-perplexity-key")
    monkeypatch.delenv("PERPLEXITY_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={"results": [{"title": "AI news", "url": "https://example.com/ai-news", "snippet": "AI news."}]}
    )
