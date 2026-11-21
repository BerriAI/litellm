"""
Tests for the SpecStream Search provider (GET https://api.specstream.ai/v1/web-search).
"""

from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.specstream.search.transformation import SpecStreamSearchConfig

_BODY: Final = {
    "data": {
        "query": "litellm proxy",
        "answer": {"value": "OpenAI-compatible", "page": "https://docs.litellm.ai/docs/simple_proxy"},
        "rows": [],
        "sources": [
            {
                "url": "https://docs.litellm.ai/docs/simple_proxy",
                "title": "LiteLLM AI Gateway",
                "site": "docs.litellm.ai",
                "snippet": "A self-hosted server with one OpenAI-compatible endpoint.",
                "cited": True,
            },
            {"url": "https://example.com/b", "title": None, "snippet": None},
            {"url": "https://example.com/c", "title": "C", "snippet": "third"},
        ],
    }
}


def test_search_sends_get_with_bearer_key_and_maps_sources(monkeypatch):
    monkeypatch.setenv("SPECSTREAM_API_KEY", "test-specstream-key")
    monkeypatch.delenv("SPECSTREAM_API_BASE", raising=False)
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _BODY

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
        return_value=mock_response,
    ) as mock_get:
        response: Final = litellm.search(
            query=["litellm", "proxy"],
            search_provider="specstream",
            max_results=2,
        )

    assert mock_get.call_count == 1
    call: Final = mock_get.call_args
    assert call.kwargs["url"] == "https://api.specstream.ai/v1/web-search?q=litellm+proxy"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-specstream-key"

    assert response.object == "search"
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert (first.title, first.url, first.snippet) == (
        "LiteLLM AI Gateway",
        "https://docs.litellm.ai/docs/simple_proxy",
        "A self-hosted server with one OpenAI-compatible endpoint.",
    )
    assert first.model_extra["site"] == "docs.litellm.ai"
    degraded: Final = response.results[1]
    assert (degraded.title, degraded.snippet) == ("", "")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SPECSTREAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPECSTREAM_API_KEY"):
        SpecStreamSearchConfig().validate_environment(headers={})


def test_empty_data_returns_no_results():
    raw: Final = httpx.Response(200, json={"data": {}})
    response: Final = SpecStreamSearchConfig().transform_search_response(raw_response=raw, logging_obj=MagicMock())
    assert response.results == []


def test_provider_is_registered():
    config: Final = litellm.utils.ProviderConfigManager.get_provider_search_config(
        provider=litellm.types.utils.SearchProviders.SPECSTREAM
    )
    assert isinstance(config, SpecStreamSearchConfig)
