"""
Tests for the DeepSeek Search provider (POST https://api.deepseek.com/v1/tools/search).
"""

from typing import Final
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.deepseek.search.transformation import DeepSeekSearchConfig


@pytest.mark.asyncio
async def test_asearch_sends_deepseek_body_and_maps_results(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "rerank_score": 2.4,
                "title": "DeepSeek V4 model released",
                "url": "https://example.com/deepseek-v4",
                "snippet": "short snippet",
                "summary": "long summary for the model",
                "content": "full page content",
                "site_name": None,
                "date": "2026-06-01T00:00:00+00:00",
                "authority": 3,
            },
            {"title": None, "url": "https://example.com/degraded", "snippet": None, "date": None},
        ]
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        response: Final = await litellm.asearch(
            query=["DeepSeek V4", "release"],
            search_provider="deepseek",
            max_results=5,
            search_domain_filter=["github.com", "huggingface.co"],
            need_content=True,
        )

    assert mock_post.call_count == 1
    call: Final = mock_post.call_args
    assert call.kwargs["url"] == "https://api.deepseek.com/v1/tools/search"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-deepseek-key"
    assert call.kwargs["headers"]["Content-Type"] == "application/json"
    assert call.kwargs["json"] == {
        "query": "DeepSeek V4 release site:github.com OR site:huggingface.co",
        "count": 5,
        "need_content": True,
    }

    assert response.object == "search"
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert (first.title, first.url, first.snippet) == (
        "DeepSeek V4 model released",
        "https://example.com/deepseek-v4",
        "short snippet",
    )
    assert first.date == "2026-06-01T00:00:00+00:00"
    assert first.last_updated is None
    assert first.model_extra["summary"] == "long summary for the model"
    assert first.model_extra["content"] == "full page content"
    assert first.model_extra["rerank_score"] == 2.4
    degraded: Final = response.results[1]
    assert (degraded.title, degraded.snippet, degraded.date) == ("", "", None)


@pytest.mark.parametrize(
    ("max_results", "expected_count"),
    [(0, 1), (3, 3), (50, 50), (500, 50)],
)
def test_max_results_clamps_to_deepseek_count_range(max_results, expected_count):
    body: Final = DeepSeekSearchConfig().transform_search_request(
        query="litellm", optional_params={"max_results": max_results}
    )
    assert body == {"query": "litellm", "count": expected_count}


def test_query_over_deepseek_limit_fails_before_the_call():
    with pytest.raises(
        litellm.BadRequestError, match=re.escape("at most 128 characters, got 144 (3 search_domain_filter")
    ):
        DeepSeekSearchConfig().transform_search_request(
            query="litellm proxy search endpoint configuration guide for enterprise deployments with",
            optional_params={"search_domain_filter": ["docs.litellm.ai", "github.com", "huggingface.co"]},
        )


def test_unified_params_without_deepseek_equivalent_are_not_sent():
    body: Final = DeepSeekSearchConfig().transform_search_request(
        query="litellm",
        optional_params={"country": "US", "max_tokens_per_page": 1024, "ori_question": "what is litellm"},
    )
    assert body == {"query": "litellm", "ori_question": "what is litellm"}


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.deepseek.com",
        "https://api.deepseek.com/",
        "https://api.deepseek.com/beta",
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/v1/tools/search",
    ],
)
def test_complete_url_resolves_search_route_from_chat_style_api_base(api_base, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    url: Final = DeepSeekSearchConfig().get_complete_url(api_base=api_base, optional_params={})
    assert url == "https://api.deepseek.com/v1/tools/search"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekSearchConfig().validate_environment(headers={})


def test_search_cost_comes_from_cost_map(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    priced: Final[float] = litellm.model_cost["deepseek/search"]["input_cost_per_query"]
    assert priced > 0
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [{"title": "t", "url": "https://e.com", "snippet": "s"}]}
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ):
        response: Final = litellm.search(query="litellm", search_provider="deepseek")
    assert response._hidden_params["response_cost"] == pytest.approx(priced)
