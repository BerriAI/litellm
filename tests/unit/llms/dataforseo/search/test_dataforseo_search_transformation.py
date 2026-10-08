"""
Unit tests for DataForSEO Search functionality.
"""

import datetime
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.dataforseo.search.transformation import DataForSEOSearchConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.asyncio
async def test_dataforseo_search_basic():
    """
    Test DataForSEO search with mocked network response.
    """
    os.environ["DATAFORSEO_LOGIN"] = "test_login"
    os.environ["DATAFORSEO_PASSWORD"] = "test_password"

    mock_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="Latest AI Developments in 2025",
                url="https://example.com/ai-news",
                snippet="Recent advances in artificial intelligence have shown remarkable progress in machine learning and neural networks.",
            ),
            SearchResult(
                title="AI Research Breakthroughs",
                url="https://example.com/ai-research",
                snippet="Scientists announce breakthrough in AI technology with new models achieving unprecedented accuracy.",
            ),
        ],
    )

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_search",
        new_callable=AsyncMock,
    ) as mock_search:
        mock_search.return_value = mock_response

        response = await litellm.asearch(
            query="latest developments in AI",
            search_provider="dataforseo",
        )

        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].title == "Latest AI Developments in 2025"
        assert response.results[0].url == "https://example.com/ai-news"
        assert len(response.results[0].snippet) > 0


def _transform(payload: object) -> SearchResponse:
    logging_obj = Logging(
        model="dataforseo/search",
        messages=[],
        stream=False,
        call_type="search",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return DataForSEOSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload), logging_obj=logging_obj
    )


def _successful_task(result: object) -> dict[str, object]:
    return {"status_code": 20000, "status_message": "Ok.", "tasks": [{"status_code": 20000, "result": result}]}


def test_dataforseo_keeps_only_organic_items_of_the_first_result_in_provider_order():
    response = _transform(
        _successful_task(
            [
                {
                    "keyword": "litellm",
                    "items_count": 4,
                    "items": [
                        {
                            "type": "organic",
                            "rank_absolute": 1,
                            "title": "LiteLLM",
                            "url": "https://example.com/litellm",
                            "description": "Call every LLM API",
                            "links": None,
                        },
                        {"type": "people_also_ask", "items": [{"title": "What is LiteLLM?"}]},
                        {"type": "paid", "title": "Ad", "url": "https://example.com/ad", "description": "Sponsored"},
                        {
                            "type": "organic",
                            "rank_absolute": 4,
                            "title": "Docs",
                            "url": "https://example.com/docs",
                            "description": "Docs",
                        },
                    ],
                },
                "a second result entry is never read",
            ]
        )
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(title="LiteLLM", url="https://example.com/litellm", snippet="Call every LLM API"),
        SearchResult(title="Docs", url="https://example.com/docs", snippet="Docs"),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status_code": 20000, "tasks": []},
        {"tasks": [{"status_code": 40501, "status_message": "Invalid Field.", "result": None}]},
        {"tasks": [{"status_code": 20000}]},
        _successful_task([]),
        _successful_task(""),
        _successful_task([{"keyword": "litellm", "items_count": 0}]),
        _successful_task([{"items": []}]),
        _successful_task([{"items": ""}]),
        _successful_task([{"items": {}}]),
        _successful_task([{"items": [{"type": "paid", "title": "Ad"}, {"title": "no type"}]}]),
        [],
        "plain text body",
    ],
)
def test_dataforseo_response_without_organic_items_is_empty(payload):
    assert _transform(payload).results == []


def test_dataforseo_organic_item_missing_optional_fields_defaults_to_empty_strings():
    assert _transform(_successful_task([{"items": [{"type": "organic"}]}])).results == [
        SearchResult(title="", url="", snippet="", date=None, last_updated=None)
    ]


@pytest.mark.parametrize("field", ["title", "url", "description"])
@pytest.mark.parametrize("value", [None, 7, ["text"]])
def test_dataforseo_organic_item_with_non_string_field_is_rejected(field, value):
    item = {"type": "organic", "title": "T", "url": "https://example.com", "description": "D", field: value}

    with pytest.raises(ValidationError):
        _transform(_successful_task([{"items": [item]}]))


def test_dataforseo_organic_item_with_several_non_string_fields_reports_every_field():
    item = {"type": "organic", "title": 7, "url": None, "description": ["text"]}

    with pytest.raises(ValidationError) as exc_info:
        _transform(_successful_task([{"items": [item]}]))

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",)]


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)
