import datetime
import json
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig

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


def _transform(payload: object) -> SearchResponse:
    logging_obj = Logging(
        model="perplexity/search",
        messages=[],
        stream=False,
        call_type="search",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return PerplexitySearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload), logging_obj=logging_obj
    )


def test_perplexity_results_keep_provider_order_fields_and_dates():
    response = _transform(
        {
            "id": "search-1",
            "server_time": None,
            "results": [
                {
                    "title": "LiteLLM",
                    "url": "https://example.com/litellm",
                    "snippet": "Call every LLM API",
                    "date": "2024-01-05",
                    "last_updated": "2024-02-01",
                },
                {
                    "title": "Docs",
                    "url": "https://example.com/docs",
                    "snippet": "Docs",
                    "date": None,
                    "last_updated": None,
                },
            ],
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(
            title="LiteLLM",
            url="https://example.com/litellm",
            snippet="Call every LLM API",
            date="2024-01-05",
            last_updated="2024-02-01",
        ),
        SearchResult(title="Docs", url="https://example.com/docs", snippet="Docs", date=None, last_updated=None),
    ]


@pytest.mark.parametrize(
    "payload",
    [{}, {"id": "search-1"}, {"id": "search-1", "results": []}, {"results": ""}, {"results": {}}],
)
def test_perplexity_response_without_results_is_empty(payload):
    assert _transform(payload).results == []


def test_perplexity_result_missing_optional_fields_defaults_to_empty_strings_and_no_dates():
    assert _transform({"results": [{"score": 0.5}]}).results == [
        SearchResult(title="", url="", snippet="", date=None, last_updated=None)
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", None),
        ("title", 7),
        ("url", None),
        ("url", ["https://example.com"]),
        ("snippet", None),
        ("snippet", {"text": "snippet"}),
        ("date", 1704412800),
        ("last_updated", ["2024-02-01"]),
    ],
)
def test_perplexity_result_with_field_of_wrong_type_is_rejected(field, value):
    result = {"title": "T", "url": "https://example.com", "snippet": "S", field: value}

    with pytest.raises(ValidationError):
        _transform({"results": [result]})


def test_perplexity_result_with_several_fields_of_wrong_type_reports_every_field():
    result = {"title": 7, "url": None, "snippet": ["snippet"], "date": 1704412800, "last_updated": ["2024-02-01"]}

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [result]})

    assert [error["loc"] for error in exc_info.value.errors()] == [
        ("title",),
        ("url",),
        ("snippet",),
        ("date",),
        ("last_updated",),
    ]


def test_perplexity_malformed_result_is_rejected_without_naming_its_position():
    results = [{}] * 429 + ["not-an-object"]

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": results})

    assert "429" not in str(exc_info.value)
