import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.brave.search.transformation import BraveSearchConfig


def _transform(payload: object, query: str = "q=litellm") -> SearchResponse:
    request = httpx.Request("GET", f"https://api.search.brave.com/res/v1/web/search?{query}")
    return BraveSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload, request=request), logging_obj=None
    )


def _entry(index: int) -> dict[str, str]:
    return {"title": f"Title {index}", "url": f"https://example.com/{index}", "description": f"Snippet {index}"}


def test_brave_results_are_mapped_section_by_section_with_normalized_dates():
    response = _transform(
        {
            "type": "search",
            "query": {"original": "litellm"},
            "web": {
                "type": "search",
                "results": [
                    {
                        "title": "LiteLLM",
                        "url": "https://example.com/litellm",
                        "description": "Call every LLM API",
                        "page_age": "2024-03-02T10:00:00",
                        "age": "January 1, 2020",
                        "fetched_content_timestamp": 1700000000,
                        "meta_url": {"hostname": "example.com"},
                    },
                    {"title": "Docs", "url": "https://example.com/docs", "description": "Docs", "page_age": None},
                ],
            },
            "news": {
                "results": [
                    {"title": "News", "url": "https://example.com/news", "description": "Fresh", "age": "March 5, 2024"}
                ]
            },
            "videos": {"results": [{"title": "Video", "url": "https://example.com/video"}]},
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(
            title="LiteLLM",
            url="https://example.com/litellm",
            snippet="Call every LLM API",
            date="2024-03-02",
            last_updated="2023-11-14",
        ),
        SearchResult(title="Docs", url="https://example.com/docs", snippet="Docs"),
        SearchResult(title="News", url="https://example.com/news", snippet="Fresh", date="2024-03-05"),
        SearchResult(title="Video", url="https://example.com/video", snippet=""),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"type": "search", "query": {"original": "litellm"}},
        {"web": {}},
        {"web": {"results": []}, "news": {"results": []}},
        {"web": {"results": ""}},
        {"web": {"results": {}}},
    ],
)
def test_brave_response_without_results_is_empty(payload):
    assert _transform(payload).results == []


def test_brave_result_missing_optional_fields_defaults_to_empty_strings_and_no_dates():
    assert _transform({"web": {"results": [{"profile": {"name": "Example"}}]}}).results == [
        SearchResult(title="", url="", snippet="", date=None, last_updated=None)
    ]


def test_brave_count_limits_results_across_sections_without_reading_entries_past_the_limit():
    response = _transform(
        {
            "web": {"results": [_entry(1), _entry(2), "not a result object"]},
            "news": {"results": ["not a result object"]},
        },
        query="q=litellm&count=2",
    )

    assert [result.title for result in response.results] == ["Title 1", "Title 2"]


def test_brave_result_filter_restricts_the_sections_that_are_read():
    response = _transform(
        {"web": {"results": [_entry(1)]}, "news": {"results": [_entry(2)]}, "videos": None},
        query="q=litellm&result_filter=news",
    )

    assert [result.title for result in response.results] == ["Title 2"]


@pytest.mark.parametrize("field", ["title", "url", "description"])
@pytest.mark.parametrize("value", [None, 7, ["text"]])
def test_brave_result_with_non_string_field_is_rejected(field, value):
    with pytest.raises(ValidationError):
        _transform({"web": {"results": [{**_entry(1), field: value}]}})


def test_brave_result_with_several_non_string_fields_reports_every_field():
    with pytest.raises(ValidationError) as exc_info:
        _transform({"web": {"results": [{"title": 7, "url": None, "description": ["text"]}]}})

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",)]


@pytest.mark.parametrize("payload", [{"web": {"results": [_entry(1)]}}, {}, [], "plain text body"])
def test_brave_non_integer_count_is_reported_whatever_the_response_body_is(payload):
    with pytest.raises(ValueError, match="invalid literal for int"):
        _transform(payload, query="q=litellm&count=abc")
