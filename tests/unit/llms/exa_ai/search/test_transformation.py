from typing import Final
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig


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


def _transform(payload: object) -> SearchResponse:
    return ExaAISearchConfig().transform_search_response(httpx.Response(200, json=payload), logging_obj=Mock())


def test_transform_search_response_maps_every_exa_result_in_provider_order():
    response: Final = _transform(
        {
            "requestId": "b5947044c4b78efa9552a7c89b306d95",
            "resolvedSearchType": "neural",
            "results": [
                {
                    "id": "https://example.com/first",
                    "title": "First",
                    "url": "https://example.com/first",
                    "publishedDate": "2024-01-05T00:00:00.000Z",
                    "author": "Ada",
                    "score": 0.42,
                    "text": "First text",
                    "highlights": ["ignored when text is present"],
                    "highlightScores": [0.5],
                    "image": None,
                },
                {"title": "Second", "url": "https://example.com/second", "highlights": ["one", "two"]},
                {"title": "Third", "url": "https://example.com/third", "summary": "Summary", "publishedDate": None},
            ],
            "costDollars": {"total": 0.005},
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(
            title="First", url="https://example.com/first", snippet="First text", date="2024-01-05T00:00:00.000Z"
        ),
        SearchResult(title="Second", url="https://example.com/second", snippet="one\n\ntwo"),
        SearchResult(title="Third", url="https://example.com/third", snippet="Summary"),
    ]


def test_transform_search_response_result_without_optional_fields_gets_empty_defaults():
    assert _transform({"results": [{}]}).results == [SearchResult(title="", url="", snippet="")]


@pytest.mark.parametrize(
    ("highlights", "expected_snippet"),
    [
        (None, "a summary"),
        ("", "a summary"),
        ({}, "a summary"),
        ([""], "a summary"),
        (["", ""], "\n\n"),
        ("abc", "a\n\nb\n\nc"),
        ({"first": 1, "second": 2}, "first\n\nsecond"),
    ],
)
def test_transform_search_response_joins_whatever_the_highlights_value_iterates_over(
    highlights: object, expected_snippet: str
):
    response: Final = _transform(
        {
            "results": [
                {"title": "Title", "url": "https://example.com", "highlights": highlights, "summary": "a summary"}
            ]
        }
    )

    assert response.results[0].snippet == expected_snippet


@pytest.mark.parametrize("payload", [{}, {"results": []}, {"results": ""}, {"results": {}}, {"requestId": "abc"}])
def test_transform_search_response_without_results_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "result",
    [
        {"title": None, "url": "https://example.com"},
        {"title": "Title", "url": 7},
        {"title": "Title", "url": "https://example.com", "text": 7},
        {"title": "Title", "url": "https://example.com", "summary": ["a summary"]},
        {"title": "Title", "url": "https://example.com", "publishedDate": 1704412800},
    ],
)
def test_transform_search_response_rejects_result_with_wrongly_typed_field(result: dict[str, object]):
    with pytest.raises(ValidationError):
        _transform({"results": [result]})


def test_transform_search_response_reports_every_wrongly_typed_field_of_a_result():
    result: Final = {"title": 7, "url": None, "text": ["text"], "publishedDate": 1704412800}

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [result]})

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",), ("date",)]


def test_transform_search_response_rejects_malformed_result_without_naming_its_position():
    results = [{}] * 429 + ["not-an-object"]

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": results})

    assert "429" not in str(exc_info.value)
