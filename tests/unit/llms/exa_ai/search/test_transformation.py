from typing import Final
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
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
    return ExaAISearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "id": "doc-1",
                "title": "Title",
                "url": "https://a.example",
                "text": "Body",
                "publishedDate": "2025-01-02T00:00:00.000Z",
                "score": 0.4,
            },
            ("Title", "https://a.example", "Body", "2025-01-02T00:00:00.000Z", None),
        ),
        ({"summary": "Only summary", "publishedDate": None}, ("", "", "Only summary", None, None)),
        ({"text": None, "highlights": None, "summary": None}, ("", "", "", None, None)),
        ({"highlights": "ab"}, ("", "", "a\n\nb", None, None)),
        ({"highlights": {"first": 1, "second": 2}}, ("", "", "first\n\nsecond", None, None)),
        ({"highlights": [""], "summary": "fallback"}, ("", "", "fallback", None, None)),
        ({"text": 0, "highlights": 0, "summary": 0}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_result(
    result: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"requestId": "r", "results": [result]})

    assert [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results] == [expected]
    assert response.object == "search"


def test_transform_search_response_keeps_result_order():
    response = _transform({"results": [{"title": "first"}, {"title": "second"}]})

    assert [result.title for result in response.results] == ["first", "second"]


@pytest.mark.parametrize("payload", [{}, {"results": []}, {"results": {}}, {"results": ""}])
def test_transform_search_response_without_any_result_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"results": None},
        {"results": "not a list"},
        {"results": {"title": "not a list"}},
        {"results": ["not an object"]},
        {"results": [{"title": None}]},
        {"results": [{"url": 7}]},
        {"results": [{"text": ["not", "text"]}]},
        {"results": [{"highlights": 5}]},
        {"results": [{"highlights": ["a", 1]}]},
        {"results": [{"summary": {"not": "text"}}]},
        {"results": [{"publishedDate": 20250102}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
