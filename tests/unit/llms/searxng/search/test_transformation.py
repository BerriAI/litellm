from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.searxng.search.transformation import SearXNGSearchConfig


def _transform(payload: object) -> SearchResponse:
    return SearXNGSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"title": "Title", "url": "https://a.example", "content": "Body", "engine": "google", "score": 1.5},
            ("Title", "https://a.example", "Body", None, None),
        ),
        ({"content": "Only content"}, ("", "", "Only content", None, None)),
        ({}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_result(
    result: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"query": "q", "results": [result]})

    assert _as_tuples(response) == [expected]
    assert response.object == "search"


@pytest.mark.parametrize(
    ("date_fields", "expected_date"),
    [
        ({"publishedDate": "2025-01-02T00:00:00"}, "2025-01-02T00:00:00"),
        ({"pubdate": "2025-03-04"}, "2025-03-04"),
        ({"publishedDate": "2025-01-02", "pubdate": "2025-03-04"}, "2025-01-02"),
        ({"publishedDate": None, "pubdate": "2025-03-04"}, "2025-03-04"),
        ({"publishedDate": "", "pubdate": "2025-03-04"}, "2025-03-04"),
        ({"publishedDate": 0, "pubdate": "2025-03-04"}, "2025-03-04"),
        ({"publishedDate": "", "pubdate": ""}, ""),
        ({"publishedDate": None, "pubdate": None}, None),
        ({"publishedDate": 0}, None),
        ({}, None),
    ],
)
def test_transform_search_response_date_prefers_published_date_then_pubdate(
    date_fields: dict[str, object], expected_date: str | None
):
    response = _transform({"results": [{"title": "Title", **date_fields}]})

    assert response.results[0].date == expected_date


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
        {"results": [{"content": ["not", "text"]}]},
        {"results": [{"publishedDate": 20250102}]},
        {"results": [{"publishedDate": None, "pubdate": 20250102}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
