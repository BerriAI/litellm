from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig


def _transform(payload: object) -> SearchResponse:
    return PerplexitySearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "title": "Title",
                "url": "https://a.example",
                "snippet": "Body",
                "date": "2025-01-02",
                "last_updated": "2025-02-03",
                "score": 0.5,
            },
            ("Title", "https://a.example", "Body", "2025-01-02", "2025-02-03"),
        ),
        ({"title": "Only title"}, ("Only title", "", "", None, None)),
        ({"date": None, "last_updated": None}, ("", "", "", None, None)),
        ({}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_result(
    result: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"results": [result]})

    assert _as_tuples(response) == [expected]
    assert response.object == "search"


def test_transform_search_response_keeps_result_order():
    response = _transform({"results": [{"title": "first"}, {"title": "second"}], "id": "abc"})

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
        {"results": [{"snippet": ["not", "text"]}]},
        {"results": [{"date": 20250102}]},
        {"results": [{"last_updated": False}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
