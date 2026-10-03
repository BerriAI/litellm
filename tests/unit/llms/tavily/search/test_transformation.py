from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.tavily.search.transformation import TavilySearchConfig


def _transform(payload: object) -> SearchResponse:
    return TavilySearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"title": "Title", "url": "https://a.example", "content": "Body", "score": 0.9, "raw_content": None},
            ("Title", "https://a.example", "Body", None, None),
        ),
        ({"content": "Only content"}, ("", "", "Only content", None, None)),
        ({"title": "", "url": "", "content": ""}, ("", "", "", None, None)),
        ({}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_result(
    result: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"query": "q", "results": [result]})

    assert _as_tuples(response) == [expected]
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
        {"results": [{"content": ["not", "text"]}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
