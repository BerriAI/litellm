from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.google_pse.search.transformation import GooglePSESearchConfig


def _transform(payload: object) -> SearchResponse:
    return GooglePSESearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            {"kind": "customsearch#result", "title": "Title", "link": "https://a.example", "snippet": "Body"},
            ("Title", "https://a.example", "Body", None, None),
        ),
        ({"snippet": "Only snippet"}, ("", "", "Only snippet", None, None)),
        ({"url": "https://ignored.example", "pagemap": {"metatags": []}}, ("", "", "", None, None)),
        ({}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_item(
    item: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"kind": "customsearch#search", "items": [item]})

    assert _as_tuples(response) == [expected]
    assert response.object == "search"


def test_transform_search_response_keeps_item_order():
    response = _transform({"items": [{"title": "first"}, {"title": "second"}]})

    assert [result.title for result in response.results] == ["first", "second"]


@pytest.mark.parametrize("payload", [{}, {"items": []}, {"items": {}}, {"items": ""}, {"results": [{"title": "x"}]}])
def test_transform_search_response_without_any_item_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"items": None},
        {"items": "not a list"},
        {"items": {"title": "not a list"}},
        {"items": ["not an object"]},
        {"items": [{"title": None}]},
        {"items": [{"link": 7}]},
        {"items": [{"snippet": ["not", "text"]}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
