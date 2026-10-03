import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.searchapi.search.transformation import SearchAPIConfig


def _transform(payload: object) -> SearchResponse:
    return SearchAPIConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=None,
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"position": 1, "title": "Title", "link": "https://a.example", "snippet": "Body", "date": "Jan 2, 2025"},
            ("Title", "https://a.example", "Body", "Jan 2, 2025", None),
        ),
        ({"link": "https://b.example"}, ("", "https://b.example", "", None, None)),
        ({"url": "https://ignored.example", "date": None}, ("", "", "", None, None)),
        ({}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_organic_result(
    result: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"search_metadata": {"id": "abc"}, "organic_results": [result]})

    assert _as_tuples(response) == [expected]
    assert response.object == "search"


def test_transform_search_response_keeps_result_order():
    response = _transform({"organic_results": [{"title": "first"}, {"title": "second"}]})

    assert [result.title for result in response.results] == ["first", "second"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"organic_results": []}, {"organic_results": {}}, {"organic_results": ""}, {"results": [{"title": "x"}]}],
)
def test_transform_search_response_without_any_organic_result_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"organic_results": None},
        {"organic_results": "not a list"},
        {"organic_results": {"title": "not a list"}},
        {"organic_results": ["not an object"]},
        {"organic_results": [{"title": None}]},
        {"organic_results": [{"link": 7}]},
        {"organic_results": [{"snippet": ["not", "text"]}]},
        {"organic_results": [{"date": 20250102}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
