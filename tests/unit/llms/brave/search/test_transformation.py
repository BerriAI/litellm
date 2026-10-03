import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.brave.search.transformation import BraveSearchConfig

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _transform(payload: object, query_string: str = "") -> SearchResponse:
    return BraveSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload, request=httpx.Request("GET", BRAVE_URL + query_string)),
        logging_obj=None,
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
                "description": "Body",
                "page_age": "2024-03-04T10:00:00",
                "age": "March 9, 2024",
                "fetched_content_timestamp": 1700000000,
                "language": "en",
            },
            ("Title", "https://a.example", "Body", "2024-03-04", "2023-11-14"),
        ),
        ({"age": "March 9, 2024"}, ("", "", "", "2024-03-09", None)),
        ({"page_age": "", "age": "2023/5/6"}, ("", "", "", "2023-05-06", None)),
        ({"page_age": ["2024-01-02"]}, ("", "", "", "2024-01-02", None)),
        ({"page_age": {"published": "2024-01-02"}}, ("", "", "", "2024-01-02", None)),
        ({"page_age": [], "fetched_content_timestamp": {}}, ("", "", "", None, None)),
        ({"fetched_content_timestamp": "1700000000000"}, ("", "", "", None, "2023-11-14")),
        ({"fetched_content_timestamp": 1700000000.5}, ("", "", "", None, "2023-11-14")),
        ({"page_age": True}, ("", "", "", None, None)),
        ({"page_age": "not a date", "fetched_content_timestamp": None}, ("", "", "", None, None)),
        ({}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_web_result(
    result: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform({"type": "search", "web": {"type": "search", "results": [result]}})

    assert _as_tuples(response) == [expected]
    assert response.object == "search"


def test_transform_search_response_reads_sections_in_brave_order():
    response = _transform(
        {
            "videos": {"results": [{"title": "video"}]},
            "news": {"results": [{"title": "news"}]},
            "faq": {"results": [{"title": "faq"}]},
            "discussions": {"results": [{"title": "discussion"}]},
            "web": {"results": [{"title": "web 1"}, {"title": "web 2"}]},
            "infobox": {"results": [{"title": "ignored"}]},
        }
    )

    assert [result.title for result in response.results] == ["web 1", "web 2", "discussion", "faq", "news", "video"]


def test_transform_search_response_only_reads_sections_named_by_result_filter():
    response = _transform(
        {"web": {"results": [{"title": "web"}]}, "news": {"results": [{"title": "news"}]}},
        "?result_filter=news",
    )

    assert [result.title for result in response.results] == ["news"]


@pytest.mark.parametrize(
    ("payload", "query_string", "expected_titles"),
    [
        ({"web": {"results": [{"title": "a"}, {"title": "b"}, {"title": "c"}]}}, "?count=2", ["a", "b"]),
        ({"web": {"results": [{"title": "a"}]}, "news": {"results": [{"title": "b"}]}}, "?count=1", ["a"]),
        ({"web": {"results": [{"title": "a"}, "never read"]}}, "?count=1", ["a"]),
        ({"web": {"results": [{"title": "a"}]}, "news": {"results": "never read"}}, "?count=1", ["a"]),
        ({"web": {"results": [{"title": str(i)} for i in range(25)]}}, "", [str(i) for i in range(20)]),
    ],
)
def test_transform_search_response_stops_reading_results_at_the_requested_count(
    payload: dict[str, object], query_string: str, expected_titles: list[str]
):
    response = _transform(payload, query_string)

    assert [result.title for result in response.results] == expected_titles


@pytest.mark.parametrize(
    "payload",
    [{}, {"web": {}}, {"web": {"results": []}}, {"web": {"results": {}}}, {"web": {"results": ""}}, {"mixed": {}}],
)
def test_transform_search_response_without_any_result_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"web": None},
        {"web": ["not an object"]},
        {"news": "not an object"},
        {"web": {"results": None}},
        {"web": {"results": 5}},
        {"web": {"results": "not a list"}},
        {"web": {"results": {"title": "not a list"}}},
        {"web": {"results": ["not an object"]}},
        {"web": {"results": [{"title": None}]}},
        {"web": {"results": [{"url": 7}]}},
        {"web": {"results": [{"description": ["not", "text"]}]}},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
