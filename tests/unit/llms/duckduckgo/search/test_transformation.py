from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.duckduckgo.search.transformation import DuckDuckGoSearchConfig

DUCKDUCKGO_URL = "https://api.duckduckgo.com"
LONG_TEXT = "word " * 12


def _transform(payload: object, query_string: str = "") -> SearchResponse:
    return DuckDuckGoSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload, request=httpx.Request("GET", DUCKDUCKGO_URL + query_string)),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"Heading": "Python", "AbstractURL": "https://a.example", "AbstractText": "A language", "Type": "A"},
            [("Python", "https://a.example", "A language", None, None)],
        ),
        (
            {"AbstractURL": "https://a.example", "AbstractText": "A language"},
            [("", "https://a.example", "A language", None, None)],
        ),
        ({"Heading": "Python", "AbstractURL": "", "AbstractText": "A language"}, []),
        ({"Heading": "Python", "AbstractURL": "https://a.example", "AbstractText": ""}, []),
        ({"Heading": None, "AbstractURL": None, "AbstractText": None}, []),
        (
            {"RelatedTopics": [{"Text": "Title - the snippet", "FirstURL": "https://b.example"}]},
            [("Title", "https://b.example", "the snippet", None, None)],
        ),
        (
            {"RelatedTopics": [{"Text": "short text", "FirstURL": "https://b.example"}]},
            [("short text", "https://b.example", "short text", None, None)],
        ),
        (
            {"RelatedTopics": [{"Text": LONG_TEXT, "FirstURL": "https://b.example"}]},
            [(LONG_TEXT[:50] + "...", "https://b.example", LONG_TEXT, None, None)],
        ),
        (
            {
                "RelatedTopics": [
                    {"Name": "Group", "Topics": [{"Text": "Nested - body", "FirstURL": "https://c.example"}]}
                ]
            },
            [("Nested", "https://c.example", "body", None, None)],
        ),
        (
            {
                "Heading": "Python",
                "AbstractURL": "https://a.example",
                "AbstractText": "A language",
                "RelatedTopics": [
                    "skipped",
                    None,
                    {"Text": "missing url"},
                    {"Text": "One - first", "FirstURL": "https://1.example"},
                    {"Topics": [{"Text": "Two - second", "FirstURL": "https://2.example"}, {"Name": "no url"}]},
                ],
            },
            [
                ("Python", "https://a.example", "A language", None, None),
                ("One", "https://1.example", "first", None, None),
                ("Two", "https://2.example", "second", None, None),
            ],
        ),
    ],
)
def test_transform_search_response_maps_abstract_and_related_topics(
    payload: dict[str, object], expected: list[tuple[str, str, str, str | None, str | None]]
):
    response = _transform(payload)

    assert _as_tuples(response) == expected
    assert response.object == "search"


@pytest.mark.parametrize(
    ("query_string", "expected_titles"),
    [
        ("", ["Python", "One", "Two", "Three"]),
        ("?_max_results=3", ["Python", "One", "Two"]),
        ("?_max_results=2", ["Python", "One"]),
        ("?_max_results=1", ["Python"]),
        ("?_max_results=not-a-number", ["Python", "One", "Two", "Three"]),
    ],
)
def test_transform_search_response_stops_reading_topics_at_max_results(query_string: str, expected_titles: list[str]):
    response = _transform(
        {
            "Heading": "Python",
            "AbstractURL": "https://a.example",
            "AbstractText": "A language",
            "RelatedTopics": [
                {"Text": "One - first", "FirstURL": "https://1.example"},
                {
                    "Topics": [
                        {"Text": "Two - second", "FirstURL": "https://2.example"},
                        {"Text": "Three - third", "FirstURL": "https://3.example"},
                    ]
                },
            ],
        },
        query_string,
    )

    assert [result.title for result in response.results] == expected_titles


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"RelatedTopics": []},
        {"RelatedTopics": {}},
        {"RelatedTopics": ""},
        {"RelatedTopics": "abc"},
        {"RelatedTopics": {"a": 1}},
    ],
)
def test_transform_search_response_without_any_usable_topic_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"RelatedTopics": None},
        {"RelatedTopics": 5},
        {"Heading": None, "AbstractURL": "https://a.example", "AbstractText": "A language"},
        {"Heading": "Python", "AbstractURL": 7, "AbstractText": "A language"},
        {"Heading": "Python", "AbstractURL": "https://a.example", "AbstractText": ["not", "text"]},
        {"RelatedTopics": [{"Text": "Title - body", "FirstURL": 7}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
