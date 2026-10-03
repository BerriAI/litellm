from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.dataforseo.search.transformation import DataForSEOSearchConfig


def _transform(payload: object) -> SearchResponse:
    return DataForSEOSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _task_payload(items: object) -> dict[str, object]:
    return {
        "status_code": 20000,
        "tasks": [{"id": "task-1", "status_code": 20000, "result": [{"keyword": "q", "items": items}]}],
    }


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            {"type": "organic", "rank_group": 1, "title": "Title", "url": "https://a.example", "description": "Body"},
            ("Title", "https://a.example", "Body", None, None),
        ),
        ({"type": "organic", "description": "Only description"}, ("", "", "Only description", None, None)),
        ({"type": "organic"}, ("", "", "", None, None)),
    ],
)
def test_transform_search_response_maps_one_organic_item(
    item: dict[str, object], expected: tuple[str, str, str, str | None, str | None]
):
    response = _transform(_task_payload([item]))

    assert _as_tuples(response) == [expected]
    assert response.object == "search"


def test_transform_search_response_keeps_only_organic_items_in_order():
    response = _transform(
        _task_payload(
            [
                {"type": "paid", "title": "ad"},
                {"type": "organic", "title": "first"},
                {"type": "people_also_ask", "title": None, "url": 7, "description": ["not", "text"]},
                {"title": "no type"},
                {"type": "organic", "title": "second"},
            ]
        )
    )

    assert [result.title for result in response.results] == ["first", "second"]


def test_transform_search_response_only_reads_the_first_task_and_first_result():
    response = _transform(
        {
            "tasks": [
                {
                    "status_code": 20000,
                    "result": [
                        {"items": [{"type": "organic", "title": "kept"}]},
                        {"items": [{"type": "organic", "title": "second result"}]},
                        "never read",
                    ],
                },
                {"status_code": 20000, "result": [{"items": [{"type": "organic", "title": "second task"}]}]},
                "never read",
            ]
        }
    )

    assert [result.title for result in response.results] == ["kept"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status_code": 20000},
        [],
        ["not", "an", "object"],
        "",
        "plain text body",
        {"tasks": []},
        {"tasks": {}},
        {"tasks": ""},
        {"tasks": [{}]},
        {"tasks": [{"status_code": 40501, "result": None}]},
        {"tasks": [{"status_code": "20000", "result": [{"items": [{"type": "organic", "title": "x"}]}]}]},
        {"tasks": [{"status_code": 20000}]},
        {"tasks": [{"status_code": 20000, "result": []}]},
        {"tasks": [{"status_code": 20000, "result": {}}]},
        {"tasks": [{"status_code": 20000, "result": ""}]},
        {"tasks": [{"status_code": 20000, "result": [{}]}]},
        _task_payload([]),
        _task_payload({}),
        _task_payload(""),
    ],
)
def test_transform_search_response_without_any_organic_item_is_empty(payload: object):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "payload",
    [
        5,
        True,
        ["tasks"],
        "body mentioning tasks",
        {"tasks": None},
        {"tasks": 5},
        {"tasks": "not a list"},
        {"tasks": {"not": "a list"}},
        {"tasks": ["not an object"]},
        {"tasks": [{"status_code": 20000, "result": None}]},
        {"tasks": [{"status_code": 20000, "result": "not a list"}]},
        {"tasks": [{"status_code": 20000, "result": {"not": "a list"}}]},
        {"tasks": [{"status_code": 20000, "result": ["not an object"]}]},
        _task_payload(None),
        _task_payload("not a list"),
        _task_payload(["not an object"]),
        _task_payload([{"type": "organic", "title": None}]),
        _task_payload([{"type": "organic", "url": 7}]),
        _task_payload([{"type": "organic", "description": ["not", "text"]}]),
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
