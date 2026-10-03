from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.linkup.search.transformation import LinkupSearchConfig


def _transform(payload: object) -> SearchResponse:
    return LinkupSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str]]:
    return [(result.title, result.url, result.snippet) for result in response.results]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"type": "text", "name": "Title", "url": "https://a.example", "content": "Body"},
            ("Title", "https://a.example", "Body"),
        ),
        ({"name": "Untyped"}, ("Untyped", "", "")),
        ({"type": "text"}, ("", "", "")),
        (
            {"type": "image", "name": "Picture", "url": "https://a.example/i.png"},
            ("Picture", "https://a.example/i.png", ""),
        ),
        (
            {"type": "image", "url": "https://a.example/i.png", "content": "Alt"},
            ("https://a.example/i.png", "https://a.example/i.png", "Alt"),
        ),
        ({"type": "image"}, ("", "", "")),
    ],
)
def test_transform_search_response_maps_one_result(result: dict[str, object], expected: tuple[str, str, str]):
    assert _as_tuples(_transform({"results": [result]})) == [expected]


def test_transform_search_response_skips_unknown_types_without_reading_their_fields():
    response = _transform(
        {
            "results": [
                {"type": "video", "name": None, "url": 7, "content": ["not", "text"]},
                {"type": "text", "name": "Kept", "url": "https://a.example", "content": "Body"},
            ]
        }
    )

    assert _as_tuples(response) == [("Kept", "https://a.example", "Body")]
    assert response.object == "search"


def test_transform_search_response_without_results_key_is_empty():
    assert _transform({"answer": "no sources"}).results == []


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"results": "not a list"},
        {"results": ["not an object"]},
        {"results": [{"type": "text", "name": None}]},
        {"results": [{"type": "image", "name": None, "url": "https://a.example/i.png"}]},
        {"results": [{"type": "image", "url": 7}]},
    ],
)
def test_transform_search_response_rejects_malformed_payloads(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
