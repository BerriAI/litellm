from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.firecrawl.search.transformation import FirecrawlSearchConfig


def _transform(payload: object) -> SearchResponse:
    return FirecrawlSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
    )


def _as_tuples(response: SearchResponse) -> list[tuple[str, str, str, str | None, str | None]]:
    return [(r.title, r.url, r.snippet, r.date, r.last_updated) for r in response.results]


def test_transform_search_response_maps_the_self_hosted_list_format_in_order():
    response = _transform(
        {
            "success": True,
            "data": [
                {"title": "First", "url": "https://a.example", "markdown": "md", "description": "ignored"},
                {"title": "Second", "url": "https://b.example", "description": "desc"},
                {},
            ],
        }
    )

    assert _as_tuples(response) == [
        ("First", "https://a.example", "md", None, None),
        ("Second", "https://b.example", "desc", None, None),
        ("", "", "", None, None),
    ]
    assert response.object == "search"


def test_transform_search_response_maps_cloud_web_results_before_news_results():
    response = _transform(
        {
            "data": {
                "web": [
                    {"title": "Web", "url": "https://w.example", "markdown": "md", "description": "ignored"},
                    {"title": "Web 2", "url": "https://w2.example", "description": "desc"},
                ],
                "news": [
                    {"title": "News", "url": "https://n.example", "snippet": "headline", "date": "2024-01-02"},
                    {"title": "News 2", "url": "https://n2.example", "markdown": "md", "snippet": "ignored"},
                ],
                "images": [{"title": "never read"}],
            },
            "unrelated": {"nested": [1, 2]},
        }
    )

    assert _as_tuples(response) == [
        ("Web", "https://w.example", "md", None, None),
        ("Web 2", "https://w2.example", "desc", None, None),
        ("News", "https://n.example", "headline", "2024-01-02", None),
        ("News 2", "https://n2.example", "md", None, None),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"success": True},
        {"data": []},
        {"data": {}},
        {"data": None},
        {"data": "text"},
        {"data": 5},
        {"data": {"web": [], "news": []}},
        {"data": {"images": [{"title": "never read"}]}},
    ],
)
def test_transform_search_response_without_web_or_news_results_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize("payload", [[], [{"data": []}], "data", 5, True])
def test_transform_search_response_rejects_a_body_that_is_not_an_object(payload: object):
    with pytest.raises(ValidationError):
        _transform(payload)
