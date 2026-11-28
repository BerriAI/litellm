"""
Tests for the cloro Search provider (POST https://api.cloro.dev/v1/monitor/google).
"""

from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.cloro.search.transformation import CloroSearchConfig

AI_OVERVIEW: Final = {
    "text": "Rank trackers report where a page ranks for a keyword.",
    "sources": [{"label": "Rank tracker guide", "url": "https://example.com/guide", "position": 1}],
    "citationPills": [],
}

PEOPLE_ALSO_ASK: Final = [{"question": "What is the best rank tracker tool?", "type": "AIOVERVIEW"}]


def _organic_row(position: int, link: str) -> dict[str, object]:
    return {
        "position": position,
        "title": f"Result {position}",
        "link": link,
        "displayedLink": "https://example.com > results",
        "snippet": f"Snippet {position}",
        "page": 1,
        "redirectLink": f"https://www.google.com/goto?url=CAES{position}",
    }


def _cloro_response(
    rows: list[dict[str, object]], credits_charged: str | None = None, **extra: object
) -> httpx.Response:
    return httpx.Response(
        200,
        headers={} if credits_charged is None else {"X-Credits-Charged": credits_charged},
        json={"success": True, "result": {"organicResults": rows, "peopleAlsoAsk": PEOPLE_ALSO_ASK, **extra}},
    )


@pytest.mark.asyncio
async def test_asearch_sends_cloro_body_and_maps_results(monkeypatch):
    monkeypatch.setenv("CLORO_API_KEY", "test-cloro-key")
    monkeypatch.delenv("CLORO_API_BASE", raising=False)
    mock_response: Final = _cloro_response(
        [_organic_row(1, "https://example.com/1"), _organic_row(2, "https://example.com/2")],
        credits_charged="7",
        aioverview=AI_OVERVIEW,
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_post:
        response: Final = await litellm.asearch(
            query="best rank tracker",
            search_provider="cloro",
            country="de",
            search_domain_filter=["example.com"],
            include={"aioverview": True},
        )

    call: Final = mock_post.call_args
    assert call.kwargs["url"] == "https://api.cloro.dev/v1/monitor/google"
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-cloro-key"
    assert call.kwargs["json"] == {
        "query": "(best rank tracker) (site:example.com)",
        "country": "DE",
        "include": {"aioverview": True},
    }
    assert [(r.title, r.url, r.snippet) for r in response.results] == [
        ("Result 1", "https://example.com/1", "Snippet 1"),
        ("Result 2", "https://example.com/2", "Snippet 2"),
    ]
    assert response.provider_metadata == {"peopleAlsoAsk": PEOPLE_ALSO_ASK, "aioverview": AI_OVERVIEW}
    per_credit: Final[float] = litellm.model_cost["cloro/search"]["input_cost_per_query"] / 5
    assert response._hidden_params["response_cost"] == pytest.approx(7 * per_credit)


@pytest.mark.parametrize(
    ("domains", "expected_query"),
    [
        (["-example.com"], "(q) -site:example.com"),
        (["-a.com", "-b.com"], "(q) -site:a.com -site:b.com"),
        (["a.com", "b.com", "-c.com"], "(q) (site:a.com OR site:b.com) -site:c.com"),
    ],
)
def test_excluded_domains_become_negative_site_terms(domains: list[str], expected_query: str):
    body: Final = CloroSearchConfig().transform_search_request(
        query="q", optional_params={"search_domain_filter": domains}
    )
    assert body["query"] == expected_query


def test_country_defaults_to_us_because_cloro_requires_it():
    body: Final = CloroSearchConfig().transform_search_request(query=["rank", "tracker"], optional_params={})
    assert body == {"query": "rank tracker", "country": "US"}


def test_unified_params_without_cloro_equivalent_are_not_sent():
    body: Final = CloroSearchConfig().transform_search_request(
        query="litellm",
        optional_params={"max_results": 3, "max_tokens_per_page": 1024, "device": "mobile", "pages": 2},
    )
    assert body == {"query": "litellm", "country": "US", "device": "mobile", "pages": 2}


def test_rows_without_a_direct_link_are_dropped():
    rows: Final = [
        _organic_row(1, ""),
        _organic_row(2, "https://www.google.com/goto?url=CAES2"),
        {"title": "No link at all"},
        _organic_row(4, "https://example.com/4"),
    ]
    response: Final = CloroSearchConfig().transform_search_response(
        raw_response=_cloro_response(rows), logging_obj=MagicMock()
    )
    assert [r.url for r in response.results] == ["https://example.com/4"]


@pytest.mark.parametrize(("max_results", "expected_count"), [(1, 1), (2, 2), (50, 3), (0, 1)])
def test_results_are_cut_to_max_results(max_results, expected_count):
    rows: Final = [_organic_row(i, f"https://example.com/{i}") for i in range(1, 4)]
    response: Final = CloroSearchConfig().transform_search_response(
        raw_response=_cloro_response(rows),
        logging_obj=MagicMock(),
        optional_params={"max_results": max_results},
    )
    assert len(response.results) == expected_count


def test_missing_organic_results_is_an_empty_search():
    response: Final = CloroSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json={"success": True, "result": {"relatedSearches": []}}),
        logging_obj=MagicMock(),
    )
    assert response.results == []
    assert response.provider_metadata == {"relatedSearches": []}


@pytest.mark.parametrize(
    "api_base",
    [None, "https://api.cloro.dev", "https://api.cloro.dev/", "https://api.cloro.dev/v1/monitor/google"],
)
def test_complete_url_resolves_google_route(api_base, monkeypatch):
    monkeypatch.delenv("CLORO_API_BASE", raising=False)
    url: Final = CloroSearchConfig().get_complete_url(api_base=api_base, optional_params={})
    assert url == "https://api.cloro.dev/v1/monitor/google"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CLORO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CLORO_API_KEY"):
        CloroSearchConfig().validate_environment(headers={})


def _search_cost(monkeypatch: pytest.MonkeyPatch, response: httpx.Response, **params: object) -> tuple[float, object]:
    """``litellm.search`` against a mocked cloro response: (response cost, JSON body sent to cloro)."""
    monkeypatch.setenv("CLORO_API_KEY", "test-cloro-key")
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", return_value=response) as mock_post:
        result: Final = litellm.search(query="litellm", search_provider="cloro", **params)
    return result._hidden_params["response_cost"], mock_post.call_args.kwargs["json"]


def test_search_cost_comes_from_cost_map(monkeypatch):
    priced: Final[float] = litellm.model_cost["cloro/search"]["input_cost_per_query"]
    assert priced > 0
    cost, _ = _search_cost(monkeypatch, _cloro_response([_organic_row(1, "https://example.com/1")]))
    assert cost == pytest.approx(priced)


@pytest.mark.parametrize(
    ("credits_charged", "expected_credits"),
    [("9", 9), ("7", 7), ("5", 5), ("0", 0), (None, 5), ("not-a-number", 5), ("-3", 5), ("7.5", 5)],
)
def test_search_cost_bills_the_credits_cloro_charged(monkeypatch, credits_charged, expected_credits):
    """``X-Credits-Charged`` is billed at the cost map's per-credit rate (5 credits = one query), so
    ``pages`` and ``include.aioverview`` reach spend. Without a valid header the per-query price applies."""
    per_credit: Final[float] = litellm.model_cost["cloro/search"]["input_cost_per_query"] / 5
    cost, body = _search_cost(
        monkeypatch,
        _cloro_response([_organic_row(1, "https://example.com/1")], credits_charged=credits_charged),
        pages=2,
        include={"aioverview": True},
    )
    assert body == {"query": "litellm", "country": "US", "pages": 2, "include": {"aioverview": True}}
    assert cost == pytest.approx(expected_credits * per_credit)


def test_caller_cannot_set_the_reported_credits(monkeypatch):
    priced: Final[float] = litellm.model_cost["cloro/search"]["input_cost_per_query"]
    cost, body = _search_cost(
        monkeypatch, _cloro_response([_organic_row(1, "https://example.com/1")]), _cloro_credits_charged=0
    )
    assert "_cloro_credits_charged" not in body
    assert cost == pytest.approx(priced)
