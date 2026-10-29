import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.exa_ai.search.cost_calculator import EXA_AI_COST_DOLLARS_PARAM
from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig
from litellm.search.cost_calculator import search_provider_cost_per_query

EXA_SEARCH_URL: Final = "https://api.exa.ai/search"
EXA_REPORTED_COST: Final = 0.123456


@pytest.fixture
def exa_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.delenv("EXA_API_BASE", raising=False)


@pytest.mark.parametrize(
    ("content_fields", "expected_snippet"),
    [
        ({"text": "full text"}, "full text"),
        ({"highlights": ["first highlight", "second highlight"]}, "first highlight\n\nsecond highlight"),
        ({"summary": "a summary"}, "a summary"),
        ({"text": "full text", "highlights": ["a highlight"], "summary": "a summary"}, "full text"),
        ({"highlights": ["a highlight"], "summary": "a summary"}, "a highlight"),
        ({"text": "", "highlights": ["a highlight"]}, "a highlight"),
        ({"highlights": [], "summary": "a summary"}, "a summary"),
        ({}, ""),
    ],
)
def test_transform_search_response_snippet_falls_back_through_content_modes(
    content_fields: dict[str, str | list[str]], expected_snippet: str
):
    raw_response: Final = httpx.Response(
        200,
        json={"results": [{"title": "Title", "url": "https://example.com", **content_fields}]},
    )

    response: Final = ExaAISearchConfig().transform_search_response(raw_response, logging_obj=Mock())

    assert response.results[0].snippet == expected_snippet


@pytest.mark.usefixtures("exa_api_key")
def test_search_maps_exa_results_to_search_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(EXA_SEARCH_URL).respond(
        json={
            "results": [
                {
                    "title": "AI news roundup",
                    "url": "https://example.com/ai-news",
                    "text": "The latest in artificial intelligence.",
                    "publishedDate": "2026-01-15T00:00:00.000Z",
                },
                {"title": "Second", "url": "https://example.com/second", "text": "Second text."},
            ]
        }
    )

    response: Final = litellm.search(query="artificial intelligence recent news", search_provider="exa_ai")

    assert response.object == "search"
    assert isinstance(response.results, list)
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert first.title == "AI news roundup"
    assert first.url == "https://example.com/ai-news"
    assert first.snippet == "The latest in artificial intelligence."
    assert first.date == "2026-01-15T00:00:00.000Z"
    assert response.results[1].url == "https://example.com/second"


@pytest.mark.usefixtures("exa_api_key")
def test_search_sends_max_results_as_num_results(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(EXA_SEARCH_URL).respond(
        json={"results": [{"title": "ML", "url": "https://example.com/ml", "text": "Machine learning."}]}
    )

    response: Final = litellm.search(query="machine learning", search_provider="exa_ai", max_results=5)

    assert json.loads(route.calls.last.request.content) == {
        "query": "machine learning",
        "numResults": 5,
        "contents": {"text": True},
    }
    assert route.calls.last.request.headers["x-api-key"] == "test-exa-key"
    assert [result.url for result in response.results] == ["https://example.com/ml"]


def _transform(payload: object) -> SearchResponse:
    return ExaAISearchConfig().transform_search_response(httpx.Response(200, json=payload), logging_obj=Mock())


def test_transform_search_response_maps_every_exa_result_in_provider_order():
    response: Final = _transform(
        {
            "requestId": "b5947044c4b78efa9552a7c89b306d95",
            "resolvedSearchType": "neural",
            "results": [
                {
                    "id": "https://example.com/first",
                    "title": "First",
                    "url": "https://example.com/first",
                    "publishedDate": "2024-01-05T00:00:00.000Z",
                    "author": "Ada",
                    "score": 0.42,
                    "text": "First text",
                    "highlights": ["ignored when text is present"],
                    "highlightScores": [0.5],
                    "image": None,
                },
                {"title": "Second", "url": "https://example.com/second", "highlights": ["one", "two"]},
                {"title": "Third", "url": "https://example.com/third", "summary": "Summary", "publishedDate": None},
            ],
            "costDollars": {"total": 0.005},
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult.model_validate(
            {
                "id": "https://example.com/first",
                "title": "First",
                "url": "https://example.com/first",
                "snippet": "First text",
                "date": "2024-01-05T00:00:00.000Z",
                "author": "Ada",
                "highlights": ["ignored when text is present"],
                "highlight_scores": [0.5],
                "score": 0.42,
                "image": None,
            }
        ),
        SearchResult.model_validate(
            {
                "title": "Second",
                "url": "https://example.com/second",
                "snippet": "one\n\ntwo",
                "highlights": ["one", "two"],
            }
        ),
        SearchResult.model_validate(
            {"title": "Third", "url": "https://example.com/third", "snippet": "Summary", "summary": "Summary"}
        ),
    ]
    assert response.model_extra == {
        "provider_metadata": {
            "requestId": "b5947044c4b78efa9552a7c89b306d95",
            "resolvedSearchType": "neural",
            "costDollars": {"total": 0.005},
        }
    }


@pytest.mark.parametrize(
    ("exa_fields", "expected_extras"),
    [
        ({"text": "full text"}, {}),
        ({"score": 0.3}, {"score": 0.3}),
        ({"highlights": ["a highlight"]}, {"highlights": ["a highlight"]}),
        ({"highlightScores": [0.1]}, {"highlight_scores": [0.1]}),
        ({"score": None}, {"score": None}),
        ({"highlights": [], "highlightScores": [], "score": 0}, {"highlights": [], "highlight_scores": [], "score": 0}),
        ({"publishedDate": "2024-01-05T00:00:00.000Z"}, {}),
        (
            {"id": "https://example.com", "author": "Ada", "image": None, "summary": "a summary"},
            {"id": "https://example.com", "author": "Ada", "image": None, "summary": "a summary"},
        ),
        (
            {"favicon": "https://example.com/favicon.ico", "subpages": []},
            {"favicon": "https://example.com/favicon.ico", "subpages": []},
        ),
    ],
)
def test_transform_search_response_attaches_only_the_exa_result_fields_it_returned(
    exa_fields: dict[str, object], expected_extras: dict[str, object]
):
    response: Final = _transform(
        {
            "results": [
                {"title": "Plain", "url": "https://example.com/plain", "text": "plain"},
                {"title": "Title", "url": "https://example.com", **exa_fields},
            ]
        }
    )

    assert [result.model_extra for result in response.results] == [{}, expected_extras]


@pytest.mark.parametrize(
    ("payload", "expected_metadata"),
    [
        ({}, {}),
        ({"results": []}, {}),
        ({"requestId": "abc"}, {"requestId": "abc"}),
        ({"context": "joined result text", "results": []}, {"context": "joined result text"}),
        (
            {"requestId": "abc", "searchTime": 412.5, "costDollars": {"total": 0.007}, "results": [{"text": "t"}]},
            {"requestId": "abc", "searchTime": 412.5, "costDollars": {"total": 0.007}},
        ),
    ],
)
def test_transform_search_response_keeps_the_exa_envelope_as_provider_metadata(
    payload: dict[str, object], expected_metadata: dict[str, object]
):
    response: Final = _transform(payload)

    assert response.model_extra == {"provider_metadata": expected_metadata}
    assert response.model_dump(mode="json")["provider_metadata"] == expected_metadata


def test_transform_search_response_result_without_optional_fields_gets_empty_defaults():
    assert _transform({"results": [{}]}).results == [SearchResult(title="", url="", snippet="")]


@pytest.mark.parametrize(
    ("highlights", "expected_snippet"),
    [
        (None, "a summary"),
        ("", "a summary"),
        ({}, "a summary"),
        ([""], "a summary"),
        (["", ""], "\n\n"),
        ("abc", "a\n\nb\n\nc"),
        ({"first": 1, "second": 2}, "first\n\nsecond"),
    ],
)
def test_transform_search_response_joins_whatever_the_highlights_value_iterates_over(
    highlights: object, expected_snippet: str
):
    response: Final = _transform(
        {
            "results": [
                {"title": "Title", "url": "https://example.com", "highlights": highlights, "summary": "a summary"}
            ]
        }
    )

    assert response.results[0].snippet == expected_snippet


@pytest.mark.parametrize(
    "payload", [{}, {"results": []}, {"results": None}, {"results": ""}, {"results": {}}, {"requestId": "abc"}]
)
def test_transform_search_response_without_results_is_empty(payload: dict[str, object]):
    assert _transform(payload).results == []


@pytest.mark.parametrize(
    "result",
    [
        {"title": None, "url": "https://example.com"},
        {"title": "Title", "url": 7},
        {"title": "Title", "url": "https://example.com", "text": 7},
        {"title": "Title", "url": "https://example.com", "summary": ["a summary"]},
        {"title": "Title", "url": "https://example.com", "publishedDate": 1704412800},
    ],
)
def test_transform_search_response_rejects_result_with_wrongly_typed_field(result: dict[str, object]):
    with pytest.raises(ValidationError):
        _transform({"results": [result]})


def test_transform_search_response_reports_every_wrongly_typed_field_of_a_result():
    result: Final = {"title": 7, "url": None, "text": ["text"], "publishedDate": 1704412800}

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [result]})

    assert [error["loc"] for error in exc_info.value.errors()] == [("title",), ("url",), ("snippet",), ("date",)]


def test_transform_search_response_rejects_malformed_result_without_naming_its_position():
    results = [{}] * 429 + ["not-an-object"]

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": results})

    assert "429" not in str(exc_info.value)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://gateway.example.com/exa",
        "https://gateway.example.com/exa/",
        "https://gateway.example.com/exa/search/",
    ],
)
def test_get_complete_url_ignores_a_trailing_slash_on_api_base(api_base: str) -> None:
    assert (
        ExaAISearchConfig().get_complete_url(api_base=api_base, optional_params={})
        == "https://gateway.example.com/exa/search"
    )


def _catalog_price(max_results: int) -> float:
    price, _ = search_provider_cost_per_query(
        model="exa_ai/search", custom_llm_provider="exa_ai", optional_params={"max_results": max_results}
    )
    return price


@pytest.mark.usefixtures("exa_api_key")
@pytest.mark.parametrize("reported_cost", [EXA_REPORTED_COST, 0, 3])
def test_search_bills_the_cost_exa_reports(respx_mock: respx.MockRouter, reported_cost: float) -> None:
    respx_mock.post(EXA_SEARCH_URL).respond(
        json={"results": [], "costDollars": {"total": reported_cost, "search": {"neural": reported_cost}}}
    )
    assert _catalog_price(25) != reported_cost

    response: Final = litellm.search(query="asyncio TaskGroup", search_provider="exa_ai", max_results=25)

    assert response.hidden_params["response_cost"] == reported_cost


@pytest.mark.usefixtures("exa_api_key")
@pytest.mark.parametrize(
    "body",
    [
        '{"results": []}',
        '{"results": [], "costDollars": {}}',
        '{"results": [], "costDollars": {"total": null}}',
        '{"results": [], "costDollars": {"total": -0.01}}',
        '{"results": [], "costDollars": {"total": "0.022"}}',
        '{"results": [], "costDollars": {"total": true}}',
        '{"results": [], "costDollars": {"total": NaN}}',
        '{"results": [], "costDollars": {"total": Infinity}}',
        '{"results": [], "costDollars": "0.022"}',
        '{"results": [], "costDollars": [0.022]}',
    ],
)
def test_search_without_a_usable_exa_cost_bills_the_catalog_price(respx_mock: respx.MockRouter, body: str) -> None:
    respx_mock.post(EXA_SEARCH_URL).respond(content=body, headers={"content-type": "application/json"})

    response: Final = litellm.search(query="asyncio TaskGroup", search_provider="exa_ai", max_results=25)

    assert response.hidden_params["response_cost"] == pytest.approx(_catalog_price(25))


@pytest.mark.usefixtures("exa_api_key")
@pytest.mark.parametrize(
    ("config", "expected_cost"),
    [("cost_discount_config", EXA_REPORTED_COST * 0.9), ("cost_margin_config", EXA_REPORTED_COST * 1.1)],
)
def test_an_exa_discount_or_margin_applies_on_top_of_the_cost_exa_reports(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, config: str, expected_cost: float
) -> None:
    monkeypatch.setattr(litellm, config, {"exa_ai": 0.1})
    respx_mock.post(EXA_SEARCH_URL).respond(json={"results": [], "costDollars": {"total": EXA_REPORTED_COST}})

    response: Final = litellm.search(query="asyncio TaskGroup", search_provider="exa_ai", max_results=25)

    assert response.hidden_params["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.usefixtures("exa_api_key")
def test_a_caller_cannot_set_the_cost_exa_reports(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(EXA_SEARCH_URL).respond(json={"results": []})

    response: Final = litellm.search(
        query="asyncio TaskGroup",
        search_provider="exa_ai",
        max_results=25,
        **{EXA_AI_COST_DOLLARS_PARAM: {"total": 0}},
    )

    assert EXA_AI_COST_DOLLARS_PARAM not in json.loads(route.calls.last.request.content)
    assert response.hidden_params["response_cost"] == pytest.approx(_catalog_price(25))


@pytest.mark.usefixtures("exa_api_key")
@pytest.mark.parametrize("serialized", (False, True))
def test_costing_an_exa_reply_directly_returns_the_cost_exa_reports(
    respx_mock: respx.MockRouter, serialized: bool
) -> None:
    respx_mock.post(EXA_SEARCH_URL).respond(json={"results": [], "costDollars": {"total": EXA_REPORTED_COST}})
    response: Final = litellm.search(query="asyncio TaskGroup", search_provider="exa_ai", max_results=25)
    assert _catalog_price(25) != EXA_REPORTED_COST

    costed_response: Final = response.model_dump() if serialized else response
    completion_cost: Final = litellm.completion_cost(
        completion_response=costed_response, model="exa_ai/search", custom_llm_provider="exa_ai", call_type="search"
    )
    per_token: Final = litellm.cost_per_token(
        model="exa_ai/search", custom_llm_provider="exa_ai", call_type="search", response=costed_response
    )

    assert completion_cost == pytest.approx(EXA_REPORTED_COST)
    assert per_token == pytest.approx((EXA_REPORTED_COST, 0.0))
