import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig

EXA_SEARCH_URL: Final = "https://api.exa.ai/search"


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
    expected: Final = [
        SearchResult(
            title="First", url="https://example.com/first", snippet="First text", date="2024-01-05T00:00:00.000Z"
        ),
        SearchResult(title="Second", url="https://example.com/second", snippet="one\n\ntwo"),
        SearchResult(title="Third", url="https://example.com/third", snippet="Summary"),
    ]

    assert [result.model_dump(include=set(SearchResult.model_fields)) for result in response.results] == [
        result.model_dump() for result in expected
    ]


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


@pytest.mark.parametrize("payload", [{}, {"results": []}, {"results": ""}, {"results": {}}, {"requestId": "abc"}])
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


OUTPUT_SCHEMA: Final = {"type": "object", "properties": {"answer": {"type": "string"}}}
OUTPUT: Final = {
    "content": {"answer": "Grounded answer"},
    "grounding": [
        {"field": "answer", "citations": [{"url": "https://example.com", "title": "Source"}], "confidence": "high"}
    ],
}
FULL_RESULT: Final = {
    "title": "Source",
    "url": "https://example.com",
    "publishedDate": "2026-01-01",
    "author": None,
    "id": "document-id",
    "image": "https://example.com/image.png",
    "favicon": "https://example.com/icon.png",
    "text": "",
    "highlights": ["relevant passage", "another passage"],
    "highlightScores": [0.8, 0.7],
    "summary": "summary",
    "subpages": [{"title": "Subpage", "url": "https://example.com/sub"}],
    "extras": {
        "links": ["https://example.com/link"],
        "richLinks": [{"url": "https://example.com/link", "anchor": "Link"}],
    },
    "entities": [{"id": "entity", "type": "company", "properties": {"name": "Example"}}],
    "futureField": {"nested": [False, 0, None]},
}
FULL_RESPONSE: Final = {
    "results": [FULL_RESULT],
    "output": OUTPUT,
    "requestId": "request-id",
    "costDollars": {"total": 0.012},
    "searchTime": 24,
    "resolvedSearchType": "auto",
    "context": "",
    "futureMetadata": [],
}


@pytest.mark.parametrize("mode", ["instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.usefixtures("exa_api_key")
@pytest.mark.asyncio
async def test_public_search_preserves_native_contract(respx_mock: respx.MockRouter, mode: str, asynchronous: bool):
    native: Final = {
        "type": mode,
        "additionalQueries": ["related question"],
        "outputSchema": OUTPUT_SCHEMA,
        "systemPrompt": "Use grounded sources",
        "objective": "Research",
        "userIdentifier": "opaque-id",
        "userLocation": {"latitude": 40.7, "longitude": -74.0, "country": "US"},
        "includeDomains": ["example.com"],
        "excludeDomains": ["other.example"],
        "startPublishedDate": "2026-01-01T00:00:00Z",
        "endPublishedDate": "2026-02-01T00:00:00Z",
        "moderation": False,
        "numResults": 3,
        "category": "publication",
        "compliance": "hipaa",
        "contents": {
            "highlights": {"query": "evidence", "dynamic": True, "verbosity": "medium"},
            "extras": {"links": 2, "imageLinks": 2, "richLinks": 2, "richImageLinks": 2, "codeBlocks": 2},
            "maxAgeHours": -1,
            "livecrawlTimeout": 5000,
            "subpages": 1,
            "subpageTarget": ["about"],
        },
    }
    route: Final = respx_mock.post(EXA_SEARCH_URL).respond(json=FULL_RESPONSE)
    headers: Final = {"Exa-Beta": "dynamic-highlights-2026-08-28"}
    result: Final = (
        await litellm.asearch(query="question", search_provider="exa_ai", extra_headers=headers, **native)
        if asynchronous
        else litellm.search(query="question", search_provider="exa_ai", extra_headers=headers, **native)
    )
    assert json.loads(route.calls.last.request.content) == {"query": "question", **native}
    assert route.calls.last.request.headers["exa-beta"] == headers["Exa-Beta"]
    dumped: Final = result.model_dump()
    for key, value in FULL_RESPONSE.items():
        if key != "results":
            assert dumped[key] == value
    for key, value in FULL_RESULT.items():
        assert dumped["results"][0][key] == value
    assert dumped["results"][0]["snippet"] == "\n\n".join(FULL_RESULT["highlights"])
    assert dumped["results"][0]["date"] == FULL_RESULT["publishedDate"]
    assert json.loads(result.model_dump_json()) == dumped


@pytest.mark.parametrize(
    "contents",
    [
        None,
        {},
        {"text": False},
        {"highlights": True},
        {
            "text": {
                "maxCharacters": 800,
                "includeHtmlTags": True,
                "verbosity": "full",
                "includeSections": ["body"],
                "excludeSections": ["footer"],
            }
        },
        {"summary": {"query": "summarize", "schema": OUTPUT_SCHEMA}},
        {"highlights": {"maxCharacters": 700, "query": "evidence"}, "snapshotAsOf": "2026-01-01"},
        {"text": True, "livecrawl": "never", "crawledBeforeDate": "2026-01-01", "context": {"maxCharacters": 200}},
    ],
)
@pytest.mark.usefixtures("exa_api_key")
def test_contents_are_sent_without_injecting_text(respx_mock: respx.MockRouter, contents: object):
    route: Final = respx_mock.post(EXA_SEARCH_URL).respond(json={"results": []})
    litellm.search(query="question", search_provider="exa_ai", contents=contents)
    assert json.loads(route.calls.last.request.content) == {"query": "question", "contents": contents}


@pytest.mark.parametrize(
    "output", [None, {}, {"content": "", "grounding": []}, {"content": {}, "grounding": []}, OUTPUT]
)
def test_output_preserves_present_empty_and_null_values(output: object):
    response: Final = _transform({"results": [], "output": output})
    assert response.model_dump()["output"] == output


def test_absent_output_stays_absent():
    assert _transform({"results": []}).model_dump() == {"results": [], "object": "search"}


@pytest.fixture(autouse=True)
def use_httpx_for_respx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def test_legacy_parameters_and_unified_precedence_are_preserved():
    legacy: Final = {
        "max_results": 4,
        "numResults": 9,
        "country": "US",
        "userLocation": "GB",
        "search_domain_filter": ["example.com"],
        "includeDomains": ["other.example"],
        "startCrawlDate": "2020-01-01",
        "endCrawlDate": "2021-01-01",
        "includeText": ["one"],
        "excludeText": ["two"],
        "context": True,
        "max_tokens_per_page": 100,
    }
    before: Final = json.dumps(legacy, sort_keys=True)
    request: Final = ExaAISearchConfig().transform_search_request(["first", "second"], legacy)
    assert request == {
        "query": "first second",
        "numResults": 4,
        "userLocation": "US",
        "includeDomains": ["example.com"],
        "startCrawlDate": "2020-01-01",
        "endCrawlDate": "2021-01-01",
        "includeText": ["one"],
        "excludeText": ["two"],
        "context": True,
        "contents": {"text": True},
    }
    assert json.dumps(legacy, sort_keys=True) == before


@pytest.mark.parametrize("total", [0, 0.012, 2])
@pytest.mark.parametrize("serialized", [False, True])
def test_exa_reported_total_prices_the_whole_request(total: float, serialized: bool) -> None:
    from litellm.search.cost_calculator import search_provider_cost_per_query

    response: Final = SearchResponse.model_validate({"results": [], "costDollars": {"total": total}})
    payload: Final = response.model_dump() if serialized else response
    assert search_provider_cost_per_query("exa_ai/search", "exa_ai", number_of_queries=3, response=payload) == (
        total,
        0.0,
    )
    assert (
        litellm.completion_cost(
            completion_response=payload,
            model="exa_ai/search",
            custom_llm_provider="exa_ai",
            call_type="search",
            optional_params={"query": ["one", "two"], "costDollars": {"total": 999}},
        )
        == total
    )
    assert litellm.cost_per_token(
        model="exa_ai/search", custom_llm_provider="exa_ai", call_type="search", response=payload
    ) == (total, 0.0)


@pytest.mark.parametrize(
    "reported",
    [
        None,
        {},
        {"total": None},
        {"total": -1},
        {"total": True},
        {"total": "0.01"},
        {"total": float("nan")},
        {"total": float("inf")},
        [],
    ],
)
def test_exa_missing_or_invalid_total_uses_estimate(reported: object) -> None:
    from litellm.search.cost_calculator import search_provider_cost_per_query

    estimate: Final = search_provider_cost_per_query("exa_ai/search", "exa_ai")
    assert (
        search_provider_cost_per_query(
            "exa_ai/search",
            "exa_ai",
            response={"costDollars": reported},
            optional_params={"costDollars": {"total": 999}},
        )
        == estimate
    )
