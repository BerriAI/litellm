import datetime
import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig

PERPLEXITY_SEARCH_URL: Final = "https://api.perplexity.ai/search"


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(autouse=True)
def perplexity_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "test-perplexity-key")
    monkeypatch.delenv("PERPLEXITY_API_BASE", raising=False)


def test_search_maps_perplexity_results_to_search_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={
            "results": [
                {
                    "title": "AI news roundup",
                    "url": "https://example.com/ai-news",
                    "snippet": "The latest in artificial intelligence.",
                    "date": "2026-01-15",
                    "last_updated": "2026-01-16",
                },
                {"title": "Second", "url": "https://example.com/second", "snippet": "Second snippet."},
            ]
        }
    )

    response: Final = litellm.search(query="artificial intelligence recent news", search_provider="perplexity")

    assert response.object == "search"
    assert isinstance(response.results, list)
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert first.title == "AI news roundup"
    assert first.url == "https://example.com/ai-news"
    assert first.snippet == "The latest in artificial intelligence."
    assert first.date == "2026-01-15"
    assert first.last_updated == "2026-01-16"
    assert response.results[1].snippet == "Second snippet."


def test_search_sends_max_results_in_request_body(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={"results": [{"title": "ML", "url": "https://example.com/ml", "snippet": "Machine learning."}]}
    )

    response: Final = litellm.search(query="machine learning", search_provider="perplexity", max_results=5)

    assert json.loads(route.calls.last.request.content) == {"query": "machine learning", "max_results": 5}
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-perplexity-key"
    assert [result.url for result in response.results] == ["https://example.com/ml"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_search_forwards_provider_filters_without_internal_params(
    respx_mock: respx.MockRouter, httpx_transport: None, is_async: bool
) -> None:
    route: Final = respx_mock.post(PERPLEXITY_SEARCH_URL).respond(json={"results": []})
    filters: Final = {
        "search_recency_filter": "week",
        "search_after_date_filter": "01/01/2026",
        "search_before_date_filter": "10/01/2026",
        "last_updated_after_filter": "01/01/2026",
        "last_updated_before_filter": "10/01/2026",
        "search_language_filter": ["en"],
        "search_mode": "academic",
        "max_tokens": 2048,
        "search_domain_filter": ["europa.eu"],
    }
    params: Final = {
        "query": "AI Act obligations for model providers",
        "search_provider": "perplexity",
        "metadata": {"private": "local-only"},
        "input_cost_per_query": 0.01,
        "_litellm_private_marker": "local-only",
        **filters,
    }

    if is_async:
        response = await litellm.asearch(**params)
    else:
        response = litellm.search(**params)

    assert response.object == "search"
    assert json.loads(route.calls.last.request.content) == {
        "query": "AI Act obligations for model providers",
        **filters,
    }


@pytest.mark.parametrize("query", ["original query", ["first query", "second query"]])
def test_search_request_preserves_query_and_present_values_without_mutating_options(query: str | list[str]) -> None:
    optional_params: Final[dict[str, object]] = {
        "query": "replacement query",
        "search_recency_filter": None,
        "search_language_filter": [],
        "max_tokens": 0,
        "provider_option": {"enabled": False},
    }
    original: Final = dict(optional_params)

    body: Final = PerplexitySearchConfig().transform_search_request(
        query=query,
        optional_params=optional_params,
        api_key="transport-only",
        api_base="https://transport.invalid",
        headers={"Authorization": "transport-only"},
    )

    assert body == {
        "query": query,
        "search_language_filter": [],
        "max_tokens": 0,
        "provider_option": {"enabled": False},
    }
    assert optional_params == original


def _transform(payload: object) -> SearchResponse:
    logging_obj = Logging(
        model="perplexity/search",
        messages=[],
        stream=False,
        call_type="search",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="call-id",
        function_id="function-id",
    )
    return PerplexitySearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload), logging_obj=logging_obj
    )


def test_perplexity_results_keep_provider_order_fields_and_dates():
    response = _transform(
        {
            "id": "search-1",
            "server_time": None,
            "results": [
                {
                    "title": "LiteLLM",
                    "url": "https://example.com/litellm",
                    "snippet": "Call every LLM API",
                    "date": "2024-01-05",
                    "last_updated": "2024-02-01",
                },
                {
                    "title": "Docs",
                    "url": "https://example.com/docs",
                    "snippet": "Docs",
                    "date": None,
                    "last_updated": None,
                },
            ],
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(
            title="LiteLLM",
            url="https://example.com/litellm",
            snippet="Call every LLM API",
            date="2024-01-05",
            last_updated="2024-02-01",
        ),
        SearchResult(title="Docs", url="https://example.com/docs", snippet="Docs", date=None, last_updated=None),
    ]


@pytest.mark.parametrize(
    "payload",
    [{}, {"id": "search-1"}, {"id": "search-1", "results": []}, {"results": ""}, {"results": {}}],
)
def test_perplexity_response_without_results_is_empty(payload):
    assert _transform(payload).results == []


def test_perplexity_result_missing_optional_fields_defaults_to_empty_strings_and_no_dates():
    assert _transform({"results": [{"score": 0.5}]}).results == [
        SearchResult(title="", url="", snippet="", date=None, last_updated=None)
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", None),
        ("title", 7),
        ("url", None),
        ("url", ["https://example.com"]),
        ("snippet", None),
        ("snippet", {"text": "snippet"}),
        ("date", 1704412800),
        ("last_updated", ["2024-02-01"]),
    ],
)
def test_perplexity_result_with_field_of_wrong_type_is_rejected(field, value):
    result = {"title": "T", "url": "https://example.com", "snippet": "S", field: value}

    with pytest.raises(ValidationError):
        _transform({"results": [result]})


def test_perplexity_result_with_several_fields_of_wrong_type_reports_every_field():
    result = {"title": 7, "url": None, "snippet": ["snippet"], "date": 1704412800, "last_updated": ["2024-02-01"]}

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": [result]})

    assert [error["loc"] for error in exc_info.value.errors()] == [
        ("title",),
        ("url",),
        ("snippet",),
        ("date",),
        ("last_updated",),
    ]


def test_perplexity_malformed_result_is_rejected_without_naming_its_position():
    results = [{}] * 429 + ["not-an-object"]

    with pytest.raises(ValidationError) as exc_info:
        _transform({"results": results})

    assert "429" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_search_with_search_tools(
    respx_mock: respx.MockRouter, httpx_transport: None
) -> None:
    route: Final = respx_mock.post(PERPLEXITY_SEARCH_URL).respond(
        json={
            "results": [
                {
                    "title": "AI news",
                    "url": "https://example.com/news",
                    "snippet": "Recent AI developments.",
                }
            ]
        }
    )
    router: Final = litellm.Router(
        search_tools=[
            {
                "search_tool_name": "litellm-search",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "test-perplexity-key",
                },
            }
        ]
    )

    response: Final = await router.asearch(
        query="latest AI developments",
        search_tool_name="litellm-search",
        max_results=3,
        search_recency_filter="week",
        search_language_filter=["en"],
    )

    assert json.loads(route.calls.last.request.content) == {
        "query": "latest AI developments",
        "max_results": 3,
        "search_recency_filter": "week",
        "search_language_filter": ["en"],
    }
    assert response.object == "search"
    assert [(result.title, result.url, result.snippet) for result in response.results] == [
        ("AI news", "https://example.com/news", "Recent AI developments.")
    ]


@pytest.mark.parametrize(
    "api_base",
    [
        "https://gateway.example.com/perplexity",
        "https://gateway.example.com/perplexity/",
        "https://gateway.example.com/perplexity/search/",
    ],
)
def test_search_ignores_a_trailing_slash_on_api_base(respx_mock: respx.MockRouter, api_base: str) -> None:
    route: Final = respx_mock.post("https://gateway.example.com/perplexity/search").respond(
        json={"results": [{"title": "ML", "url": "https://example.com/ml", "snippet": "Machine learning."}]}
    )

    response: Final = litellm.search(
        query="machine learning", search_provider="perplexity", api_base=api_base, api_key="test-perplexity-key"
    )

    assert route.called
    assert [result.url for result in response.results] == ["https://example.com/ml"]
