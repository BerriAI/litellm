import json
from collections.abc import Iterator
from typing import Final

import openai
import pytest
import respx

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.bigdata.search.transformation import BigdataSearchConfig

SEARCH_URL: Final = f"{BigdataSearchConfig.DEFAULT_API_BASE}/search"
EMPTY_RESPONSE: Final = {"results": [], "external_results": {}}


@pytest.fixture(autouse=True)
def bigdata_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("BIGDATA_API_KEY", raising=False)
    monkeypatch.delenv("BIGDATA_API_BASE", raising=False)
    monkeypatch.setattr(  # test-quality-ok: select HTTPX so respx can intercept the external HTTP boundary
        litellm, "disable_aiohttp_transport", True
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def test_search_maps_request_and_flattens_documents_and_web_results(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(SEARCH_URL).respond(
        json={
            "results": [
                {
                    "id": "0123456789abcdef0123456789abcdef",
                    "headline": "Chipmaker raises guidance",
                    "timestamp": "2026-11-02T13:00:00Z",
                    "source": {"id": "src-1", "name": "Newswire", "rank": "RANK_1"},
                    "url": "https://news.example/guidance",
                    "document_type": "NEWS",
                    "chunks": [
                        {"cnum": 4, "text": "Revenue guidance was raised.", "relevance": 0.91, "sentiment": 0.6},
                        {"cnum": 9, "text": "Margins also improved.", "relevance": 0.72, "sentiment": 0.3},
                    ],
                },
                {
                    "id": "fedcba9876543210fedcba9876543210",
                    "headline": "Q3 earnings call",
                    "timestamp": "2026-10-30T21:00:00Z",
                    "source": {"id": "src-2", "name": "Transcripts", "rank": "RANK_1"},
                    "document_type": "TRANSCRIPT",
                    "chunks": [{"cnum": 1, "text": "We expect strong demand.", "relevance": 0.8}],
                },
            ],
            "external_results": {
                "web": [
                    {
                        "headline": "Analyst blog",
                        "url": "https://blog.example/post",
                        "src_name": "blog.example",
                        "source": {"name": "blog.example"},
                        "chunks": [{"cnum": 0, "text": "Demand looks durable."}],
                    }
                ]
            },
            "usage": {"api_query_units": 0.7},
            "metadata": {"request_id": "req-1", "timestamp": "2026-11-17T10:00:00Z"},
            "pagination": {"has_cursor": False},
        }
    )

    response: Final = litellm.search(
        query="chipmaker guidance",
        search_provider="bigdata",
        api_key="test-key",
        max_results=12,
        country="us",
        max_tokens_per_page=500,
        filters={"timestamp": {"start": "2026-10-01T00:00:00Z", "end": "2026-11-17T00:00:00Z"}},
        external_search={"mode": "INCLUDE", "values": ["web"]},
        response_detail="standard",
    )

    assert route.calls.last.request.headers["X-API-KEY"] == "test-key"
    assert json.loads(route.calls.last.request.content) == {
        "search_mode": "fast",
        "response_detail": "standard",
        "query": {
            "text": "chipmaker guidance",
            "max_chunks": 12,
            "external_search": {"mode": "INCLUDE", "values": ["web"]},
            "filters": {"timestamp": {"start": "2026-10-01T00:00:00Z", "end": "2026-11-17T00:00:00Z"}},
        },
    }
    assert response.model_dump() == {
        "object": "search",
        "results": [
            {
                "title": "Chipmaker raises guidance",
                "url": "https://news.example/guidance",
                "snippet": "Revenue guidance was raised.\n\nMargins also improved.",
                "date": "2026-11-02T13:00:00Z",
                "last_updated": None,
                "id": "0123456789abcdef0123456789abcdef",
                "source": {"id": "src-1", "name": "Newswire", "rank": "RANK_1"},
                "document_type": "NEWS",
                "chunks": [
                    {"cnum": 4, "text": "Revenue guidance was raised.", "relevance": 0.91, "sentiment": 0.6},
                    {"cnum": 9, "text": "Margins also improved.", "relevance": 0.72, "sentiment": 0.3},
                ],
            },
            {
                "title": "Q3 earnings call",
                "url": "",
                "snippet": "We expect strong demand.",
                "date": "2026-10-30T21:00:00Z",
                "last_updated": None,
                "id": "fedcba9876543210fedcba9876543210",
                "source": {"id": "src-2", "name": "Transcripts", "rank": "RANK_1"},
                "document_type": "TRANSCRIPT",
                "chunks": [{"cnum": 1, "text": "We expect strong demand.", "relevance": 0.8}],
            },
            {
                "title": "Analyst blog",
                "url": "https://blog.example/post",
                "snippet": "Demand looks durable.",
                "date": None,
                "last_updated": None,
                "src_name": "blog.example",
                "source": {"name": "blog.example"},
                "chunks": [{"cnum": 0, "text": "Demand looks durable."}],
            },
        ],
        "provider_metadata": {
            "usage": {"api_query_units": 0.7},
            "metadata": {"request_id": "req-1", "timestamp": "2026-11-17T10:00:00Z"},
            "pagination": {"has_cursor": False},
        },
    }


@pytest.mark.parametrize("api_base", [None, "https://api.bigdata.com/v1", "https://api.bigdata.com/v1/search/"])
async def test_asearch_uses_environment_credentials_at_default_endpoint(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, api_base: str | None
) -> None:
    monkeypatch.setenv("BIGDATA_API_KEY", "environment-key")
    route: Final = respx_mock.post(SEARCH_URL).respond(json=EMPTY_RESPONSE)

    response: Final = await litellm.asearch(query="no matches", search_provider="bigdata", api_base=api_base)

    assert response.model_dump() == {"object": "search", "results": [], "provider_metadata": {}}
    assert route.calls.last.request.headers["X-API-KEY"] == "environment-key"
    assert json.loads(route.calls.last.request.content) == {"search_mode": "fast", "query": {"text": "no matches"}}


async def test_asearch_uses_explicit_credentials_at_custom_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://bigdata.example/v1/search").respond(json=EMPTY_RESPONSE)

    await litellm.asearch(
        query="no matches", search_provider="bigdata", api_key="custom-key", api_base="https://bigdata.example/v1/"
    )

    assert route.calls.last.request.headers["X-API-KEY"] == "custom-key"


def test_native_options_override_mapped_defaults_without_mutating_input() -> None:
    params: Final = {
        "max_results": 3,
        "max_chunks": 40,
        "search_mode": "smart",
        "include_audit": True,
        "context": "Focus on the last quarter",
        "ranking_params": {"freshness_boost": 5.0},
        "auto_enrich_filters": False,
    }
    before: Final = json.dumps(params, sort_keys=True)

    assert BigdataSearchConfig().transform_search_request(["first", "second"], params) == {
        "search_mode": "smart",
        "include_audit": True,
        "query": {
            "text": "first second",
            "max_chunks": 40,
            "context": "Focus on the last quarter",
            "ranking_params": {"freshness_boost": 5.0},
            "auto_enrich_filters": False,
        },
    }
    assert json.dumps(params, sort_keys=True) == before


def test_domain_filter_is_rejected_before_any_request(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(SEARCH_URL).respond(json=EMPTY_RESPONSE)

    with pytest.raises(openai.BadRequestError, match="search_domain_filter"):
        litellm.search(
            query="query", search_provider="bigdata", api_key="key", search_domain_filter=["news.example"]
        )

    assert not route.called


def test_empty_domain_filter_is_not_a_restriction(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(SEARCH_URL).respond(json=EMPTY_RESPONSE)

    response: Final = litellm.search(query="query", search_provider="bigdata", api_key="key", search_domain_filter=[])

    assert response.results == []


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "{}",
        '{"results": null, "external_results": {}}',
        '{"results": [{}], "external_results": {}}',
        '{"results": [], "external_results": {"web": [{"headline": "no chunks"}]}}',
    ],
)
def test_malformed_response_is_not_reported_as_empty_success(respx_mock: respx.MockRouter, body: str) -> None:
    respx_mock.post(SEARCH_URL).respond(text=body)

    with pytest.raises(openai.APIError, match="invalid search response") as error:
        litellm.search(query="query", search_provider="bigdata", api_key="key")

    assert error.value.status_code == 502


@pytest.mark.parametrize("status", [400, 401, 429, 503])
def test_upstream_http_errors_keep_status(respx_mock: respx.MockRouter, status: int) -> None:
    respx_mock.post(SEARCH_URL).respond(status_code=status, json={"errors": [{"message": "upstream failure"}]})

    with pytest.raises(openai.APIError, match="upstream failure") as error:
        litellm.search(query="query", search_provider="bigdata", api_key="key")

    assert error.value.status_code == status


def test_upstream_fields_cannot_replace_standard_search_fields(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(SEARCH_URL).respond(
        json={
            "results": [
                {
                    "headline": "Headline",
                    "url": "https://news.example",
                    "timestamp": "2026-11-01T00:00:00Z",
                    "chunks": [{"cnum": 1, "text": "Body"}],
                    "title": "upstream",
                    "snippet": "upstream",
                    "date": "upstream",
                    "last_updated": "upstream",
                }
            ],
            "external_results": {},
        }
    )

    response: Final = litellm.search(query="query", search_provider="bigdata", api_key="key")

    assert [(r.title, r.snippet, r.date, r.last_updated) for r in response.results] == [
        ("Headline", "Body", "2026-11-01T00:00:00Z", None)
    ]


def test_environment_key_cannot_be_sent_to_caller_selected_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIGDATA_API_KEY", "server-key")

    with pytest.raises(ValueError, match="Refusing to send"):
        BigdataSearchConfig().validate_environment({}, api_base="https://untrusted.example/v1")


def test_environment_key_follows_operator_configured_host(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIGDATA_API_KEY", "server-key")
    monkeypatch.setenv("BIGDATA_API_BASE", "https://bigdata-proxy.example/v1")
    route: Final = respx_mock.post("https://bigdata-proxy.example/v1/search").respond(json=EMPTY_RESPONSE)

    litellm.search(query="query", search_provider="bigdata")

    assert route.calls.last.request.headers["X-API-KEY"] == "server-key"


def test_missing_key_fails_before_network() -> None:
    with pytest.raises(ValueError, match="BIGDATA_API_KEY"):
        BigdataSearchConfig().validate_environment({})


@pytest.mark.parametrize("tool_parameter", ["model", "search_tool_name"])
async def test_router_resolves_named_bigdata_tool_and_forwards_its_configuration(
    respx_mock: respx.MockRouter, tool_parameter: str
) -> None:
    route: Final = respx_mock.post("https://bigdata.example/v1/search").respond(
        json={
            "results": [
                {
                    "headline": "Source",
                    "url": "https://news.example",
                    "timestamp": "2026-11-01T00:00:00Z",
                    "chunks": [{"cnum": 1, "text": "Grounding"}],
                }
            ],
            "external_results": {},
        }
    )
    router: Final = litellm.Router(
        model_list=[],
        search_tools=[
            {
                "search_tool_name": "bigdata-search",
                "litellm_params": {
                    "search_provider": "bigdata",
                    "api_key": "tool-key",
                    "api_base": "https://bigdata.example/v1",
                    "search_mode": "smart",
                },
            }
        ],
        num_retries=0,
    )

    response: Final = await router.asearch(**{tool_parameter: "bigdata-search"}, query="ground this", max_results=2)

    assert isinstance(response, SearchResponse)
    assert [(r.title, r.url, r.snippet) for r in response.results] == [("Source", "https://news.example", "Grounding")]
    assert route.calls.last.request.headers["X-API-KEY"] == "tool-key"
    assert json.loads(route.calls.last.request.content) == {
        "search_mode": "smart",
        "query": {"text": "ground this", "max_chunks": 2},
    }
