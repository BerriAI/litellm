import json
from collections.abc import Iterator
from typing import Final

import openai
import pytest
import respx

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.seltz.search.transformation import SeltzSearchConfig

SELTZ_SEARCH_URL: Final = "https://api.seltz.ai/v1/search"


@pytest.fixture(autouse=True)
def seltz_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("SELTZ_API_KEY", raising=False)
    monkeypatch.delenv("SELTZ_API_BASE", raising=False)
    monkeypatch.setattr(  # test-quality-ok: select HTTPX so respx can intercept the external HTTP boundary
        litellm, "disable_aiohttp_transport", True
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def test_search_maps_unified_params_and_keeps_seltz_fields(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(SELTZ_SEARCH_URL).respond(
        json={
            "request_id": "req-1",
            "documents": [
                {
                    "url": "https://example.com/filing",
                    "content": "Revenue grew.",
                    "published_date": "2025-10-28T00:00:00Z",
                    "snippets": [{"text": "Revenue grew.", "score": 0.9}],
                    "properties": {"company": {"name": "Example"}},
                    "language": "en",
                },
                {
                    "url": "https://example.com/other",
                    "content": None,
                    "snippets": [{"text": "First passage"}, {"text": None}, {"text": "Second passage"}],
                },
            ],
        }
    )
    params: Final = {
        "max_results": 3,
        "search_domain_filter": ["example.com"],
        "country": "US",
        "max_tokens_per_page": 512,
        "scope": "news",
        "tier": "  pro  ",
        "exclude_domains": ["spam.example"],
        "from_date": "now-7d",
        "to_date": "2025-10-28",
        "fields": {"snippets": True},
    }
    before: Final = {key: value for key, value in params.items()}

    response: Final = litellm.search(
        query=["seltz", "pricing"],
        search_provider="seltz",
        api_key="caller-key",
        **params,
    )

    assert params == before
    assert route.calls.last.request.headers["x-api-key"] == "caller-key"
    assert route.calls.last.request.headers["content-type"] == "application/json"
    assert json.loads(route.calls.last.request.content) == {
        "query": "seltz pricing",
        "max_results": 3,
        "include_domains": ["example.com"],
        "scope": "news",
        "tier": "  pro  ",
        "exclude_domains": ["spam.example"],
        "from_date": "now-7d",
        "to_date": "2025-10-28",
        "fields": {"snippets": True},
    }
    assert response.model_dump() == {
        "object": "search",
        "request_id": "req-1",
        "results": [
            {
                "title": "",
                "url": "https://example.com/filing",
                "snippet": "Revenue grew.",
                "date": "2025-10-28T00:00:00Z",
                "last_updated": None,
                "content": "Revenue grew.",
                "snippets": [{"text": "Revenue grew.", "score": 0.9}],
                "properties": {"company": {"name": "Example"}},
                "language": "en",
            },
            {
                "title": "",
                "url": "https://example.com/other",
                "snippet": "First passage\nSecond passage",
                "date": None,
                "last_updated": None,
                "snippets": [{"text": "First passage"}, {"text": None}, {"text": "Second passage"}],
            },
        ],
    }


def test_native_include_domains_overrides_the_unified_filter_and_a_query_string_is_sent() -> None:
    params: Final = {
        "search_domain_filter": ["unified.example"],
        "include_domains": ["native.example"],
        "country": "US",
        "max_tokens_per_page": 10,
        "api_key": "must-not-be-in-the-body",
    }
    before: Final = params.copy()

    assert SeltzSearchConfig().transform_search_request("how much is the fish", params) == {
        "query": "how much is the fish",
        "include_domains": ["native.example"],
    }
    assert params == before


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, SELTZ_SEARCH_URL),
        ("https://api.seltz.ai", SELTZ_SEARCH_URL),
        ("https://api.seltz.ai/", SELTZ_SEARCH_URL),
        ("https://api.seltz.ai/v1/search", SELTZ_SEARCH_URL),
        ("https://api.seltz.ai/v1/search/", SELTZ_SEARCH_URL),
        ("https://gateway.example/seltz", "https://gateway.example/seltz/v1/search"),
        ("https://gateway.example/seltz/", "https://gateway.example/seltz/v1/search"),
    ],
)
def test_search_url_is_appended_once(api_base: str | None, expected: str) -> None:
    assert SeltzSearchConfig().get_complete_url(api_base, {}) == expected


def test_environment_base_and_key_are_used_when_the_caller_omits_them(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SELTZ_API_KEY", "environment-key")
    monkeypatch.setenv("SELTZ_API_BASE", "https://seltz.example")
    route: Final = respx_mock.post("https://seltz.example/v1/search").respond(json={"documents": []})

    response: Final = litellm.search(query="no matches", search_provider="seltz")

    assert response.model_dump() == {"object": "search", "results": []}
    assert route.calls.last.request.headers["x-api-key"] == "environment-key"
    assert json.loads(route.calls.last.request.content) == {"query": "no matches"}


async def test_asearch_uses_an_explicit_key_at_a_custom_base(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://gateway.example/seltz/v1/search").respond(
        json={"documents": [], "request_id": "req-2"}
    )

    response: Final = await litellm.asearch(
        query="no matches",
        search_provider="seltz",
        api_key="custom-key",
        api_base="https://gateway.example/seltz/",
    )

    assert response.model_dump() == {"object": "search", "results": [], "request_id": "req-2"}
    assert route.calls.last.request.headers["x-api-key"] == "custom-key"


def test_environment_key_cannot_be_sent_to_a_caller_selected_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELTZ_API_KEY", "server-key")
    with pytest.raises(ValueError, match="Refusing to send"):
        SeltzSearchConfig().validate_environment({}, api_base="https://untrusted.example")


def test_explicit_key_wins_and_headers_are_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELTZ_API_KEY", "server-key")
    original: Final = {"X-Request-ID": "request-1"}

    result: Final = SeltzSearchConfig().validate_environment(
        original, api_key="caller-key", api_base="https://custom.example"
    )

    assert result == {"X-Request-ID": "request-1", "x-api-key": "caller-key", "Content-Type": "application/json"}
    assert original == {"X-Request-ID": "request-1"}


def test_missing_key_fails_before_network() -> None:
    with pytest.raises(ValueError, match="SELTZ_API_KEY"):
        SeltzSearchConfig().validate_environment({})


@pytest.mark.parametrize(
    "body",
    ["not json", '{"documents": null}', '{"documents": ["nope"]}', '{"documents": [{"url": 7}]}'],
)
def test_malformed_response_is_not_reported_as_empty_success(respx_mock: respx.MockRouter, body: str) -> None:
    respx_mock.post(SELTZ_SEARCH_URL).respond(text=body)
    with pytest.raises(openai.APIError, match="invalid search response") as error:
        litellm.search(query="query", search_provider="seltz", api_key="key")
    assert error.value.status_code == 502


@pytest.mark.parametrize("body", [{}, {"request_id": "req-3"}, {"documents": []}])
def test_response_without_documents_is_empty(respx_mock: respx.MockRouter, body: dict[str, object]) -> None:
    respx_mock.post(SELTZ_SEARCH_URL).respond(json=body)
    response: Final = litellm.search(query="query", search_provider="seltz", api_key="key")
    assert response.results == []
    assert response.object == "search"


@pytest.mark.parametrize("status", [401, 402, 404, 429])
def test_upstream_http_errors_keep_status(respx_mock: respx.MockRouter, status: int) -> None:
    respx_mock.post(SELTZ_SEARCH_URL).respond(
        status_code=status,
        json={"error": {"code": "UNAUTHENTICATED", "message": "upstream failure"}},
        headers={"Retry-After": "2"},
    )
    with pytest.raises(openai.APIError, match="upstream failure") as error:
        litellm.search(query="query", search_provider="seltz", api_key="key")
    assert error.value.status_code == status


def test_document_metadata_cannot_replace_standard_result_fields(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(SELTZ_SEARCH_URL).respond(
        json={
            "documents": [
                {
                    "url": "https://example.com",
                    "content": "Passage",
                    "published_date": "2025-01-02T00:00:00Z",
                    "snippet": "upstream",
                    "title": "upstream",
                    "date": "upstream",
                    "last_updated": "upstream",
                }
            ],
            "results": "upstream",
            "object": "upstream",
            "trace": "trace-1",
        }
    )

    response: Final = litellm.search(query="query", search_provider="seltz", api_key="key")

    assert isinstance(response, SearchResponse)
    assert response.model_dump() == {
        "object": "search",
        "trace": "trace-1",
        "results": [
            {
                "title": "",
                "url": "https://example.com",
                "snippet": "Passage",
                "date": "2025-01-02T00:00:00Z",
                "last_updated": None,
                "content": "Passage",
            }
        ],
    }


@pytest.mark.parametrize("tool_parameter", ["model", "search_tool_name"])
async def test_router_resolves_named_seltz_tool_and_forwards_its_configuration(
    respx_mock: respx.MockRouter, tool_parameter: str
) -> None:
    route: Final = respx_mock.post("https://seltz.example/v1/search").respond(
        json={"documents": [{"url": "https://example.com", "content": "Grounding"}]}
    )
    router: Final = litellm.Router(
        model_list=[],
        search_tools=[
            {
                "search_tool_name": "seltz-search",
                "litellm_params": {
                    "search_provider": "seltz",
                    "api_key": "tool-key",
                    "api_base": "https://seltz.example",
                    "scope": "web",
                },
            }
        ],
        num_retries=0,
    )

    response: Final = await router.asearch(**{tool_parameter: "seltz-search"}, query="ground this", max_results=2)

    assert isinstance(response, SearchResponse)
    assert response.results[0].snippet == "Grounding"
    assert route.calls.last.request.headers["x-api-key"] == "tool-key"
    assert json.loads(route.calls.last.request.content) == {
        "query": "ground this",
        "max_results": 2,
        "scope": "web",
    }
