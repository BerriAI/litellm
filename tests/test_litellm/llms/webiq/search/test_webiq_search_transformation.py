import json
from collections.abc import Iterator
from typing import Final

import openai
import pytest
import respx

import litellm
from litellm.llms.webiq.search.transformation import WebIQSearchConfig


@pytest.fixture(autouse=True)
def webiq_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("WEBIQ_API_KEY", raising=False)
    monkeypatch.setattr(  # test-quality-ok: select HTTPX so respx can intercept the external HTTP boundary
        litellm, "disable_aiohttp_transport", True
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def test_search_maps_request_and_preserves_sources(respx_mock: respx.MockRouter) -> None:
    endpoint: Final = f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web"
    route: Final = respx_mock.post(endpoint).respond(
        json={
            "webResults": [
                {
                    "title": "A source",
                    "url": "https://example.com/article",
                    "content": "Relevant passage",
                    "lastUpdatedAt": "2026-01-02T00:00:00Z",
                    "crawledAt": "2026-01-03T00:00:00Z",
                    "language": "en",
                    "contentTier": "standard",
                    "instrumentationSuffix": "source-1",
                }
            ],
            "traceId": "trace-1",
            "querySignals": {"freshness": True},
        }
    )

    response: Final = litellm.search(
        query="RAG research",
        search_provider="webiq",
        api_key="test-key",
        max_results=3,
        country="gb",
        search_domain_filter=["example.com", "example.org", "-excluded.example"],
        language="en",
    )

    assert json.loads(route.calls.last.request.content) == {
        "query": "(RAG research) (site:example.com OR site:example.org) -site:excluded.example",
        "maxResults": 3,
        "region": "GB",
        "contentFormat": "passage",
        "maxLength": 5000,
        "language": "en",
    }
    assert route.calls.last.request.headers["x-apikey"] == "test-key"
    assert response.model_dump() == {
        "object": "search",
        "results": [
            {
                "title": "A source",
                "url": "https://example.com/article",
                "snippet": "Relevant passage",
                "date": "2026-01-02T00:00:00Z",
                "last_updated": "2026-01-02T00:00:00Z",
                "crawledAt": "2026-01-03T00:00:00Z",
                "language": "en",
                "contentTier": "standard",
                "instrumentationSuffix": "source-1",
            }
        ],
        "traceId": "trace-1",
        "querySignals": {"freshness": True},
    }


async def test_asearch_uses_environment_credentials_at_default_endpoint(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEBIQ_API_KEY", "environment-key")
    route: Final = respx_mock.post(f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web").respond(json={"webResults": []})

    response: Final = await litellm.asearch(query="no matches", search_provider="webiq")

    assert response.model_dump() == {"object": "search", "results": []}
    assert route.calls.last.request.headers["x-apikey"] == "environment-key"
    assert json.loads(route.calls.last.request.content) == {
        "query": "no matches",
        "contentFormat": "passage",
        "maxLength": 5000,
    }


async def test_asearch_uses_explicit_credentials_at_custom_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://webiq.example/v3/search/web").respond(json={"webResults": []})

    response: Final = await litellm.asearch(
        query="no matches", search_provider="webiq", api_key="custom-key", api_base="https://webiq.example/v3/"
    )

    assert response.model_dump() == {"object": "search", "results": []}
    assert route.calls.last.request.headers["x-apikey"] == "custom-key"
    assert json.loads(route.calls.last.request.content) == {
        "query": "no matches",
        "contentFormat": "passage",
        "maxLength": 5000,
    }


@pytest.mark.parametrize("updated", [None, "", "2026-02-01T00:00:00Z"])
def test_optional_updated_date_never_uses_crawl_time(respx_mock: respx.MockRouter, updated: str | None) -> None:
    respx_mock.post(f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web").respond(
        json={
            "webResults": [
                {
                    "title": "Title",
                    "url": "https://example.com",
                    "content": "Passage",
                    "lastUpdatedAt": updated,
                    "crawledAt": "2026-03-01T00:00:00Z",
                }
            ]
        }
    )
    response: Final = litellm.search(query="query", search_provider="webiq", api_key="key")
    assert (response.results[0].date, response.results[0].last_updated) == (updated or None, updated or None)


def test_missing_optional_dates_and_result_order(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web").respond(
        json={
            "webResults": [
                {"title": title, "url": f"https://example.com/{title}", "content": title}
                for title in ("first", "second")
            ]
        }
    )
    response: Final = litellm.search(query="query", search_provider="webiq", api_key="key")
    assert [(r.title, r.snippet, r.date, r.last_updated) for r in response.results] == [
        ("first", "first", None, None),
        ("second", "second", None, None),
    ]


@pytest.mark.parametrize("body", ["not json", "{}", '{"webResults": null}', '{"webResults": [{}]}'])
def test_malformed_response_is_not_reported_as_empty_success(respx_mock: respx.MockRouter, body: str) -> None:
    respx_mock.post(f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web").respond(text=body)
    with pytest.raises(openai.APIError, match="invalid search response") as error:
        litellm.search(query="query", search_provider="webiq", api_key="key")
    assert error.value.status_code == 502


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_upstream_http_errors_keep_status(respx_mock: respx.MockRouter, status: int) -> None:
    respx_mock.post(f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web").respond(
        status_code=status, json={"error": {"message": "upstream failure"}}, headers={"Retry-After": "2"}
    )
    with pytest.raises(openai.APIError, match="upstream failure") as error:
        litellm.search(query="query", search_provider="webiq", api_key="key")
    assert error.value.status_code == status


def test_native_options_override_mapped_defaults_without_mutating_input() -> None:
    params: Final = {
        "max_results": 3,
        "maxResults": 7,
        "country": "us",
        "region": "JP",
        "maxLength": 2000,
        "max_tokens_per_page": 100,
        "contentFormat": "markdown",
        "safeSearch": "strict",
    }
    before: Final = params.copy()
    assert WebIQSearchConfig().transform_search_request(["first", "second"], params) == {
        "query": "first second",
        "maxResults": 7,
        "region": "JP",
        "maxLength": 2000,
        "contentFormat": "markdown",
        "safeSearch": "strict",
    }
    assert params == before


@pytest.mark.parametrize(
    "domains, expected",
    [
        ([], "query"),
        ([""], "query"),
        (["-"], "query"),
        (["-example.com"], "(query) -site:example.com"),
        (["example.com"], "(query) (site:example.com)"),
    ],
)
def test_domain_filter_queries(domains: list[str], expected: str) -> None:
    assert WebIQSearchConfig().transform_search_request("query", {"search_domain_filter": domains}) == {
        "query": expected,
        "contentFormat": "passage",
        "maxLength": 5000,
    }


@pytest.mark.parametrize("suffix", ["", "/", "/search/web", "/search/web/"])
def test_endpoint_is_appended_once(suffix: str) -> None:
    assert WebIQSearchConfig().get_complete_url(f"https://webiq.example/v3{suffix}", {}) == (
        "https://webiq.example/v3/search/web"
    )


def test_environment_key_cannot_be_sent_to_caller_selected_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBIQ_API_KEY", "server-key")
    with pytest.raises(ValueError, match="Refusing to send"):
        WebIQSearchConfig().validate_environment({}, api_base="https://untrusted.example/v3")


def test_explicit_key_wins_and_headers_are_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBIQ_API_KEY", "server-key")
    config: Final = WebIQSearchConfig()
    original: Final = {"X-Request-ID": "request-1"}
    result: Final = config.validate_environment(original, api_key="caller-key", api_base="https://custom.example/v3")
    assert result == {"X-Request-ID": "request-1", "x-apikey": "caller-key", "Content-Type": "application/json"}
    assert original == {"X-Request-ID": "request-1"}
    assert config.validate_environment(result, api_key="caller-key") == result


def test_missing_key_fails_before_network() -> None:
    with pytest.raises(ValueError, match="WEBIQ_API_KEY"):
        WebIQSearchConfig().validate_environment({})


@pytest.mark.parametrize(
    "result_metadata, response_metadata",
    [
        ({"snippet": "upstream", "date": "upstream", "last_updated": "upstream"}, {}),
        ({}, {"results": "upstream", "object": "upstream"}),
    ],
)
def test_metadata_cannot_replace_standard_search_fields(
    respx_mock: respx.MockRouter, result_metadata: dict[str, str], response_metadata: dict[str, str]
) -> None:
    respx_mock.post(f"{WebIQSearchConfig.DEFAULT_API_BASE}/search/web").respond(
        json={
            "webResults": [
                {
                    "title": "Title",
                    "url": "https://example.com",
                    "content": "Passage",
                    "lastUpdatedAt": "2026-01-02T00:00:00Z",
                    "language": "en",
                    **result_metadata,
                }
            ],
            "traceId": "trace-1",
            **response_metadata,
        }
    )

    response: Final = litellm.search(query="query", search_provider="webiq", api_key="key")

    assert response.model_dump() == {
        "object": "search",
        "results": [
            {
                "title": "Title",
                "url": "https://example.com",
                "snippet": "Passage",
                "date": "2026-01-02T00:00:00Z",
                "last_updated": "2026-01-02T00:00:00Z",
                "language": "en",
            }
        ],
        "traceId": "trace-1",
    }
