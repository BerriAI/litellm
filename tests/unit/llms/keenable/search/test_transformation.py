import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx

import litellm
from litellm.llms.keenable.search.transformation import KeenableSearchConfig

KEENABLE_SEARCH_URL: Final = "https://api.keenable.ai/v1/search"


@pytest.fixture
def keenable_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "test-keenable-key")
    monkeypatch.delenv("KEENABLE_API_BASE", raising=False)


@pytest.mark.usefixtures("keenable_api_key")
def test_search_maps_keenable_results_to_search_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(KEENABLE_SEARCH_URL).respond(
        json={
            "query": "typescript best practices",
            "mode": "pro",
            "results": [
                {
                    "title": "TypeScript Best Practices 2026",
                    "url": "https://example.com/ts-best-practices",
                    "description": "",
                    "snippet": "Use strict mode and prefer interfaces.",
                    "published_at": "2026-01-15T10:30:00Z",
                    "acquired_at": "2026-01-16T08:12:34Z",
                },
                {
                    "title": "Second",
                    "url": "https://example.com/second",
                    "snippet": "Second snippet.",
                },
            ],
        }
    )

    response: Final = litellm.search(query="typescript best practices", search_provider="keenable")

    assert response.object == "search"
    assert len(response.results) == 2
    first: Final = response.results[0]
    assert first.title == "TypeScript Best Practices 2026"
    assert first.url == "https://example.com/ts-best-practices"
    assert first.snippet == "Use strict mode and prefer interfaces."
    assert first.date == "2026-01-15T10:30:00Z"
    assert first.last_updated == "2026-01-16T08:12:34Z"
    assert response.results[1].url == "https://example.com/second"
    assert response.model_dump(mode="json")["provider_metadata"] == {
        "query": "typescript best practices",
        "mode": "pro",
    }


@pytest.mark.usefixtures("keenable_api_key")
def test_search_sends_unified_params_and_api_key(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(KEENABLE_SEARCH_URL).respond(
        json={"results": [{"title": "ML", "url": "https://example.com/ml", "snippet": "Machine learning."}]}
    )

    response: Final = litellm.search(
        query="machine learning",
        search_provider="keenable",
        max_results=5,
        search_domain_filter=["arxiv.org", "nature.com"],
        max_tokens_per_page=50,
    )

    assert json.loads(route.calls.last.request.content) == {
        "query": "machine learning",
        "max_results": 5,
        "site": "arxiv.org",
    }
    assert route.calls.last.request.headers["x-api-key"] == "test-keenable-key"
    assert [result.url for result in response.results] == ["https://example.com/ml"]


@pytest.mark.usefixtures("keenable_api_key")
def test_search_ignores_max_tokens_per_page(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(KEENABLE_SEARCH_URL).respond(json={"results": []})

    litellm.search(query="q", search_provider="keenable", max_tokens_per_page=50)

    assert "snippet_max_length" not in json.loads(route.calls.last.request.content)


@pytest.mark.usefixtures("keenable_api_key")
def test_search_joins_list_query_and_passes_keenable_params(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(KEENABLE_SEARCH_URL).respond(json={"results": []})

    litellm.search(
        query=["AI news", "ML trends"],
        search_provider="keenable",
        mode="realtime",
        published_after="7d",
    )

    assert json.loads(route.calls.last.request.content) == {
        "query": "AI news ML trends",
        "mode": "realtime",
        "published_after": "7d",
    }


def test_transform_search_response_falls_back_to_description_when_snippet_missing() -> None:
    response: Final = KeenableSearchConfig().transform_search_response(
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Legacy",
                        "url": "https://example.com/legacy",
                        "description": "old description field",
                    }
                ]
            },
        ),
        logging_obj=Mock(),
    )

    assert response.results[0].snippet == "old description field"


def test_validate_environment_uses_title_header_when_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("KEENABLE_TITLE", raising=False)

    headers: Final = KeenableSearchConfig().validate_environment(headers={})

    assert headers["X-Keenable-Title"] == "LiteLLM"
    assert "X-API-Key" not in headers


@pytest.mark.usefixtures("keenable_api_key")
@pytest.mark.parametrize(
    "api_base",
    [None, "https://api.keenable.ai/v1", "https://api.keenable.ai/v1/"],
)
def test_get_complete_url_appends_search_once(api_base: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEENABLE_API_BASE", raising=False)

    assert KeenableSearchConfig().get_complete_url(
        api_base=api_base, optional_params={}, api_key="test-keenable-key"
    ) == ("https://api.keenable.ai/v1/search")


def test_get_complete_url_uses_public_route_when_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("KEENABLE_API_BASE", raising=False)

    assert KeenableSearchConfig().get_complete_url(api_base=None, optional_params={}) == (
        "https://api.keenable.ai/v1/search/public"
    )


def test_search_keyless_hits_public_route_with_title_header(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("KEENABLE_API_BASE", raising=False)
    monkeypatch.delenv("KEENABLE_TITLE", raising=False)
    route: Final = respx_mock.post("https://api.keenable.ai/v1/search/public").respond(
        json={"results": [{"title": "T", "url": "https://example.com", "snippet": "S"}]}
    )

    response: Final = litellm.search(query="q", search_provider="keenable")

    assert route.calls.last.request.headers["x-keenable-title"] == "LiteLLM"
    assert "x-api-key" not in {k.lower(): v for k, v in route.calls.last.request.headers.items()}
    assert [result.url for result in response.results] == ["https://example.com"]

