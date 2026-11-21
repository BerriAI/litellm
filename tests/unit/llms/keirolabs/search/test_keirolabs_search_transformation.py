import json
from collections.abc import Iterator
from typing import Final

import openai
import pytest
import respx

import litellm
from litellm.llms.keirolabs.search.transformation import KeirolabsSearchConfig

ENDPOINT: Final = "https://api.keirolabs.cloud/api/v2/search/lite"


@pytest.fixture(autouse=True)
def keirolabs_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("KEIROLABS_API_KEY", raising=False)
    monkeypatch.delenv("KEIROLABS_API_BASE", raising=False)
    monkeypatch.setattr(  # test-quality-ok: select HTTPX so respx can intercept the external HTTP boundary
        litellm, "disable_aiohttp_transport", True
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def test_search_maps_request_and_results(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).respond(
        json={
            "query": "asyncio",
            "count": 2,
            "type": "lite",
            "creditsCharged": 0.0375,
            "results": [
                {
                    "title": "asyncio docs",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "snippet": "asyncio is a library to write concurrent code",
                    "domain": "docs.python.org",
                    "position": 1,
                    "score": None,
                },
                {"title": "No snippet", "url": "https://example.org/", "snippet": None, "position": 2},
            ],
        }
    )

    response: Final = litellm.search(
        query="asyncio",
        search_provider="keirolabs",
        api_key="test-key",
        max_results=5,
        search_domain_filter=["docs.python.org", "example.org"],
    )

    assert json.loads(route.calls.last.request.content) == {
        "query": "(asyncio) (site:docs.python.org OR site:example.org)",
        "maxResults": 5,
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
    assert response.model_dump() == {
        "object": "search",
        "results": [
            {
                "title": "asyncio docs",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "snippet": "asyncio is a library to write concurrent code",
                "date": None,
                "last_updated": None,
            },
            {"title": "No snippet", "url": "https://example.org/", "snippet": "", "date": None, "last_updated": None},
        ],
    }


async def test_asearch_uses_environment_key_and_custom_base(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEIROLABS_API_KEY", "environment-key")
    monkeypatch.setenv("KEIROLABS_API_BASE", "https://keiro.internal.example/")
    route: Final = respx_mock.post("https://keiro.internal.example/api/v2/search/lite").respond(json={"results": []})

    response: Final = await litellm.asearch(query="model context protocol", search_provider="keirolabs")

    assert json.loads(route.calls.last.request.content) == {"query": "model context protocol"}
    assert route.calls.last.request.headers["authorization"] == "Bearer environment-key"
    assert response.results == []


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_plaintext_api_base_is_rejected_before_sending_the_key(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    route: Final = respx_mock.post("http://api.keirolabs.cloud/api/v2/search/lite").respond(json={"results": []})
    if source == "environment":
        monkeypatch.setenv("KEIROLABS_API_BASE", "http://api.keirolabs.cloud")
    api_base: Final = "http://api.keirolabs.cloud" if source == "argument" else None

    with pytest.raises(openai.APIConnectionError, match="HTTPS api_base"):
        litellm.search(query="query", search_provider="keirolabs", api_key="key", api_base=api_base)
    assert not route.called


def test_missing_key_is_reported() -> None:
    with pytest.raises(ValueError, match="KEIROLABS_API_KEY"):
        KeirolabsSearchConfig().validate_environment(headers={})


@pytest.mark.parametrize("body", ["not json", "{}", '{"results": null}', '{"results": [{"title": "no url"}]}'])
def test_malformed_response_is_not_reported_as_empty_success(respx_mock: respx.MockRouter, body: str) -> None:
    respx_mock.post(ENDPOINT).respond(text=body)
    with pytest.raises(openai.APIError, match="invalid search response") as error:
        litellm.search(query="query", search_provider="keirolabs", api_key="key")
    assert error.value.status_code == 502


@pytest.mark.parametrize("status", [401, 402, 429, 503])
def test_upstream_http_errors_keep_status(respx_mock: respx.MockRouter, status: int) -> None:
    respx_mock.post(ENDPOINT).respond(status_code=status, json={"error": "Invalid API key"})
    with pytest.raises(openai.APIError, match="Invalid API key") as error:
        litellm.search(query="query", search_provider="keirolabs", api_key="key")
    assert error.value.status_code == status
