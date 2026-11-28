import json
from collections.abc import Iterator
from typing import Final

import openai
import pytest
import respx

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.clawifi.search.transformation import ClawifiSearchConfig

SEARCH_URL: Final = f"{ClawifiSearchConfig.DEFAULT_API_BASE}/search/free"
EMPTY_SEARCH: Final = {"success": True, "results": []}


@pytest.fixture(autouse=True)
def clawifi_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("CLAWIFI_API_KEY", raising=False)
    monkeypatch.delenv("CLAWIFI_API_BASE", raising=False)
    monkeypatch.setattr(  # test-quality-ok: select HTTPX so respx can intercept the external HTTP boundary
        litellm, "disable_aiohttp_transport", True
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def test_search_maps_unified_params_and_results(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(SEARCH_URL).respond(
        json={
            "success": True,
            "query": "model context protocol",
            "provider": "free_auto",
            "results": [
                {
                    "provider": "wikipedia",
                    "title": "Model Context Protocol",
                    "url": "https://example.com/mcp",
                    "snippet": "An open protocol",
                    "rank": 1,
                    "published_at": "2026-01-02",
                    "modified_at": "2026-03-04T05:06:07Z",
                    "data": {"engines": ["wikipedia"]},
                },
                {"provider": "wikipedia", "title": None, "url": "https://example.org/", "snippet": None, "rank": 2},
            ],
            "errors": [],
            "cache_hit": False,
        }
    )

    response: Final = litellm.search(
        query="model context protocol",
        search_provider="clawifi",
        api_key="test-key",
        max_results=2,
        country="de",
        search_domain_filter=["example.com", "example.org", "-excluded.example"],
        max_tokens_per_page=512,
        time_range="month",
        language="en",
    )

    assert json.loads(route.calls.last.request.content) == {
        "query": "model context protocol",
        "max_results": 2,
        "country": "de",
        "include_domains": ["example.com", "example.org"],
        "exclude_domains": ["excluded.example"],
        "time_range": "month",
        "language": "en",
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"
    assert response.model_dump() == {
        "object": "search",
        "results": [
            {
                "title": "Model Context Protocol",
                "url": "https://example.com/mcp",
                "snippet": "An open protocol",
                "date": "2026-01-02",
                "last_updated": "2026-03-04T05:06:07Z",
                "provider": "wikipedia",
                "rank": 1,
                "data": {"engines": ["wikipedia"]},
            },
            {
                "title": "",
                "url": "https://example.org/",
                "snippet": "",
                "date": None,
                "last_updated": None,
                "provider": "wikipedia",
                "rank": 2,
            },
        ],
        "provider_metadata": {
            "success": True,
            "query": "model context protocol",
            "provider": "free_auto",
            "errors": [],
            "cache_hit": False,
        },
    }


@pytest.mark.parametrize(
    ("params", "expected_body"),
    [
        ({}, {"query": "q"}),
        ({"search_domain_filter": ["", "-"]}, {"query": "q"}),
        ({"search_domain_filter": ["-excluded.example"]}, {"query": "q", "exclude_domains": ["excluded.example"]}),
        (
            {"search_domain_filter": ["derived.example"], "include_domains": ["native.example"]},
            {"query": "q", "include_domains": ["native.example"]},
        ),
        ({"max_tokens_per_page": 100, "allow_browser_fallback": True}, {"query": "q", "allow_browser_fallback": True}),
    ],
)
def test_request_body(
    respx_mock: respx.MockRouter, params: dict[str, object], expected_body: dict[str, object]
) -> None:
    route: Final = respx_mock.post(SEARCH_URL).respond(json=EMPTY_SEARCH)

    litellm.search(query="q", search_provider="clawifi", api_key="key", **params)

    assert json.loads(route.calls.last.request.content) == expected_body


@pytest.mark.parametrize("api_base", [None, "https://api.clawifi.dev", "https://api.clawifi.dev/search/free/"])
async def test_asearch_sends_the_environment_key_to_the_default_endpoint(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, api_base: str | None
) -> None:
    monkeypatch.setenv("CLAWIFI_API_KEY", "environment-key")
    route: Final = respx_mock.post(SEARCH_URL).respond(json=EMPTY_SEARCH)

    response: Final = await litellm.asearch(query=["first", "second"], search_provider="clawifi", api_base=api_base)

    assert response.model_dump() == {
        "object": "search",
        "results": [],
        "provider_metadata": {"success": True, "errors": []},
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer environment-key"
    assert json.loads(route.calls.last.request.content) == {"query": "first second"}


async def test_operator_api_base_receives_the_environment_key(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAWIFI_API_KEY", "environment-key")
    monkeypatch.setenv("CLAWIFI_API_BASE", "https://clawifi.internal.example/")
    route: Final = respx_mock.post("https://clawifi.internal.example/search/free").respond(json=EMPTY_SEARCH)

    await litellm.asearch(query="q", search_provider="clawifi")

    assert route.calls.last.request.headers["authorization"] == "Bearer environment-key"


@pytest.mark.parametrize(
    ("environment_key", "api_base", "api_key", "error"),
    [
        (None, None, None, "CLAWIFI_API_KEY"),
        ("server-key", "https://untrusted.example", None, "Refusing to send"),
        ("server-key", "http://api.clawifi.dev", None, "HTTPS"),
        (None, "http://api.clawifi.dev", "caller-key", "HTTPS"),
    ],
)
def test_request_is_refused_before_a_key_leaves_the_proxy(
    monkeypatch: pytest.MonkeyPatch, environment_key: str | None, api_base: str | None, api_key: str | None, error: str
) -> None:
    if environment_key is not None:
        monkeypatch.setenv("CLAWIFI_API_KEY", environment_key)
    with respx.mock(assert_all_called=False) as transport:
        transport.route().respond(json=EMPTY_SEARCH)
        with pytest.raises(openai.APIError, match=error):
            litellm.search(query="q", search_provider="clawifi", api_base=api_base, api_key=api_key)
        assert len(transport.calls) == 0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            '{"success": false, "results": [], "errors": [{"code": "timeout", "message": "engine timed out"},'
            ' {"code": "blocked", "message": "engine refused"}]}',
            "Clawifi search failed: engine timed out; engine refused",
        ),
        ('{"success": false, "results": []}', "Clawifi search failed: no engine answered"),
        ("not json", "invalid search response"),
        ('{"success": true}', "invalid search response"),
        ('{"success": true, "results": [{"title": "no url"}]}', "invalid search response"),
    ],
)
def test_failed_or_malformed_search_is_not_reported_as_empty_success(
    respx_mock: respx.MockRouter, body: str, message: str
) -> None:
    respx_mock.post(SEARCH_URL).respond(text=body)
    with pytest.raises(openai.APIError, match=message) as error:
        litellm.search(query="q", search_provider="clawifi", api_key="key")
    assert error.value.status_code == 502


@pytest.mark.parametrize("status", [401, 402, 429, 503])
def test_upstream_http_errors_keep_their_status(respx_mock: respx.MockRouter, status: int) -> None:
    respx_mock.post(SEARCH_URL).respond(
        status_code=status,
        json={"detail": "upstream failure", "error": {"code": "http_error", "message": "upstream failure"}},
        headers={"Retry-After": "2"},
    )
    with pytest.raises(openai.APIError, match="upstream failure") as error:
        litellm.search(query="q", search_provider="clawifi", api_key="key")
    assert error.value.status_code == status


async def test_router_forwards_the_named_tool_configuration(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(SEARCH_URL).respond(
        json={"success": True, "results": [{"url": "https://example.com", "title": "Source", "snippet": "Passage"}]}
    )
    router: Final = litellm.Router(
        model_list=[],
        search_tools=[
            {
                "search_tool_name": "clawifi-search",
                "litellm_params": {"search_provider": "clawifi", "api_key": "tool-key", "time_range": "week"},
            }
        ],
        num_retries=0,
    )

    response: Final = await router.asearch(search_tool_name="clawifi-search", query="ground this", max_results=2)

    assert isinstance(response, SearchResponse)
    assert [(result.title, result.url, result.snippet) for result in response.results] == [
        ("Source", "https://example.com", "Passage")
    ]
    assert route.calls.last.request.headers["authorization"] == "Bearer tool-key"
    assert json.loads(route.calls.last.request.content) == {
        "query": "ground this",
        "max_results": 2,
        "time_range": "week",
    }
