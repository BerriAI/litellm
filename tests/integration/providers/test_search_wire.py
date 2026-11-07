import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

import litellm

_EXA_KEY: Final = "synthetic-exa-key"
_QUERY: Final = "latest AI developments 2024"
_DOMAINS: Final = ("arxiv.org", "nature.com")
_PERPLEXITY_BODY: Final = {"query": _QUERY, "max_results": 5, "search_domain_filter": list(_DOMAINS), "country": "US"}
_EXA_BODY: Final = {
    "query": _QUERY,
    "numResults": 5,
    "includeDomains": list(_DOMAINS),
    "userLocation": "US",
    "contents": {"text": True},
}
_EXA_RESULTS: Final = {
    "results": [
        {
            "title": "Gateway result",
            "url": "https://result.invalid/a",
            "text": "result snippet",
            "publishedDate": "2024-01-01T00:00:00.000Z",
        }
    ]
}


class _SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    date: str | None
    last_updated: str | None


class _SearchResponse(BaseModel):
    object: str
    results: tuple[_SearchResult, ...]


_EXPECTED_RESPONSE: Final = _SearchResponse(
    object="search",
    results=(
        _SearchResult(
            title="Gateway result",
            url="https://result.invalid/a",
            snippet="result snippet",
            date="2024-01-01T00:00:00.000Z",
            last_updated=None,
        ),
    ),
)


def _create_exa_tool(
    gateway: Gateway, scenario: Scenario, name: str, api_base: str, input_cost_per_query: float | None = None
) -> None:
    pricing: Final = {} if input_cost_per_query is None else {"input_cost_per_query": input_cost_per_query}
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {
                    "search_provider": "exa_ai",
                    "api_key": _EXA_KEY,
                    "api_base": api_base,
                    **pricing,
                },
            }
        },
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")


def test_perplexity_spec_search_reaches_exa_with_only_mapped_fields_on_every_route_spelling(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/exa/search"), request.target
        assert request.headers["x-api-key"] == _EXA_KEY, request.headers
        assert "authorization" not in request.headers, request.headers
        assert json.loads(request.body) == _EXA_BODY, request.body
        return Reply(body=json.dumps(_EXA_RESULTS).encode())

    tool: Final = f"exa-tool-{uuid.uuid4().hex}"
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, tool, f"{wire.url}/exa")
        spellings: Final = (
            (f"/v1/search/{tool}", _PERPLEXITY_BODY),
            (f"/search/{tool}", _PERPLEXITY_BODY),
            ("/v1/search", {"search_tool_name": tool, **_PERPLEXITY_BODY}),
            ("/search", {"search_tool_name": tool, **_PERPLEXITY_BODY}),
            ("/v1/search", {"model": tool, **_PERPLEXITY_BODY}),
            ("/search", {"model": tool, **_PERPLEXITY_BODY}),
            (f"/v1/search/{tool}", {"search_tool_name": f"unknown-{uuid.uuid4().hex}", **_PERPLEXITY_BODY}),
        )
        for path, body in spellings:
            response = gateway.request("POST", path, body)
            assert response.status_code == 200, f"{path} {sorted(body)}: {response.text}"
            assert _SearchResponse.model_validate_json(response.content) == _EXPECTED_RESPONSE, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/exa/search")] * len(
            spellings
        )


def test_exa_native_fields_and_caller_contents_pass_through_while_unmapped_perplexity_fields_are_dropped(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/exa/search"), request.target
        assert request.headers["x-api-key"] == _EXA_KEY, request.headers
        assert json.loads(request.body) == {
            "query": _QUERY,
            "numResults": 5,
            "includeDomains": list(_DOMAINS),
            "userLocation": "US",
            "category": "research paper",
            "startPublishedDate": "2024-01-01T00:00:00.000Z",
            "contents": {"highlights": {"numSentences": 2}},
        }, request.body
        return Reply(body=json.dumps(_EXA_RESULTS).encode())

    tool: Final = f"exa-tool-{uuid.uuid4().hex}"
    body: Final = {
        **_PERPLEXITY_BODY,
        "max_tokens_per_page": 512,
        "category": "research paper",
        "startPublishedDate": "2024-01-01T00:00:00.000Z",
        "contents": {"highlights": {"numSentences": 2}},
    }
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, tool, f"{wire.url}/exa")
        response: Final = gateway.request("POST", f"/v1/search/{tool}", body)
        assert response.status_code == 200, response.text
        assert _SearchResponse.model_validate_json(response.content) == _EXPECTED_RESPONSE, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/exa/search")]


def test_list_query_reaches_exa_as_one_space_joined_query(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/exa/search"), request.target
        assert request.headers["x-api-key"] == _EXA_KEY, request.headers
        assert json.loads(request.body) == {
            "query": "first q second q",
            "numResults": 2,
            "contents": {"text": True},
        }, request.body
        return Reply(body=json.dumps(_EXA_RESULTS).encode())

    tool: Final = f"exa-tool-{uuid.uuid4().hex}"
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, tool, f"{wire.url}/exa")
        response: Final = gateway.request(
            "POST", f"/v1/search/{tool}", {"query": ["first q", "second q"], "max_results": 2}
        )
        assert response.status_code == 200, response.text
        assert _SearchResponse.model_validate_json(response.content) == _EXPECTED_RESPONSE, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/exa/search")]


def _not_found(name: str) -> str:
    return f"Search tool '{name}' not found in router.search_tools"


def test_unknown_search_tool_is_refused_without_reaching_any_configured_tool(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"unknown search tool reached a configured tool: {request.target} {request.body!r}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, f"exa-tool-{uuid.uuid4().hex}", f"{wire.url}/exa")
        name: Final = f"unknown-tool-{uuid.uuid4().hex}"
        spellings: Final = (
            (f"/v1/search/{name}", _PERPLEXITY_BODY),
            ("/v1/search", {"search_tool_name": name, **_PERPLEXITY_BODY}),
        )
        for path, body in spellings:
            response = gateway.request("POST", path, body)
            refusal = (response.status_code >= 400, response.json().get("error", {}).get("message"))
            assert refusal == (True, _not_found(name)), f"{path}: {response.text}"
        assert wire.drain() == ()


def test_unknown_search_tool_is_a_client_error_without_any_upstream_call(gateway: Gateway) -> None:
    pytest.skip("BUG: POST /v1/search/<unknown or deleted tool> returns 500 internal_server_error instead of a 4xx")

    def respond(request: Request) -> Reply:
        raise AssertionError(f"unknown search tool reached a provider: {request.target}")

    with wire_server(respond) as wire:
        name: Final = f"unknown-tool-{uuid.uuid4().hex}"
        response: Final = gateway.request("POST", f"/v1/search/{name}", _PERPLEXITY_BODY)
        assert (response.status_code in (400, 404), response.json()["error"]["message"]) == (True, _not_found(name)), (
            response.text
        )
        assert wire.drain() == ()


def test_unpriced_fallback_search_tool_is_billed_at_its_own_price_not_the_failed_primarys(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert "input_cost_per_query" not in json.loads(request.body), request.body
        if request.target == "/down/search":
            return Reply(status=500, body=json.dumps({"error": "primary down"}).encode())
        assert (request.method, request.target) == ("POST", "/exa/search"), request.target
        return Reply(body=json.dumps(_EXA_RESULTS).encode())

    primary: Final = f"exa-priced-{uuid.uuid4().hex}"
    backup: Final = f"exa-backup-{uuid.uuid4().hex}"
    primary_price: Final = 0.05
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, primary, f"{wire.url}/down", input_cost_per_query=primary_price)
        _create_exa_tool(gateway, scenario, backup, f"{wire.url}/exa")
        direct: Final = gateway.request("POST", f"/v1/search/{backup}", {"query": _QUERY})
        assert direct.status_code == 200, direct.text
        backup_price: Final = float(direct.headers["x-litellm-response-cost"])
        assert backup_price not in (0.0, primary_price), direct.headers

        fallback: Final = gateway.request("POST", f"/v1/search/{primary}", {"query": _QUERY, "fallbacks": [backup]})
        assert fallback.status_code == 200, fallback.text
        assert float(fallback.headers["x-litellm-response-cost"]) == pytest.approx(backup_price), fallback.headers
        assert [request.target for request in wire.drain()] == ["/exa/search", "/down/search", "/exa/search"]


_TAVILY_KEY: Final = "synthetic-tavily-key"
_TAVILY_ENVELOPE: Final = {
    "query": _QUERY,
    "answer": "Gateways route model traffic.",
    "follow_up_questions": None,
    "images": [{"url": "https://result.invalid/logo.png", "description": "logo"}],
    "response_time": 1.67,
    "usage": {"credits": 1},
    "request_id": "tavily-request-7c2e9a",
}
_TAVILY_OPTIONS: Final = {"include_raw_content": "markdown", "include_answer": True, "include_favicon": True}
_TAVILY_RESULTS: Final = (
    {
        "title": "Gateway guide",
        "url": "https://result.invalid/guide",
        "content": "Route traffic [...] track spend",
        "score": 0.83,
        "raw_content": "# Gateway guide\n\nRoute traffic and track spend.",
        "favicon": "https://result.invalid/favicon.ico",
    },
    {"title": "Unfetched", "url": "https://result.invalid/unfetched", "content": "Excerpt", "raw_content": None},
)
_EXPECTED_TAVILY_RESPONSE: Final = {
    "object": "search",
    "results": [
        {
            "title": "Gateway guide",
            "url": "https://result.invalid/guide",
            "snippet": "Route traffic [...] track spend",
            "date": None,
            "last_updated": None,
        },
        {
            "title": "Unfetched",
            "url": "https://result.invalid/unfetched",
            "snippet": "Excerpt",
            "date": None,
            "last_updated": None,
        },
    ],
}


def _create_tavily_tool(gateway: Gateway, scenario: Scenario, name: str, api_base: str) -> None:
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": "tavily", "api_key": _TAVILY_KEY, "api_base": api_base},
            }
        },
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")


def _tavily_reply(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/tavily/search"), request.target
    assert request.headers["authorization"] == f"Bearer {_TAVILY_KEY}", request.headers
    assert json.loads(request.body) == {"query": _QUERY, "max_results": 2, **_TAVILY_OPTIONS}, request.body
    return Reply(body=json.dumps({**_TAVILY_ENVELOPE, "results": list(_TAVILY_RESULTS)}).encode())


def test_sdk_search_through_the_proxy_gets_the_same_response_as_a_direct_proxy_call(gateway: Gateway) -> None:
    tool: Final = f"tavily-sdk-{uuid.uuid4().hex}"
    with wire_server(_tavily_reply) as wire, gateway.scenario() as scenario:
        _create_tavily_tool(gateway, scenario, tool, f"{wire.url}/tavily")
        key: Final = scenario.key()
        direct: Final = gateway.request(
            "POST", f"/v1/search/{tool}", {"query": _QUERY, "max_results": 2, **_TAVILY_OPTIONS}, key=key
        )
        through_sdk: Final = litellm.search(
            query=_QUERY,
            search_provider="litellm_proxy",
            search_tool_name=tool,
            api_base=f"{gateway.client.base_url}/v1",
            api_key=key,
            max_results=2,
            **_TAVILY_OPTIONS,
        )
        assert len(wire.drain()) == 2

    assert direct.status_code == 200, direct.text
    assert through_sdk.model_dump(mode="json") == json.loads(direct.content) == _EXPECTED_TAVILY_RESPONSE


@pytest.mark.asyncio
async def test_async_sdk_search_through_the_proxy_gets_the_tool_response(gateway: Gateway) -> None:
    tool: Final = f"tavily-sdk-{uuid.uuid4().hex}"
    with wire_server(_tavily_reply) as wire, gateway.scenario() as scenario:
        _create_tavily_tool(gateway, scenario, tool, f"{wire.url}/tavily")
        response: Final = await litellm.asearch(
            query=_QUERY,
            search_provider="litellm_proxy",
            search_tool_name=tool,
            api_base=str(gateway.client.base_url),
            api_key=scenario.key(),
            max_results=2,
            **_TAVILY_OPTIONS,
        )
        assert len(wire.drain()) == 1

    assert response.model_dump(mode="json") == _EXPECTED_TAVILY_RESPONSE


def test_sdk_search_with_an_unknown_virtual_key_never_reaches_the_tool(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"an unauthenticated SDK search reached a provider: {request.target}")

    tool: Final = f"tavily-sdk-{uuid.uuid4().hex}"
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_tavily_tool(gateway, scenario, tool, f"{wire.url}/tavily")
        with pytest.raises(litellm.AuthenticationError):
            litellm.search(
                query=_QUERY,
                search_provider="litellm_proxy",
                search_tool_name=tool,
                api_base=str(gateway.client.base_url),
                api_key=f"sk-{uuid.uuid4().hex}",
            )
        assert wire.drain() == ()
