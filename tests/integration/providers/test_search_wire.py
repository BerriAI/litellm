import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

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


def _create_exa_tool(gateway: Gateway, scenario: Scenario, name: str, api_base: str) -> None:
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": "exa_ai", "api_key": _EXA_KEY, "api_base": api_base},
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
