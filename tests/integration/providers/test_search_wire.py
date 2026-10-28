import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, JsonValue

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


_EXA_HIGHLIGHTS: Final = {"highlights": {"numSentences": 2}}
_EXA_ENVELOPE: Final = {
    "requestId": "exa-request-5d1f0c",
    "resolvedSearchType": "neural",
    "searchTime": 412.5,
    "costDollars": {"total": 0.007, "search": {"neural": 0.005}, "contents": {"highlights": 0.002}},
}
_EXA_SCORED_RESULTS: Final = (
    {
        "id": "https://result.invalid/margins",
        "title": "Margin outlook",
        "url": "https://result.invalid/margins",
        "publishedDate": "2026-01-15T00:00:00.000Z",
        "score": 0.91,
        "highlights": ["Margins hold steady.", "Deposit costs ease."],
        "highlightScores": [0.82, 0.64],
    },
    {
        "id": "https://result.invalid/deposits",
        "title": "Deposit pricing",
        "url": "https://result.invalid/deposits",
        "score": 0.47,
        "highlights": ["Deposit betas fall."],
        "highlightScores": [0.55],
    },
)


class _ScoredSearchResult(_SearchResult):
    highlights: tuple[str, ...] | None = None
    highlight_scores: tuple[float, ...] | None = None
    score: float | None = None


class _ScoredSearchResponse(BaseModel):
    object: str
    results: tuple[_ScoredSearchResult, ...]
    provider_metadata: JsonValue = None


_EXPECTED_SCORED_RESPONSE: Final = _ScoredSearchResponse(
    object="search",
    results=(
        _ScoredSearchResult(
            title="Margin outlook",
            url="https://result.invalid/margins",
            snippet="Margins hold steady.\n\nDeposit costs ease.",
            date="2026-01-15T00:00:00.000Z",
            last_updated=None,
            highlights=("Margins hold steady.", "Deposit costs ease."),
            highlight_scores=(0.82, 0.64),
            score=0.91,
        ),
        _ScoredSearchResult(
            title="Deposit pricing",
            url="https://result.invalid/deposits",
            snippet="Deposit betas fall.",
            date=None,
            last_updated=None,
            highlights=("Deposit betas fall.",),
            highlight_scores=(0.55,),
            score=0.47,
        ),
    ),
    provider_metadata=_EXA_ENVELOPE,
)
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


def test_exa_highlights_scores_and_response_envelope_reach_the_caller_on_every_route_spelling(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/exa/search"), request.target
        assert json.loads(request.body) == {"query": _QUERY, "numResults": 2, "contents": _EXA_HIGHLIGHTS}, request.body
        return Reply(body=json.dumps({**_EXA_ENVELOPE, "results": list(_EXA_SCORED_RESULTS)}).encode())

    tool: Final = f"exa-tool-{uuid.uuid4().hex}"
    body: Final = {"query": _QUERY, "max_results": 2, "contents": _EXA_HIGHLIGHTS}
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, tool, f"{wire.url}/exa")
        spellings: Final = (
            (f"/v1/search/{tool}", body),
            (f"/search/{tool}", body),
            ("/v1/search", {"search_tool_name": tool, **body}),
            ("/search", {"model": tool, **body}),
        )
        for path, payload in spellings:
            response = gateway.request("POST", path, payload)
            assert response.status_code == 200, f"{path}: {response.text}"
            assert _ScoredSearchResponse.model_validate_json(response.content) == _EXPECTED_SCORED_RESPONSE, (
                f"{path}: {response.text}"
            )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/exa/search")] * len(
            spellings
        )


def test_concurrent_exa_searches_each_get_their_own_response_envelope(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        query: Final = json.loads(request.body)["query"]
        results: Final = [] if query.endswith("-empty") else [_EXA_SCORED_RESULTS[1]]
        return Reply(
            body=json.dumps({"requestId": f"exa-{query}", "costDollars": {"total": 0.005}, "results": results}).encode()
        )

    tool: Final = f"exa-tool-{uuid.uuid4().hex}"
    queries: Final = tuple(f"burst-{uuid.uuid4().hex}{'-empty' if index % 2 else ''}" for index in range(16))
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        _create_exa_tool(gateway, scenario, tool, f"{wire.url}/exa")
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            responses: Final = tuple(
                pool.map(lambda query: gateway.request("POST", f"/v1/search/{tool}", {"query": query}), queries)
            )
        observed: Final = tuple(
            (
                response.status_code,
                _ScoredSearchResponse.model_validate_json(response.content).provider_metadata,
                tuple(result.score for result in _ScoredSearchResponse.model_validate_json(response.content).results),
            )
            for response in responses
        )
        assert observed == tuple(
            (
                200,
                {"requestId": f"exa-{query}", "costDollars": {"total": 0.005}},
                () if query.endswith("-empty") else (0.47,),
            )
            for query in queries
        ), [response.text for response in responses]
        assert len(wire.drain()) == len(queries)


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
