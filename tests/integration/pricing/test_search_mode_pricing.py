import json
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support import interception_vendor as iv
from integration._support.client import Gateway, Scenario, eventually, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-search-key"
_QUERY: Final = "integration search pricing"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PACKAGED_PRICES: Final = _JSON_OBJECT.validate_json(
    (Path(__file__).resolve().parents[3] / "litellm" / "model_prices_and_context_window_backup.json").read_bytes()
)
_RESULTS: Final = {
    "tavily": {"results": [{"title": "Priced result", "url": "https://result.invalid/a", "content": "snippet"}]},
    "linkup": {
        "results": [{"type": "text", "name": "Priced result", "url": "https://result.invalid/a", "content": "snippet"}]
    },
}
_MODE_PARAM: Final = {"tavily": "search_depth", "linkup": "depth"}
_QUERY_FIELD: Final = {"tavily": "query", "linkup": "q"}
_MODE_ENTRY: Final = {"tavily": "tavily/search-advanced", "linkup": "linkup/search-deep"}
_MODE_VALUE: Final = {"tavily": "advanced", "linkup": "deep"}
_BASE_MODE: Final = {"tavily": "basic", "linkup": "standard"}
_NO_PARAMS: Final[Mapping[str, JsonValue]] = MappingProxyType({})


def _price(entry: str) -> float:
    prices: Final = _PACKAGED_PRICES[entry]
    assert isinstance(prices, dict), entry
    rate: Final = prices["input_cost_per_query"]
    assert isinstance(rate, int | float), (entry, rate)
    return float(rate)


def _search_tool(
    gateway: Gateway,
    scenario: Scenario,
    provider: str,
    api_base: str,
    litellm_params: Mapping[str, JsonValue] = _NO_PARAMS,
) -> str:
    name: Final = f"{provider}-priced-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {
                    "search_provider": provider,
                    "api_key": _API_KEY,
                    "api_base": api_base,
                    **litellm_params,
                },
            }
        },
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{string_value(created['search_tool_id'])}")
    return name


def _spend(gateway: Gateway, call_id: str) -> float:
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return float(str(rows[0]["spend"]))


@pytest.mark.parametrize(
    ("provider", "mode", "wire_mode", "priced_entry"),
    [
        ("tavily", "advanced", "advanced", "tavily/search-advanced"),
        ("tavily", "basic", "basic", "tavily/search"),
        ("tavily", None, None, "tavily/search"),
        ("linkup", "deep", "deep", "linkup/search-deep"),
        ("linkup", "standard", "standard", "linkup/search"),
        ("linkup", None, "standard", "linkup/search"),
    ],
    ids=["tavily-advanced", "tavily-basic", "tavily-default", "linkup-deep", "linkup-standard", "linkup-default"],
)
def test_search_is_charged_at_the_price_entry_for_the_mode_the_provider_ran(
    gateway: Gateway, provider: str, mode: str | None, wire_mode: str | None, priced_entry: str
) -> None:
    mode_param: Final = _MODE_PARAM[provider]

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", f"/{provider}/search"), request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
        assert json.loads(request.body).get(mode_param) == wire_mode, request.body
        return Reply(body=json.dumps(_RESULTS[provider]).encode())

    body: Final = {"query": _QUERY} if mode is None else {"query": _QUERY, mode_param: mode}
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        tool: Final = _search_tool(gateway, scenario, provider, f"{wire.url}/{provider}")
        spellings: Final = (
            (f"/v1/search/{tool}", body),
            (f"/search/{tool}", body),
            ("/v1/search", {"search_tool_name": tool, **body}),
            ("/search", {"model": tool, **body}),
        )
        responses: Final = tuple(gateway.request("POST", path, sent) for path, sent in spellings)
        for (path, _), response in zip(spellings, responses, strict=True):
            assert response.status_code == 200, f"{path}: {response.text}"
            assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_price(priced_entry)), (
                path,
                response.headers,
            )
        assert [request.target for request in wire.drain()] == [f"/{provider}/search"] * len(spellings)
        assert [_spend(gateway, response.headers["x-litellm-call-id"]) for response in responses] == pytest.approx(
            [_price(priced_entry)] * len(spellings)
        )


@pytest.mark.parametrize(
    ("provider", "modes"),
    [
        ("tavily", ("ultra-fast", "ADVANCED", " advanced", "", 7, ["advanced"], {"depth": "advanced"})),
        ("linkup", ("DEEP", "deeper", "", 7, ["deep"], {"depth": "deep"})),
    ],
    ids=["tavily", "linkup"],
)
def test_a_mode_with_no_price_entry_of_its_own_is_charged_at_the_base_price(
    gateway: Gateway, provider: str, modes: tuple[JsonValue, ...]
) -> None:
    mode_param: Final = _MODE_PARAM[provider]

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", f"/{provider}/search"), request.target
        return Reply(body=json.dumps(_RESULTS[provider]).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        tool: Final = _search_tool(gateway, scenario, provider, f"{wire.url}/{provider}")
        responses: Final = tuple(
            gateway.request("POST", f"/v1/search/{tool}", {"query": _QUERY, mode_param: mode}) for mode in modes
        )
        assert [response.status_code for response in responses] == [200] * len(modes), [r.text for r in responses]
        assert [json.loads(request.body)[mode_param] for request in wire.drain()] == list(modes)
        assert [float(response.headers["x-litellm-response-cost"]) for response in responses] == pytest.approx(
            [_price(f"{provider}/search")] * len(modes)
        )
        assert [_spend(gateway, response.headers["x-litellm-call-id"]) for response in responses] == pytest.approx(
            [_price(f"{provider}/search")] * len(modes)
        )


def _burst_cells(provider: str) -> tuple[tuple[str, str | None], ...]:
    return tuple((provider, mode) for mode in (_MODE_VALUE[provider], _BASE_MODE[provider], None))


def _burst_price(provider: str, mode: str | None) -> float:
    return _price(_MODE_ENTRY[provider] if mode == _MODE_VALUE[provider] else f"{provider}/search")


def test_a_concurrent_burst_with_upstream_failures_charges_each_success_at_its_mode_price_and_no_failure(
    gateway: Gateway,
) -> None:
    plan: Final = tuple(
        (provider, mode, index % 7 == 0)
        for index, (provider, mode) in enumerate((*_burst_cells("tavily"), *_burst_cells("linkup")) * 4)
    )

    def respond(request: Request) -> Reply:
        provider: Final = request.target.split("/")[1]
        if json.loads(request.body)[_QUERY_FIELD[provider]].endswith("-down"):
            return Reply(status=500, body=b'{"detail":{"error":"scripted search outage"}}')
        return Reply(body=json.dumps(_RESULTS[provider]).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        tools: Final = {
            provider: _search_tool(gateway, scenario, provider, f"{wire.url}/{provider}")
            for provider in ("tavily", "linkup")
        }

        def send(index: int) -> httpx.Response:
            provider, mode, fails = plan[index]
            query: Final = f"{_QUERY} {index}{'-down' if fails else ''}"
            body: Final = {"query": query} if mode is None else {"query": query, _MODE_PARAM[provider]: mode}
            return gateway.request("POST", f"/v1/search/{tools[provider]}", body)

        with ThreadPoolExecutor(max_workers=12) as pool:
            responses: Final = tuple(pool.map(send, range(len(plan))))
        assert [response.status_code for response in responses] == [500 if fails else 200 for *_, fails in plan], [
            response.text for response in responses
        ]
        assert len(wire.drain()) == len(plan)
        successes: Final = tuple(
            (response, _burst_price(provider, mode))
            for (provider, mode, fails), response in zip(plan, responses, strict=True)
            if not fails
        )
        assert [float(response.headers["x-litellm-response-cost"]) for response, _ in successes] == pytest.approx(
            [price for _, price in successes]
        )
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT request_id, status, spend FROM "LiteLLM_SpendLogs" WHERE model_group = ANY(%s)',
                (list(tools.values()),),
            ),
            lambda found: len(found) == len(plan),
            seconds=70,
        )
        assert {str(row["request_id"]): float(str(row["spend"])) for row in rows if row["status"] == "success"} == (
            pytest.approx({response.headers["x-litellm-call-id"]: price for response, price in successes})
        )
        assert [float(str(row["spend"])) for row in rows if row["status"] != "success"] == [0.0] * (
            len(plan) - len(successes)
        )


def _owned_config(
    tmp_path: Path, search_tools: Sequence[JsonValue], litellm_settings: Mapping[str, JsonValue] = _NO_PARAMS
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["search_tools"] = list(search_tools)
    config["litellm_settings"].update(litellm_settings)
    path: Final = tmp_path / f"search-mode-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _config_tool(name: str, provider: str, api_base: str, litellm_params: Mapping[str, JsonValue]) -> JsonValue:
    return {
        "search_tool_name": name,
        "litellm_params": {"search_provider": provider, "api_key": _API_KEY, "api_base": api_base, **litellm_params},
    }


def test_a_mode_set_on_a_config_search_tool_is_charged_at_that_modes_price_unless_the_request_overrides_it(
    gateway: Gateway, tmp_path: Path
) -> None:
    cases: Final = (("tavily", "advanced", "basic"), ("linkup", "deep", "standard"))

    def respond(request: Request) -> Reply:
        provider: Final = request.target.split("/")[1]
        assert (request.method, request.target) == ("POST", f"/{provider}/search"), request.target
        return Reply(body=json.dumps(_RESULTS[provider]).encode())

    with wire_server(respond) as wire:
        tools: Final = [
            _config_tool(f"{provider}-configured", provider, f"{wire.url}/{provider}", {_MODE_PARAM[provider]: mode})
            for provider, mode, _ in cases
        ]
        with owned_proxy(gateway, tmp_path, {}, config=_owned_config(tmp_path, tools)) as candidate:
            for provider, configured_mode, base_mode in cases:
                mode_param = _MODE_PARAM[provider]
                configured = candidate.request("POST", f"/v1/search/{provider}-configured", {"query": _QUERY})
                overridden = candidate.request(
                    "POST", f"/v1/search/{provider}-configured", {"query": _QUERY, mode_param: base_mode}
                )
                assert (configured.status_code, overridden.status_code) == (200, 200), (
                    configured.text,
                    overridden.text,
                )
                assert [json.loads(request.body)[mode_param] for request in wire.drain()] == [
                    configured_mode,
                    base_mode,
                ]
                assert (
                    float(configured.headers["x-litellm-response-cost"]),
                    float(overridden.headers["x-litellm-response-cost"]),
                    _spend(candidate, configured.headers["x-litellm-call-id"]),
                    _spend(candidate, overridden.headers["x-litellm-call-id"]),
                ) == pytest.approx((_price(_MODE_ENTRY[provider]), _price(f"{provider}/search")) * 2), provider


def test_a_price_set_on_the_search_tool_still_wins_over_the_mode_price_entry(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert json.loads(request.body)["search_depth"] == "advanced", request.body
        return Reply(body=json.dumps(_RESULTS["tavily"]).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        tool: Final = _search_tool(gateway, scenario, "tavily", f"{wire.url}/tavily", {"input_cost_per_query": 0.02})
        response: Final = gateway.request("POST", f"/v1/search/{tool}", {"query": _QUERY, "search_depth": "advanced"})
        assert response.status_code == 200, response.text
        assert [request.target for request in wire.drain()] == ["/tavily/search"]
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.02), response.headers
        assert _spend(gateway, response.headers["x-litellm-call-id"]) == pytest.approx(0.02)


def _anthropic_reply(identity: str, content: Sequence[JsonValue], stop_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        ).encode()
    )


def test_a_web_search_intercepted_on_every_llm_endpoint_is_charged_at_the_search_tools_mode_price(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"search-mode-{uuid.uuid4().hex}"
    tool: Final = f"tavily-intercepted-{uuid.uuid4().hex}"
    chat_deployment: Final = iv.deployment("search")
    responses_deployment: Final = iv.deployment("search")

    def respond(request: Request) -> Reply:
        if request.target == "/tavily/search":
            assert json.loads(request.body)["search_depth"] == "advanced", request.body
            return Reply(body=json.dumps(_RESULTS["tavily"]).encode())
        if request.target != "/v1/messages":
            return iv.respond(request)
        body: Final = json.loads(request.body)
        if any(entry.get("type") == "web_search_20250305" for entry in body["tools"]):
            return _anthropic_reply(identity, [{"type": "text", "text": "not intercepted"}], "end_turn")
        answered: Final = any(
            isinstance(message["content"], list) and any(part["type"] == "tool_result" for part in message["content"])
            for message in body["messages"]
        )
        if answered:
            return _anthropic_reply(identity, [{"type": "text", "text": "answered"}], "end_turn")
        tool_use: Final[JsonValue] = {
            "type": "tool_use",
            "id": identity,
            "name": iv.SEARCH_TOOL,
            "input": {"query": _QUERY},
        }
        return _anthropic_reply(identity, [tool_use], "tool_use")

    with wire_server(respond) as wire:
        path: Final = _owned_config(
            tmp_path,
            [_config_tool(tool, "tavily", f"{wire.url}/tavily", {"search_depth": "advanced"})],
            {
                "callbacks": ["websearch_interception"],
                "websearch_interception_params": {
                    "enabled": True,
                    "enabled_providers": ["anthropic", "azure"],
                    "search_tool_name": tool,
                },
            },
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as models:
            azure: Final = {"api_base": wire.url, "api_key": iv.AZURE_KEY, "api_version": iv.API_VERSION}
            sends: Final = (
                (
                    "/v1/messages",
                    {
                        "model": models.model(
                            model="anthropic/claude-sonnet-4-5-20250929",
                            api_base=wire.url,
                            api_key="synthetic-anthropic-key",
                        ),
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": identity}],
                        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
                    },
                    "answered",
                ),
                (
                    "/v1/chat/completions",
                    {
                        "model": models.model(model=f"azure/{chat_deployment}", **azure),
                        "messages": [{"role": "user", "content": f"Search for {chat_deployment}"}],
                        "tools": [{"type": "web_search"}],
                    },
                    iv.answer(chat_deployment),
                ),
                (
                    "/v1/responses",
                    {
                        "model": models.model(model=f"azure/{responses_deployment}", **azure),
                        "input": f"Search for {responses_deployment}",
                        "tools": [{"type": "web_search"}],
                    },
                    iv.answer(responses_deployment),
                ),
            )
            responses: Final = tuple(
                eventually(
                    lambda route=route, body=body: candidate.request("POST", route, body),
                    lambda response, answer=answer: answer in response.text,
                    seconds=40,
                )
                for route, body, answer in sends
            )
            assert [response.status_code for response in responses] == [200, 200, 200], [r.text for r in responses]
            searches: Final = tuple(
                json.loads(request.body)["query"] for request in wire.drain() if request.target == "/tavily/search"
            )
            assert sorted(searches) == sorted((_QUERY, iv.query(chat_deployment), iv.query(responses_deployment))), (
                searches
            )
            rows: Final = eventually(
                lambda: candidate.rows(
                    """SELECT spend FROM "LiteLLM_SpendLogs" WHERE call_type IN ('search', 'asearch')""", ()
                ),
                lambda found: len(found) == len(searches),
                seconds=70,
            )
        assert [float(str(row["spend"])) for row in rows] == pytest.approx([_price("tavily/search-advanced")] * 3), rows
