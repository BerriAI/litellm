import json
import threading
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
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
_MODE_ENTRY: Final = {"tavily": "tavily/search-advanced", "linkup": "linkup/search-deep"}


def _price(entry: str) -> float:
    prices: Final = _PACKAGED_PRICES[entry]
    assert isinstance(prices, dict), entry
    rate: Final = prices["input_cost_per_query"]
    assert isinstance(rate, float) and rate > 0, (entry, rate)
    return rate


def _search_tool(
    gateway: Gateway, scenario: Scenario, provider: str, api_base: str, **litellm_params: JsonValue
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
    assert _price(_MODE_ENTRY[provider]) > _price(f"{provider}/search")
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


def _owned_config(
    tmp_path: Path, search_tools: list[JsonValue], litellm_settings: dict[str, JsonValue] | None = None
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["search_tools"] = search_tools
    config["litellm_settings"].update(litellm_settings or {})
    path: Final = tmp_path / f"search-mode-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _config_tool(name: str, provider: str, api_base: str, **litellm_params: JsonValue) -> JsonValue:
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
            _config_tool(f"{provider}-configured", provider, f"{wire.url}/{provider}", **{_MODE_PARAM[provider]: mode})
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
        tool: Final = _search_tool(gateway, scenario, "tavily", f"{wire.url}/tavily", input_cost_per_query=0.02)
        response: Final = gateway.request("POST", f"/v1/search/{tool}", {"query": _QUERY, "search_depth": "advanced"})
        assert response.status_code == 200, response.text
        assert [request.target for request in wire.drain()] == ["/tavily/search"]
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.02), response.headers
        assert _spend(gateway, response.headers["x-litellm-call-id"]) == pytest.approx(0.02)


def _anthropic_reply(identity: str, content: list[JsonValue], stop_reason: str) -> Reply:
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


def test_a_web_search_intercepted_on_messages_is_charged_at_the_search_tools_mode_price(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"search-mode-{uuid.uuid4().hex}"
    tool: Final = f"tavily-intercepted-{uuid.uuid4().hex}"
    searched: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if request.target == "/tavily/search":
            assert json.loads(request.body) == {"query": _QUERY, "search_depth": "advanced"}, request.body
            searched.set()
            return Reply(body=json.dumps(_RESULTS["tavily"]).encode())
        assert (request.method, request.target) == ("POST", "/v1/messages"), request.target
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
            "name": "litellm_web_search",
            "input": {"query": _QUERY},
        }
        return _anthropic_reply(identity, [tool_use], "tool_use")

    def send(candidate: Gateway, model: str) -> httpx.Response:
        return candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": f"{identity} {uuid.uuid4().hex}"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            },
        )

    with wire_server(respond) as wire:
        path: Final = _owned_config(
            tmp_path,
            [_config_tool(tool, "tavily", f"{wire.url}/tavily", search_depth="advanced")],
            {
                "callbacks": ["websearch_interception"],
                "websearch_interception_params": {
                    "enabled": True,
                    "enabled_providers": ["anthropic"],
                    "search_tool_name": tool,
                },
            },
        )
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as models:
            model: Final = models.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            response: Final = eventually(lambda: send(candidate, model), lambda _: searched.is_set(), seconds=40)
            assert response.status_code == 200, response.text
            rows: Final = eventually(
                lambda: candidate.rows(
                    """SELECT spend FROM "LiteLLM_SpendLogs" WHERE call_type IN ('search', 'asearch')""", ()
                ),
                lambda found: len(found) == 1,
                seconds=70,
            )
        assert float(str(rows[0]["spend"])) == pytest.approx(_price("tavily/search-advanced")), rows
