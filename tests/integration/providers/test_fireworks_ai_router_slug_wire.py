import json
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_ROUTER_SLUG: Final = "routers/glm-latest"
_ROUTER_RESOURCE: Final = "accounts/fireworks/routers/glm-latest"
_FIREROUTER_SLUGS: Final = ("firerouter", "firerouter/kimi-k3/deepseek-v4")
_API_KEY: Final = "synthetic-fireworks-key"
_PROMPT: Final = "route me through the router"
_COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, object]])


def _positive_rate(entry: dict[str, object], field: str) -> bool:
    value: Final = entry.get(field)
    return isinstance(value, (int, float)) and value > 0


def _pick_routed_model() -> str:
    catalog: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())
    return next(
        key
        for key, entry in catalog.items()
        if "/" not in key
        and entry.get("litellm_provider") == "anthropic"
        and _positive_rate(entry, "input_cost_per_token")
        and _positive_rate(entry, "output_cost_per_token")
        and f"fireworks_ai/{key}" not in catalog
    )


def _catalog_cost(model: str, field: str) -> float:
    cost_value: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())[model][field]
    assert isinstance(cost_value, (int, float))
    return float(cost_value)


_ROUTED_MODEL: Final = _pick_routed_model()


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


def _chat_completion(identity: str, model: str, prompt_tokens: int, completion_tokens: int) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "routed"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    ).encode()


def _provider_body(request: Request, target: str) -> dict[str, JsonValue]:
    assert request.method == "POST"
    assert request.target == target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    return _JSON_OBJECT.validate_json(request.body)


@pytest.mark.covers("other.provider_wire.fireworks_ai.router_slug_chat_sends_router_resource_name")
def test_fireworks_router_slug_chat_sends_router_resource_not_models_path(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = _provider_body(request, "/chat/completions")
        assert body["model"] == _ROUTER_RESOURCE, body
        assert body["messages"] == [{"role": "user", "content": _PROMPT}]
        return Reply(
            body=json.dumps(
                {
                    "id": "fw-router-chat",
                    "object": "chat.completion",
                    "created": 1,
                    "model": _ROUTER_RESOURCE,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "routed"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fireworks_ai/{_ROUTER_SLUG}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "routed"}}
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


@pytest.mark.covers("other.provider_wire.fireworks_ai.router_slug_text_completion_sends_router_resource_name")
def test_fireworks_router_slug_text_completion_sends_router_resource_not_models_path(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        body: Final = _provider_body(request, "/completions")
        assert body["model"] == _ROUTER_RESOURCE, body
        assert body["prompt"] == _PROMPT
        return Reply(
            body=json.dumps(
                {
                    "id": "fw-router-text",
                    "object": "text_completion",
                    "created": 1,
                    "model": _ROUTER_RESOURCE,
                    "choices": [{"index": 0, "text": "routed", "finish_reason": "stop", "logprobs": None}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fireworks_ai/{_ROUTER_SLUG}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/completions", {"model": model, "prompt": _PROMPT})
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"] == [{"index": 0, "text": "routed", "finish_reason": "stop", "logprobs": None}]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]


@pytest.mark.parametrize("slug", _FIREROUTER_SLUGS)
def test_fireworks_firerouter_short_name_sends_router_resource_not_models_path(gateway: Gateway, slug: str) -> None:
    resource: Final = f"accounts/fireworks/routers/{slug}"

    def respond(request: Request) -> Reply:
        body: Final = _provider_body(request, "/chat/completions")
        assert body["model"] == resource, body
        return Reply(body=_chat_completion(f"fw-{slug}", resource, 5, 1))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fireworks_ai/{slug}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"] == [
            {"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "routed"}}
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


def test_fireworks_firerouter_claude_leg_is_charged_at_the_routed_models_own_rate(gateway: Gateway) -> None:
    identity: Final = f"fw-firerouter-claude-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _provider_body(request, "/chat/completions")
        assert body["model"] == "accounts/fireworks/routers/firerouter", body
        assert request.headers["x-anthropic-api-key"] == "synthetic-anthropic-key"
        return Reply(body=_chat_completion(identity, _ROUTED_MODEL, 23, 41))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="fireworks_ai/firerouter",
            api_base=wire.url,
            api_key=_API_KEY,
            extra_headers={"x-anthropic-api-key": "synthetic-anthropic-key"},
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}]},
        )
        assert response.status_code == 200, response.text
        expected_cost: Final = 23 * _catalog_cost(_ROUTED_MODEL, "input_cost_per_token") + 41 * _catalog_cost(
            _ROUTED_MODEL, "output_cost_per_token"
        )
        assert expected_cost > 0
        assert float(response.headers["x-litellm-response-cost"]) == _approx(expected_cost)
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float, str))
    assert float(spend) == _approx(expected_cost)
