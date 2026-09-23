import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_ROUTER_SLUG: Final = "routers/glm-latest"
_ROUTER_RESOURCE: Final = "accounts/fireworks/routers/glm-latest"
_API_KEY: Final = "synthetic-fireworks-key"
_PROMPT: Final = "route me through the router"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


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
