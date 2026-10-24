import json
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "opencode_go/glm-5.3"
_LIST_COST: Final = 0.00412
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

# OpenCode's documented usage block (opencode.ai/v2/docs/console/api/inference, "Cost", read 2026-10-24):
# usage.cost is what hit the prepaid balance (0 inside the Go plan), usage.opencode.list_cost is the model price
_GO_USAGE: Final[Mapping[str, JsonValue]] = {
    "prompt_tokens": 1200,
    "completion_tokens": 300,
    "total_tokens": 1500,
    "cost": 0,
    "opencode": {"billing_source": "go", "list_cost": _LIST_COST, "allowance": _LIST_COST},
}


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


def _completion(usage: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-go",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.3",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": dict(usage),
        }
    ).encode()


def _stream(usage: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    text: Final = {
        "id": "chatcmpl-go",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm-5.3",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
    }
    finish: Final = {**text, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    final_usage: Final = {**text, "choices": [], "usage": dict(usage)}
    return (
        *(f"data: {json.dumps(event)}\n\n".encode() for event in (text, finish, final_usage)),
        b"data: [DONE]\n\n",
    )


def _respond(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target.endswith("/chat/completions"), request.target
    if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_stream(_GO_USAGE))
    return Reply(body=_completion(_GO_USAGE))


def _spend(column: str, value: str) -> float:
    rows: Final = eventually(
        lambda: read_rows(f'SELECT spend FROM "LiteLLM_SpendLogs" WHERE {column} = %s', (value,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return float(rows[0]["spend"])


def test_chat_completion_is_billed_at_the_reported_list_cost(gateway: Gateway) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key="synthetic-go-key")
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == _approx(_LIST_COST)
        request_id: Final = string_value(object_value(response.json())["id"])
        assert _spend("request_id", request_id) == _approx(_LIST_COST)


def test_streamed_chat_completion_is_billed_at_the_reported_list_cost(gateway: Gateway) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key="synthetic-go-key")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert response.status_code == 200, response.text
        assert "data: [DONE]" in response.text
        assert _spend("litellm_call_id", response.headers["x-litellm-call-id"]) == _approx(_LIST_COST)
