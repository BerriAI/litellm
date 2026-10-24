import json
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_LIST_COST: Final = 0.00412
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

# OpenCode's documented usage block (opencode.ai/v2/docs/console/api/inference, "Cost", read 2026-10-24):
# usage.cost is what hit the prepaid balance (0 inside the Go plan), usage.opencode.list_cost is the model price
_GO_BLOCK: Final[Mapping[str, JsonValue]] = {
    "cost": 0,
    "opencode": {"billing_source": "go", "list_cost": _LIST_COST, "allowance": _LIST_COST},
}
_CHAT_USAGE: Final[Mapping[str, JsonValue]] = {
    "prompt_tokens": 1200,
    "completion_tokens": 300,
    "total_tokens": 1500,
    **_GO_BLOCK,
}
_RESPONSES_USAGE: Final[Mapping[str, JsonValue]] = {
    "input_tokens": 1200,
    "output_tokens": 300,
    "total_tokens": 1500,
    **_GO_BLOCK,
}
_MESSAGES_USAGE: Final[Mapping[str, JsonValue]] = {"input_tokens": 1200, "output_tokens": 300, **_GO_BLOCK}


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


def _data_frame(event: Mapping[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


def _event_frame(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _chat_completion() -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-go",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.3",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": dict(_CHAT_USAGE),
        }
    ).encode()


def _chat_stream() -> tuple[bytes, ...]:
    text: Final = {
        "id": "chatcmpl-go",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm-5.3",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
    }
    finish: Final = {**text, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    final_usage: Final = {**text, "choices": [], "usage": dict(_CHAT_USAGE)}
    return (*(_data_frame(event) for event in (text, finish, final_usage)), b"data: [DONE]\n\n")


def _responses_response() -> Mapping[str, JsonValue]:
    return {
        "id": "resp_go",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "msg_go",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": dict(_RESPONSES_USAGE),
    }


def _responses_stream() -> tuple[bytes, ...]:
    response: Final = _responses_response()
    return (
        _event_frame(
            {"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress"}}
        ),
        _event_frame(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_go",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok",
            }
        ),
        _event_frame({"type": "response.completed", "sequence_number": 2, "response": response}),
    )


def _anthropic_message() -> Mapping[str, JsonValue]:
    return {
        "id": "msg_go",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": dict(_MESSAGES_USAGE),
    }


def _anthropic_stream() -> tuple[bytes, ...]:
    start: Final = {**_anthropic_message(), "content": [], "stop_reason": None}
    return (
        _event_frame(
            {"type": "message_start", "message": {**start, "usage": {"input_tokens": 1200, "output_tokens": 0}}}
        ),
        _event_frame({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _event_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        _event_frame({"type": "content_block_stop", "index": 0}),
        _event_frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": dict(_MESSAGES_USAGE),
            }
        ),
        _event_frame({"type": "message_stop"}),
    )


def _respond(request: Request) -> Reply:
    assert request.method == "POST", request.method
    streaming: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
    if request.target.endswith("/chat/completions"):
        return (
            Reply(content_type="text/event-stream", chunks=_chat_stream())
            if streaming
            else Reply(body=_chat_completion())
        )
    if request.target.endswith("/responses"):
        if streaming:
            return Reply(content_type="text/event-stream", chunks=_responses_stream())
        return Reply(body=json.dumps(_responses_response()).encode())
    assert request.target.endswith("/v1/messages"), request.target
    if streaming:
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream())
    return Reply(body=json.dumps(_anthropic_message()).encode())


def _logged_spend(call_id: str) -> float:
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return float(rows[0]["spend"])


def _body(path: str, model: str, stream: bool) -> Mapping[str, JsonValue]:
    if path == "/v1/responses":
        return {"model": model, "input": "hi", "stream": stream}
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "stream": stream}


@pytest.mark.parametrize("stream", (False, True), ids=("non-stream", "stream"))
@pytest.mark.parametrize(
    ("path", "model"),
    (
        ("/v1/chat/completions", "opencode_go/glm-5.3"),
        ("/v1/responses", "opencode_go/gpt-6-luna"),
        ("/v1/messages", "opencode_go/claude-haiku-5-5"),
    ),
    ids=("chat", "responses", "messages"),
)
def test_request_is_billed_at_the_reported_list_cost(gateway: Gateway, path: str, model: str, stream: bool) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(model=model, api_base=wire.url, api_key="synthetic-go-key")
        response: Final = gateway.request("POST", path, _body(path, deployment, stream))
        assert response.status_code == 200, response.text
        assert "ok" in response.text
        if not stream:
            assert float(response.headers["x-litellm-response-cost"]) == _approx(_LIST_COST)
        assert _logged_spend(response.headers["x-litellm-call-id"]) == _approx(_LIST_COST)
