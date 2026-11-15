import json
import uuid
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


def _chat_completion(tag: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-go-{tag}",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.3",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": dict(_CHAT_USAGE),
        }
    ).encode()


def _chat_stream(tag: str) -> tuple[bytes, ...]:
    text: Final = {
        "id": f"chatcmpl-go-{tag}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm-5.3",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
    }
    finish: Final = {**text, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    final_usage: Final = {**text, "choices": [], "usage": dict(_CHAT_USAGE)}
    return (*(_data_frame(event) for event in (text, finish, final_usage)), b"data: [DONE]\n\n")


def _responses_response(tag: str) -> Mapping[str, JsonValue]:
    return {
        "id": f"resp_go_{tag}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-6-luna",
        "output": [
            {
                "id": f"msg_go_{tag}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": dict(_RESPONSES_USAGE),
    }


def _responses_stream(tag: str) -> tuple[bytes, ...]:
    response: Final = _responses_response(tag)
    return (
        _event_frame(
            {"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress"}}
        ),
        _event_frame(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": f"msg_go_{tag}",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok",
            }
        ),
        _event_frame({"type": "response.completed", "sequence_number": 2, "response": response}),
    )


def _anthropic_message(tag: str) -> Mapping[str, JsonValue]:
    return {
        "id": f"msg_go_{tag}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": dict(_MESSAGES_USAGE),
    }


def _anthropic_stream(tag: str) -> tuple[bytes, ...]:
    start: Final = {**_anthropic_message(tag), "content": [], "stop_reason": None}
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
    tag: Final = uuid.uuid4().hex
    if ":streamGenerateContent" in request.target:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "contents" in body and "messages" not in body, body
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _data_frame(
                    {
                        "candidates": [
                            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 1200,
                            "candidatesTokenCount": 300,
                            "totalTokenCount": 1500,
                        },
                    }
                ),
            ),
        )
    if request.target.endswith("/chat/completions"):
        return (
            Reply(content_type="text/event-stream", chunks=_chat_stream(tag))
            if streaming
            else Reply(body=_chat_completion(tag))
        )
    if request.target.endswith("/responses"):
        if streaming:
            return Reply(content_type="text/event-stream", chunks=_responses_stream(tag))
        return Reply(body=json.dumps(_responses_response(tag)).encode())
    assert request.target.endswith("/v1/messages"), request.target
    messages_body: Final = _JSON_OBJECT.validate_json(request.body)
    assert "messages" in messages_body and "contents" not in messages_body, messages_body
    if streaming:
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream(tag))
    return Reply(body=json.dumps(_anthropic_message(tag)).encode())


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


@pytest.mark.parametrize(
    ("model", "stream"),
    (
        ("opencode_go/minimax-m3", True),
        ("opencode_go/minimax-m3", False),
        ("opencode_go/minimax-m3.1-flash", True),
        ("opencode_go/gemini-4-argon", True),
    ),
    ids=("messages-stream", "messages-non-stream", "openai-stream", "gemini-stream"),
)
def test_chat_completion_uses_go_model_stream_protocol(gateway: Gateway, model: str, stream: bool) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(model=model, api_base=wire.url, api_key="synthetic-go-key")
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _body("/v1/chat/completions", deployment, stream)
        )
        assert response.status_code == 200, response.text
        if not stream:
            body: Final = _JSON_OBJECT.validate_json(response.content)
            nonstream_choices: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(body["choices"])
            message: Final = _JSON_OBJECT.validate_python(nonstream_choices[0]["message"])
            assert message["content"] == "ok", response.text
            assert nonstream_choices[0]["finish_reason"] == "stop", response.text
            return
        events: Final = tuple(
            _JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        )
        choice_groups: Final = tuple(
            TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(event["choices"]) for event in events
        )
        choices: Final = tuple(group[0] for group in choice_groups if group)
        deltas: Final = tuple(_JSON_OBJECT.validate_python(choice["delta"]) for choice in choices)
        assert "".join(str(delta.get("content") or "") for delta in deltas) == "ok", response.text
        assert tuple(choice["finish_reason"] for choice in choices if choice.get("finish_reason")) == ("stop",), (
            response.text
        )
        assert response.text.rstrip().endswith("data: [DONE]"), response.text
        if model == "opencode_go/minimax-m3":
            assert _logged_spend(response.headers["x-litellm-call-id"]) == _approx(_LIST_COST)
