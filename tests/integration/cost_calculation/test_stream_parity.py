"""A streamed request bills exactly what the same request bills without streaming.

Streaming reconstructs the response, its usage and its model name from chunks after the fact, in a separate code
path from the non-streamed one (LIT-6241, LIT-6872, LIT-7632, LIT-7729, LIT-9044, LIT-9065). For each client route
and backend the two requests go through the same deployment and key, and the two LiteLLM_SpendLogs rows must agree
on model, model_group, tokens, key hash and spend, with spend equal to the provider usage at the deployment rates
"""

import json
from collections.abc import Callable
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PROMPT_TOKENS: Final = 30
COMPLETION_TOKENS: Final = 40
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
EXPECTED_SPEND: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE
TEXT: Final = "hi"


def _data(payload: dict[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _event(name: str, payload: dict[str, JsonValue]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


def _openai_chat(body: dict[str, JsonValue]) -> Reply:
    identity: Final = f"chatcmpl-{uuid4().hex[:12]}"
    model: Final = str(body["model"])
    usage: Final = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    if body.get("stream") is not True:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": TEXT}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": model}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _data({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": TEXT}}]}),
            _data({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            _data({**chunk, "choices": [], "usage": usage}) + b"data: [DONE]\n\n",
        ),
    )


def _openai_responses(body: dict[str, JsonValue]) -> Reply:
    identity: Final = f"resp_{uuid4().hex[:12]}"
    item: Final = {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": TEXT, "annotations": []}],
    }
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": str(body["model"]),
        "output": [item],
        "usage": {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
    }
    if body.get("stream") is not True:
        return Reply(body=json.dumps(response).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _event(
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**response, "status": "in_progress", "output": []},
                },
            ),
            _event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {**item, "status": "in_progress", "content": []},
                },
            ),
            _event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": TEXT,
                },
            ),
            _event(
                "response.output_item.done",
                {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": item},
            ),
            _event("response.completed", {"type": "response.completed", "sequence_number": 4, "response": response}),
        ),
    )


def _anthropic_messages(body: dict[str, JsonValue]) -> Reply:
    identity: Final = f"msg_{uuid4().hex[:12]}"
    model: Final = str(body["model"])
    if body.get("stream") is not True:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": TEXT}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
                }
            ).encode()
        )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": identity,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": 1},
                    },
                },
            ),
            _event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": TEXT}},
            ),
            _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": COMPLETION_TOKENS},
                },
            ),
            _event("message_stop", {"type": "message_stop"}),
        ),
    )


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    body: Final = json.loads(request.body)
    if request.target.endswith("/chat/completions"):
        return _openai_chat(body)
    if request.target.endswith("/responses"):
        return _openai_responses(body)
    if request.target.endswith("/messages"):
        return _anthropic_messages(body)
    raise AssertionError(request.target)


def _chat_body(model: str, content: str, stream: bool) -> dict[str, JsonValue]:
    extra: Final[dict[str, JsonValue]] = {"stream": True, "stream_options": {"include_usage": True}} if stream else {}
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


def _messages_body(model: str, content: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": content}], "stream": stream}


def _responses_body(model: str, content: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "input": content, "stream": stream}


Backend = Callable[[str], dict[str, JsonValue]]
Body = Callable[[str, str, bool], dict[str, JsonValue]]

_OPENAI: Final[Backend] = lambda url: {
    "model": "openai/gpt-6.1-sol",
    "api_key": "integration-provider-key",
    "api_base": f"{url}/v1",
}  # noqa: E731
_OPENAI_RESPONSES: Final[Backend] = lambda url: {
    "model": "openai/responses/gpt-6.1-sol",
    "api_key": "integration-provider-key",
    "api_base": f"{url}/v1",
}  # noqa: E731
_ANTHROPIC: Final[Backend] = lambda url: {
    "model": "anthropic/claude-opus-4-8",
    "api_key": "integration-provider-key",
    "api_base": url,
}  # noqa: E731

_CELLS: Final = (
    pytest.param("/v1/chat/completions", _chat_body, _OPENAI, id="chat-openai"),
    pytest.param("/v1/chat/completions", _chat_body, _ANTHROPIC, id="chat-anthropic"),
    pytest.param("/v1/chat/completions", _chat_body, _OPENAI_RESPONSES, id="chat-responses-bridge"),
    pytest.param("/v1/messages", _messages_body, _ANTHROPIC, id="messages-anthropic"),
    pytest.param("/v1/messages", _messages_body, _OPENAI, id="messages-chat-bridge"),
    pytest.param("/v1/responses", _responses_body, _OPENAI_RESPONSES, id="responses-openai"),
    pytest.param("/v1/responses", _responses_body, _ANTHROPIC, id="responses-anthropic-bridge"),
)

_COLUMNS: Final = (
    "model",
    "model_group",
    "custom_llm_provider",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "spend",
    "api_key",
)


def _rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT litellm_call_id, model, model_group, custom_llm_provider, prompt_tokens, completion_tokens, total_tokens,"
        ' spend, api_key FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
        (sha256(key.encode()).hexdigest(),),
    )


def _call(gateway: Gateway, route: str, body: dict[str, JsonValue], key: str) -> str:
    response: Final = gateway.request("POST", route, body, key=key)
    assert response.status_code == 200, response.text
    assert TEXT in response.text, response.text
    return response.headers["x-litellm-call-id"]


@pytest.mark.parametrize(("route", "body", "backend"), _CELLS)
@pytest.mark.timeout(180)
def test_streamed_request_bills_the_same_row_as_the_non_streamed_request(
    gateway: Gateway, route: str, body: Body, backend: Backend
) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            **backend(wire.url), input_cost_per_token=INPUT_RATE, output_cost_per_token=OUTPUT_RATE
        )
        key: Final = scenario.key(models=[model])
        content: Final = f"stream parity {uuid4().hex}"
        plain_call: Final = _call(gateway, route, body(model, content, False), key)
        streamed_call: Final = _call(gateway, route, body(model, content, True), key)

        rows: Final = eventually(
            lambda: _rows(key),
            lambda found: {str(row["litellm_call_id"]) for row in found} >= {plain_call, streamed_call},
            seconds=70,
        )
        plain_rows: Final = [row for row in rows if row["litellm_call_id"] == plain_call]
        streamed_rows: Final = [row for row in rows if row["litellm_call_id"] == streamed_call]
        assert len(plain_rows) == 1 and len(streamed_rows) == 1, rows

        plain: Final = {column: plain_rows[0][column] for column in _COLUMNS}
        streamed: Final = {column: streamed_rows[0][column] for column in _COLUMNS}
        assert streamed == plain, (route, plain, streamed)
        assert plain["api_key"] == sha256(key.encode()).hexdigest(), plain
        assert plain["model_group"] == model, plain
        assert (plain["prompt_tokens"], plain["completion_tokens"]) == (PROMPT_TOKENS, COMPLETION_TOKENS), plain
        assert float(str(plain["spend"])) == pytest.approx(EXPECTED_SPEND), plain
