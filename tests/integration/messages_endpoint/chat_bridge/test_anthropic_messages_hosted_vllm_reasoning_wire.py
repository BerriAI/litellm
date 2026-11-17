import json
import uuid
from typing import Final

import anthropic
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "glm-reasoning"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_TOOL_USE_ID: Final = "toolu_weather_1"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_TOOLS: Final[list[dict[str, JsonValue]]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def _completion(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "It is raining."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        }
    ).encode()


def _streamed_completion(identity: str) -> Reply:
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": _BACKEND}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "It is raining."}}]},
        {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
        },
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
    )


def _tool_loop(thinking: str, marker: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"What is the weather in Paris? {marker}"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": thinking, "signature": "opaque-signature"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": _TOOL_USE_ID, "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": _TOOL_USE_ID, "content": "light rain, 14C"}],
        },
    ]


def _expected_upstream(thinking: str, marker: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"What is the weather in Paris? {marker}"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "reasoning_content": thinking,
            "tool_calls": [
                {
                    "id": _TOOL_USE_ID,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": _TOOL_USE_ID, "content": "light rain, 14C"},
    ]


def _only_body(wire: Wire) -> dict[str, JsonValue]:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/v1/chat/completions")]
    return _JSON_OBJECT.validate_json(received[0].body)


def _sent_messages(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return _MESSAGES.validate_python(body["messages"])


def _spend_status(identity: str) -> JsonValue:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]["status"]


def _post_messages(gateway: Gateway, model: str, messages: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    response: Final = gateway.request(
        "POST",
        "/v1/messages",
        {"model": model, "max_tokens": 256, "messages": messages, "cache": {"no-cache": True}},
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    return _JSON_OBJECT.validate_json(response.content)


def test_anthropic_sdk_thinking_block_reaches_hosted_vllm_as_reasoning_content(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"chatcmpl-messages-{marker}"
    thinking: Final = f"The user wants Paris weather, codeword mango{marker[:4]}."
    with wire_server(lambda _: Reply(body=_completion(identity))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model,
            max_tokens=256,
            tools=_TOOLS,  # pyright: ignore[reportArgumentType]  # plain JSON tool definitions
            messages=_tool_loop(thinking, marker),  # pyright: ignore[reportArgumentType]  # plain JSON content blocks
        )
        assert message.id == identity
        assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", "It is raining.")]
        body: Final = _only_body(wire)
        assert _sent_messages(body) == _expected_upstream(thinking, marker)
        assert "thinking_blocks" not in json.dumps(body) and "opaque-signature" not in json.dumps(body), body
        assert _spend_status(identity) == "success"


async def test_async_anthropic_sdk_stream_forwards_thinking_to_hosted_vllm(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    identity: Final = f"chatcmpl-messages-stream-{marker}"
    thinking: Final = f"Streaming thought {marker}."
    with wire_server(lambda _: _streamed_completion(identity)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        stream: Final = await client.messages.create(
            model=model,
            max_tokens=256,
            tools=_TOOLS,  # pyright: ignore[reportArgumentType]  # plain JSON tool definitions
            messages=_tool_loop(thinking, marker),  # pyright: ignore[reportArgumentType]  # plain JSON content blocks
            stream=True,
        )
        events: Final = [event async for event in stream]
        assert events[0].type == "message_start" and events[-1].type == "message_stop"
        message_id: Final = events[0].message.id
        assert "".join(
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        ) == ("It is raining.")
        body: Final = _only_body(wire)
        assert body["stream"] is True
        assert _sent_messages(body) == _expected_upstream(thinking, marker)
        assert _spend_status(message_id) == "success"


def test_redacted_thinking_alone_sends_no_reasoning_content(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}"))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        _post_messages(
            gateway,
            model,
            [
                {"role": "user", "content": f"Hello {marker}"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque-redacted"},
                        {"type": "text", "text": "Hi."},
                    ],
                },
                {"role": "user", "content": "Again"},
            ],
        )
        assert _sent_messages(_only_body(wire)) == [
            {"role": "user", "content": f"Hello {marker}"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Again"},
        ]


def test_assistant_turn_without_thinking_sends_no_reasoning_content(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda _: Reply(body=_completion(f"chatcmpl-{marker}"))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        _post_messages(
            gateway,
            model,
            [
                {"role": "user", "content": f"Hello {marker}"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
                {"role": "user", "content": "Again"},
            ],
        )
        assert _sent_messages(_only_body(wire)) == [
            {"role": "user", "content": f"Hello {marker}"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Again"},
        ]
