import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_BACKEND: Final = "deepseek-v4-pro"
_API_KEY: Final = "synthetic-openai-key"
_TOOL: Final = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}


def _response(identity: str, output: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1789788253,
        "status": "completed",
        "model": _BACKEND,
        "output": list(output),
        "usage": {"input_tokens": 30, "output_tokens": 9, "total_tokens": 39},
    }


def _reasoning(identity: str, text: str, encrypted_content: str | None = None) -> dict[str, object]:
    return {
        "type": "reasoning",
        "id": f"rs_{identity}",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        **({"encrypted_content": encrypted_content} if encrypted_content is not None else {}),
    }


def _function_call(identity: str, suffix: str, command: str) -> dict[str, object]:
    return {
        "type": "function_call",
        "id": f"fc_{identity}_{suffix}",
        "call_id": f"call_{identity}_{suffix}",
        "name": "Bash",
        "arguments": json.dumps({"command": command}),
        "status": "completed",
    }


def _message(identity: str, text: str) -> dict[str, object]:
    return {
        "type": "message",
        "id": f"msg_{identity}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _sse(events: Sequence[Mapping[str, object]]) -> tuple[bytes, ...]:
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _messages_request(model: str, messages: Sequence[Mapping[str, object]], stream: bool = False) -> dict[str, object]:
    return {
        "model": model,
        "max_tokens": 256,
        "stream": stream,
        "tools": [_TOOL],
        "messages": list(messages),
    }


def _tool_results(identity: str, outputs: Mapping[str, str]) -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": f"call_{identity}_{suffix}", "content": output}
            for suffix, output in outputs.items()
        ],
    }


def _thinking_texts(content: Sequence[Mapping[str, object]]) -> list[object]:
    return [block.get("thinking") for block in content if block.get("type") == "thinking"]


def test_messages_returns_reasoning_text_as_thinking_and_replays_it_before_tool_calls_in_order(
    gateway: Gateway,
) -> None:
    """regression: reasoning_text was dropped and assistant text landed between function_call and its output"""
    identity: Final = f"messages-reasoning-text-{uuid.uuid4().hex}"
    thought: Final = f"I should echo alpha and beta in parallel {identity}"
    first_turn: Final = _response(
        identity,
        (
            _reasoning(identity, thought),
            _message(identity, "Running two commands."),
            _function_call(identity, "a", "echo alpha"),
            _function_call(identity, "b", "echo beta"),
        ),
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        body: Final = json.loads(request.body)
        has_outputs: Final = any(item.get("type") == "function_call_output" for item in body["input"])
        output: Final = (_message(identity, "alpha beta"),) if has_outputs else ()
        return Reply(body=json.dumps(_response(identity, output) if has_outputs else first_turn).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        user_turn: Final = {"role": "user", "content": f"echo alpha and beta in parallel {identity}"}
        first: Final = gateway.request("POST", "/v1/messages", _messages_request(model, [user_turn]))
        assert first.status_code == 200, first.text
        assistant_content: Final = first.json()["content"]
        assert _thinking_texts(assistant_content) == [thought], first.text
        assert [block["type"] for block in assistant_content] == ["thinking", "text", "tool_use", "tool_use"]

        second: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_request(
                model,
                [
                    user_turn,
                    {"role": "assistant", "content": assistant_content},
                    _tool_results(identity, {"a": "alpha", "b": "beta"}),
                ],
            ),
        )
        assert second.status_code == 200, second.text
        requests: Final = wire.drain()
        assert len(requests) == 2, second.text
        replayed: Final = json.loads(requests[1].body)["input"]
        assert [(item.get("type"), item.get("call_id")) for item in replayed[1:]] == [
            ("reasoning", None),
            ("message", None),
            ("function_call", f"call_{identity}_a"),
            ("function_call", f"call_{identity}_b"),
            ("function_call_output", f"call_{identity}_a"),
            ("function_call_output", f"call_{identity}_b"),
        ], replayed
        assert replayed[1]["content"] == [{"type": "reasoning_text", "text": thought}], replayed[1]
        assert replayed[2]["content"] == [{"type": "output_text", "text": "Running two commands."}], replayed[2]


def test_messages_replays_reasoning_text_with_its_encrypted_content(gateway: Gateway) -> None:
    """regression: the replayed reasoning item carried encrypted_content but not the reasoning_text it came with"""
    identity: Final = f"messages-reasoning-text-encrypted-{uuid.uuid4().hex}"
    thought: Final = f"Think about {identity}"
    encrypted: Final = f"enc_{identity}"
    first_turn: Final = _response(
        identity, (_reasoning(identity, thought, encrypted), _function_call(identity, "a", "echo alpha"))
    )

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        has_outputs: Final = any(item.get("type") == "function_call_output" for item in body["input"])
        return Reply(
            body=json.dumps(_response(identity, (_message(identity, "alpha"),)) if has_outputs else first_turn).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        user_turn: Final = {"role": "user", "content": f"echo alpha {identity}"}
        first: Final = gateway.request("POST", "/v1/messages", _messages_request(model, [user_turn]))
        assert first.status_code == 200, first.text
        assistant_content: Final = first.json()["content"]
        assert _thinking_texts(assistant_content) == [thought], first.text
        second: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_request(
                model,
                [
                    user_turn,
                    {"role": "assistant", "content": assistant_content},
                    _tool_results(identity, {"a": "alpha"}),
                ],
            ),
        )
        assert second.status_code == 200, second.text
        replayed: Final = json.loads(wire.drain()[1].body)["input"]
        reasoning_items: Final = [item for item in replayed if item.get("type") == "reasoning"]
        assert reasoning_items == [
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": thought}],
                "encrypted_content": encrypted,
            }
        ], replayed


def test_messages_stream_emits_reasoning_text_deltas_as_thinking(gateway: Gateway) -> None:
    """regression: response.reasoning_text.delta events never reached the client as thinking_delta"""
    identity: Final = f"messages-reasoning-text-stream-{uuid.uuid4().hex}"
    pieces: Final = ("Let me ", "think ", identity)
    reasoning_item: Final = _reasoning(identity, "".join(pieces))
    completed: Final = _response(identity, (reasoning_item, _message(identity, "done")))
    events: Final[tuple[Mapping[str, object], ...]] = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning_item, "content": []}},
        *(
            {
                "type": "response.reasoning_text.delta",
                "item_id": f"rs_{identity}",
                "output_index": 0,
                "content_index": 0,
                "delta": piece,
            }
            for piece in pieces
        ),
        {
            "type": "response.reasoning_text.done",
            "item_id": f"rs_{identity}",
            "output_index": 0,
            "content_index": 0,
            "text": "".join(pieces),
        },
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning_item},
        {"type": "response.output_item.added", "output_index": 1, "item": {**_message(identity, ""), "content": []}},
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{identity}",
            "output_index": 1,
            "content_index": 0,
            "delta": "done",
        },
        {"type": "response.output_item.done", "output_index": 1, "item": _message(identity, "done")},
        {"type": "response.completed", "response": completed},
    )

    def respond(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=_sse(events))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_request(model, [{"role": "user", "content": f"think {identity}"}], stream=True),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
        frames: Final = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        started: Final = [frame["content_block"]["type"] for frame in frames if frame["type"] == "content_block_start"]
        thinking: Final = "".join(
            frame["delta"]["thinking"]
            for frame in frames
            if frame["type"] == "content_block_delta" and frame["delta"]["type"] == "thinking_delta"
        )
        assert started == ["thinking", "text"], response.text
        assert thinking == "".join(pieces), response.text


def test_messages_summary_reasoning_with_encrypted_content_replays_as_before(gateway: Gateway) -> None:
    """guard: OpenAI style summary plus encrypted_content reasoning keeps its existing round trip"""
    identity: Final = f"messages-reasoning-summary-{uuid.uuid4().hex}"
    encrypted: Final = f"enc_{identity}"
    first_turn: Final = _response(
        identity,
        (
            {
                "type": "reasoning",
                "id": f"rs_{identity}",
                "summary": [{"type": "summary_text", "text": f"Summary {identity}"}],
                "encrypted_content": encrypted,
            },
            _message(identity, "Hello"),
        ),
    )

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(first_turn).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        user_turn: Final = {"role": "user", "content": f"hi {identity}"}
        first: Final = gateway.request("POST", "/v1/messages", _messages_request(model, [user_turn]))
        assert first.status_code == 200, first.text
        assistant_content: Final = first.json()["content"]
        assert _thinking_texts(assistant_content) == [f"Summary {identity}"], first.text
        second: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_request(
                model,
                [user_turn, {"role": "assistant", "content": assistant_content}, {"role": "user", "content": "again"}],
            ),
        )
        assert second.status_code == 200, second.text
        replayed: Final = json.loads(wire.drain()[1].body)["input"]
        assert replayed[1:3] == [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": f"Summary {identity}"}],
                "encrypted_content": encrypted,
            },
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
        ], replayed


def test_messages_replays_assistant_text_in_its_original_place_among_tool_calls(gateway: Gateway) -> None:
    """regression: assistant text was moved after every function_call, splitting calls from their outputs"""
    identity: Final = f"messages-assistant-order-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(_response(identity, (_message(identity, "done"),))).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_request(
                model,
                [
                    {"role": "user", "content": f"run three commands {identity}"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "First I will look."},
                            {
                                "type": "tool_use",
                                "id": f"call_{identity}_a",
                                "name": "Bash",
                                "input": {"command": "ls"},
                            },
                            {"type": "text", "text": "Then two more."},
                            {"type": "text", "text": "In parallel."},
                            {
                                "type": "tool_use",
                                "id": f"call_{identity}_b",
                                "name": "Bash",
                                "input": {"command": "pwd"},
                            },
                            {
                                "type": "tool_use",
                                "id": f"call_{identity}_c",
                                "name": "Bash",
                                "input": {"command": "id"},
                            },
                        ],
                    },
                    _tool_results(identity, {"a": "a.txt", "b": "/tmp", "c": "uid=0"}),
                ],
            ),
        )
        assert response.status_code == 200, response.text
        replayed: Final = json.loads(wire.drain()[0].body)["input"]
        assert [(item.get("type"), item.get("call_id"), item.get("content")) for item in replayed[1:]] == [
            ("message", None, [{"type": "output_text", "text": "First I will look."}]),
            ("function_call", f"call_{identity}_a", None),
            (
                "message",
                None,
                [{"type": "output_text", "text": "Then two more."}, {"type": "output_text", "text": "In parallel."}],
            ),
            ("function_call", f"call_{identity}_b", None),
            ("function_call", f"call_{identity}_c", None),
            ("function_call_output", f"call_{identity}_a", None),
            ("function_call_output", f"call_{identity}_b", None),
            ("function_call_output", f"call_{identity}_c", None),
        ], replayed
