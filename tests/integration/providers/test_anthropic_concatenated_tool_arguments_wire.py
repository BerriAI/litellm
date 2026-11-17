import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

MODEL: Final = "claude-opus-5-5"
FILE_CONTENTS: Final = "alpha bravo charlie"
FINAL_REPLY: Final = "Both files are read."
READ_FILE_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}


class _Block(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str
    id: str | None = None
    name: str | None = None
    input: JsonValue = None
    tool_use_id: str | None = None
    content: JsonValue = None


class _Turn(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    role: str
    content: str | tuple[_Block, ...]


class _AnthropicBody(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    messages: tuple[_Turn, ...]


ToolUse = tuple[str | None, str | None, JsonValue]
ToolResult = tuple[str | None, str]


def _anthropic_reply(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages", request.target
    message: Final = {
        "id": "msg_concatenated_tool_arguments",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": FINAL_REPLY}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 40, "output_tokens": 6},
    }
    return Reply(body=json.dumps(message).encode())


@contextmanager
def _anthropic_model(gateway: Gateway) -> Iterator[tuple[str, Wire]]:
    with wire_server(_anthropic_reply) as wire, gateway.scenario() as scenario:
        yield scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key"), wire


def _chat_after_read_file_call(gateway: Gateway, model: str, arguments: str) -> JsonValue:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 64,
            "tools": [READ_FILE_TOOL],
            "messages": [
                {"role": "user", "content": "Read /a and /b."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": arguments}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": FILE_CONTENTS},
            ],
        },
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert isinstance(body, dict) and isinstance(body["choices"], list), body
    choice: Final = body["choices"][0]
    assert isinstance(choice, dict) and isinstance(choice["message"], dict), body
    return choice["message"]["content"]


def _text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    assert isinstance(content, list), content
    return "".join(str(part["text"]) for part in content if isinstance(part, dict) and part.get("type") == "text")


def _blocks(turn: _Turn) -> tuple[_Block, ...]:
    assert isinstance(turn.content, tuple), turn
    return turn.content


def _sent_tool_round(wire: Wire) -> tuple[tuple[ToolUse, ...], tuple[ToolResult, ...]]:
    requests: Final = wire.drain()
    assert len(requests) == 1, requests
    messages: Final = _AnthropicBody.model_validate_json(requests[0].body).messages
    assistant_index: Final = next(index for index, turn in enumerate(messages) if turn.role == "assistant")
    tool_uses: Final = tuple(
        (block.id, block.name, block.input) for block in _blocks(messages[assistant_index]) if block.type == "tool_use"
    )
    tool_results: Final = tuple(
        (block.tool_use_id, _text(block.content))
        for block in _blocks(messages[assistant_index + 1])
        if block.type == "tool_result"
    )
    return tool_uses, tool_results


def test_chat_splits_concatenated_tool_call_arguments_into_paired_tool_uses(gateway: Gateway) -> None:
    with _anthropic_model(gateway) as (model, wire):
        reply: Final = _chat_after_read_file_call(gateway, model, '{"path": "/a"}{"path": "/b"}')
        tool_uses, tool_results = _sent_tool_round(wire)

    first_call, (split_id, split_name, split_input) = tool_uses
    assert first_call == ("call_1", "read_file", {"path": "/a"})
    assert split_id != "call_1"
    assert (split_name, split_input) == ("read_file", {"path": "/b"})
    assert sorted(str(tool_use_id) for tool_use_id, _ in tool_results) == sorted(
        str(tool_use_id) for tool_use_id, _, _ in tool_uses
    )
    result_text: Final = dict(tool_results)
    assert result_text["call_1"] == FILE_CONTENTS
    assert "call_1" in result_text[split_id]
    assert FILE_CONTENTS not in result_text[split_id]
    assert reply == FINAL_REPLY


def test_chat_sends_valid_tool_call_arguments_as_one_tool_use(gateway: Gateway) -> None:
    with _anthropic_model(gateway) as (model, wire):
        reply: Final = _chat_after_read_file_call(gateway, model, '{"path": "/a"}')
        sent: Final = _sent_tool_round(wire)

    assert sent == ((("call_1", "read_file", {"path": "/a"}),), (("call_1", FILE_CONTENTS),))
    assert reply == FINAL_REPLY
