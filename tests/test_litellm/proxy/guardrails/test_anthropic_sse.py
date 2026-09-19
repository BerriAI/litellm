"""Assembly contract for raw Anthropic SSE streams.

Four guardrails (bedrock, model_armor, tool_permission, thirdlaw) and the OTel request/response
recorder all scan the body this module assembles, so anything it drops is invisible to every one
of them.
"""

import json
from collections.abc import Sequence

from litellm.proxy.guardrails.anthropic_sse import assemble_anthropic_sse_body


def _frames(*events: tuple[str, dict[str, object]]) -> list[bytes]:
    return [f"event: {name}\ndata: {json.dumps(body)}\n\n".encode() for name, body in events]


def _message_start() -> tuple[str, dict[str, object]]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
    )


def _tool_block_start(index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
    )


def _input_delta(partial_json: str, index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def _assembled_tool_input(*events: tuple[str, dict[str, object]]) -> object:
    """The tool block as it lands in the native Anthropic body, which is the shape the /v1/messages
    scan is handed. The ModelResponse conversion keeps the raw partial JSON by another route, so
    asserting through that one would pass either way."""
    body = assemble_anthropic_sse_body(_frames(*events))
    assert body is not None
    content = body["content"]
    assert isinstance(content, Sequence)
    return content[0]["input"]


def test_complete_tool_arguments_assemble_as_the_object_the_model_sent():
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        _input_delta('{"query": "weather'),
        _input_delta(' in sf"}'),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )
    assert arguments == {"query": "weather in sf"}


def test_truncated_tool_arguments_keep_the_text_instead_of_dropping_it():
    """max_tokens can cut a tool argument mid-JSON. That fragment still reached the client, so
    discarding it here would hand a caller a way to smuggle text past every scanner above."""
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        _input_delta('{"query": "sk-live-TRUNC'),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    assert "sk-live-TRUNC" in json.dumps(arguments)


def test_a_tool_block_with_no_arguments_assembles_as_an_empty_object():
    """A tool called with no arguments is not truncation, so it must not pick up a raw fragment."""
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )
    assert arguments == {}
