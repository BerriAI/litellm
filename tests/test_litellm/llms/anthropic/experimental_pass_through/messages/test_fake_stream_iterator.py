"""
Unit tests for FakeAnthropicMessagesStreamIterator.

Regression coverage for websearch interception: the native
``server_tool_use`` / ``web_search_tool_result`` blocks that
``_inject_native_blocks`` adds must survive the non-streaming -> streaming
re-wrap, not collapse into a bare ``content_block_stop``.
"""

import json
from typing import Any, Dict, List, cast

from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


def _events(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for chunk in FakeAnthropicMessagesStreamIterator(response=cast(AnthropicMessagesResponse, response))
        for line in chunk.decode().splitlines()
        if line.startswith("data: ")
    ]


def _response(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_web_search_tool_result_block_is_emitted_with_payload():
    search_result_block = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://docs.litellm.ai",
                "title": "LiteLLM docs",
                "page_age": None,
            }
        ],
    }
    events = _events(_response([search_result_block, {"type": "text", "text": "hi"}]))

    starts = [e for e in events if e["type"] == "content_block_start"]
    assert {"index": 0, "type": "content_block_start", "content_block": search_result_block} in starts
    assert [e["index"] for e in events if e["type"] == "content_block_stop"] == [0, 1]


def test_server_tool_use_block_keeps_type_and_input():
    events = _events(
        _response(
            [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "litellm"},
                }
            ]
        )
    )

    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"] == {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {},
    }
    delta = next(e for e in events if e["type"] == "content_block_delta")
    assert json.loads(delta["delta"]["partial_json"]) == {"query": "litellm"}


def test_unknown_block_type_is_passed_through_verbatim():
    block = {"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_2", "content": {"stdout": "42"}}
    events = _events(_response([block]))

    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"] == block


def test_text_block_still_streams_as_delta():
    events = _events(_response([{"type": "text", "text": "hello"}]))

    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"] == {"type": "text", "text": ""}
    delta = next(e for e in events if e["type"] == "content_block_delta")
    assert delta["delta"] == {"type": "text_delta", "text": "hello"}
