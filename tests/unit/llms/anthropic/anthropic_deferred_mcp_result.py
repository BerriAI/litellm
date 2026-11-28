import json
from collections.abc import Mapping
from typing import Final

# Captured from the Anthropic Messages API (claude-haiku-5-5, mcp-client-2025-04-04, DeepWiki) on 2026-11-18, shortened.
# Anthropic deferred the MCP result to turn two and answered 400 to turn three once that mcp_tool_result was dropped.

DEFERRED_MCP_PROMPT: Final = "Use read_wiki_structure for repo facebook/react and also call get_local_time, then stop."
DEFERRED_MCP_FOLLOW_UP: Final = (
    "Do not call any tool. From the tool result you already have, what is the exact title of page 2.3? "
    "Answer with the title only."
)
DEFERRED_MCP_SERVER: Final = {"type": "url", "url": "https://mcp.deepwiki.com/mcp", "name": "deepwiki"}
LOCAL_TIME_TOOL: Final = {
    "name": "get_local_time",
    "description": "Return the current local time of the machine running the client.",
    "input_schema": {"type": "object", "properties": {}},
}
LOCAL_TIME_OUTPUT: Final = "2026-11-18T17:05:00"

DEFERRED_MCP_CALL_ID: Final = "mcptoolu_01VPWoaESGm7VrP3d4ZbsYv8"
CLIENT_TOOL_CALL_ID: Final = "toolu_01FD68iVnx5LeCtWrqHu9A1j"
MCP_TOOL_ARGUMENTS: Final = {"repoName": "facebook/react"}
MCP_TOOL_USE: Final = {
    "type": "mcp_tool_use",
    "id": DEFERRED_MCP_CALL_ID,
    "name": "read_wiki_structure",
    "input": MCP_TOOL_ARGUMENTS,
    "server_name": "deepwiki",
}
CLIENT_TOOL_USE: Final = {"type": "tool_use", "id": CLIENT_TOOL_CALL_ID, "name": "get_local_time", "input": {}}
WIKI_PAGES: Final = (
    "Available pages for facebook/react:\n\n- 1 React Repository Overview\n- 2 Core Reconciler Architecture\n"
    "  - 2.3 Host Configurations and Renderer Interface\n- 8 Glossary"
)
DEFERRED_MCP_TOOL_RESULT: Final = {
    "type": "mcp_tool_result",
    "tool_use_id": DEFERRED_MCP_CALL_ID,
    "is_error": False,
    "content": [{"type": "text", "text": WIKI_PAGES}],
}
TURN_TWO_TEXT: Final = "Both calls completed.\n\n2.3 Host Configurations and Renderer Interface\n\nLocal time: 17:05:00"
TURN_THREE_TEXT: Final = "Host Configurations and Renderer Interface"


def _message(message_id: str, content: list[Mapping[str, object]], stop_reason: str) -> Mapping[str, object]:
    return {
        "model": "claude-haiku-5-5",
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 869, "output_tokens": 89},
    }


TURN_ONE_MESSAGE: Final = _message(
    "msg_011ChBPPcnaQgjExrCjMxEeN", [MCP_TOOL_USE, {**CLIENT_TOOL_USE, "caller": {"type": "direct"}}], "tool_use"
)
TURN_TWO_MESSAGE: Final = _message(
    "msg_011ChBPPiWrw16vyD1yg8FfY", [DEFERRED_MCP_TOOL_RESULT, {"type": "text", "text": TURN_TWO_TEXT}], "end_turn"
)
TURN_THREE_MESSAGE: Final = _message(
    "msg_011ChBPT9pDEaiduyzBLoyvM", [{"type": "text", "text": TURN_THREE_TEXT}], "end_turn"
)


def _stream_events(
    message_id: str, *blocks: Mapping[str, object], stop_reason: str
) -> tuple[Mapping[str, object], ...]:
    return (
        {"type": "message_start", "message": {**_message(message_id, [], ""), "stop_reason": None}},
        *blocks,
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 89},
        },
        {"type": "message_stop"},
    )


def _input_json_delta(index: int, partial_json: str) -> Mapping[str, object]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }


def _text_delta(text: str) -> Mapping[str, object]:
    return {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}}


TURN_ONE_STREAM_EVENTS: Final = _stream_events(
    "msg_011ChBPTFAS26sxUJ2veJiAX",
    {"type": "content_block_start", "index": 0, "content_block": {**MCP_TOOL_USE, "input": {}}},
    *(_input_json_delta(0, fragment) for fragment in ("", '{"repoN', 'ame": "', "faceboo", "k/react", '"}')),
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {**CLIENT_TOOL_USE, "caller": {"type": "direct"}},
    },
    _input_json_delta(1, ""),
    {"type": "content_block_stop", "index": 1},
    stop_reason="tool_use",
)
TURN_TWO_STREAM_EVENTS: Final = _stream_events(
    "msg_011ChBPTLgKRPzKbRArduK6P",
    {"type": "content_block_start", "index": 0, "content_block": DEFERRED_MCP_TOOL_RESULT},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    *(_text_delta(text) for text in (TURN_TWO_TEXT[:9], TURN_TWO_TEXT[9:40], TURN_TWO_TEXT[40:])),
    {"type": "content_block_stop", "index": 1},
    stop_reason="end_turn",
)


def anthropic_sse(events: tuple[Mapping[str, object], ...]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
