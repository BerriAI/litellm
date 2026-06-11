"""Differential parity for streaming: v1's full CustomStreamWrapper replay vs
v2's engine/stream fold over the same recorded SSE lines.

The replay method mirrors the characterization corpus
(tests/translation_characterization/_seams.py): raw SSE lines through the
real ``ModelResponseIterator`` inside ``CustomStreamWrapper`` on the v1 side;
``engine.stream.fold_lines`` plus the ``ModelResponseStream`` seam adapter on
the v2 side. uuid/time are frozen; stream ids are ambient and normalized.
"""

import json

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from litellm.translation.engine.stream import fold_lines
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.providers.anthropic.stream import parse_sse_line, reverse_names
from litellm.translation_seam import to_model_response_stream

MODEL = "claude-sonnet-4-5"

FROZEN_TIME = 1718064000.0


def _sse(events: list) -> list:
    lines = []
    for event in events:
        lines.append(f"event: {event['type']}")
        lines.append("data: " + json.dumps(event))
        lines.append("")
    return lines


_MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_stream_01",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 1},
    },
}

STREAMS = {
    "text": _sse(
        [
            _MESSAGE_START,
            {"type": "ping"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Paris is the"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " capital."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
            {"type": "message_stop"},
        ]
    ),
    "tools": _sse(
        [
            _MESSAGE_START,
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Checking."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_s1",
                    "name": "mcp_server_get_weather",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'ty": "Paris"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 30},
            },
            {"type": "message_stop"},
        ]
    ),
    "thinking": _sse(
        [
            _MESSAGE_START,
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "France. Capital."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig=="},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Paris."},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
            {"type": "message_stop"},
        ]
    ),
}

_REQUEST = {
    "model": MODEL,
    "max_tokens": 64,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "mcp.server/get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
    "messages": [{"role": "user", "content": "weather in Paris"}],
}


@pytest.fixture(autouse=True)
def _deterministic_ambient(monkeypatch):
    import time
    import uuid

    counter = iter(range(1, 10_000))
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))
    monkeypatch.setattr(time, "time", lambda: FROZEN_TIME)


def _v1_chunks(lines: list) -> list:
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    # Production passes the per-request tool-name reverse map into the
    # iterator (built by transform_request); replicate that wiring.
    config = AnthropicConfig()
    params = {k: v for k, v in _REQUEST.items() if k not in ("model", "messages")}
    optional = config.map_openai_params(dict(params), {}, MODEL, drop_params=False)
    litellm_params: dict = {}
    config.transform_request(MODEL, [dict(m) for m in _REQUEST["messages"]], optional, litellm_params, {})
    reverse_map = litellm_params.get("_anthropic_tool_name_map") or {}

    iterator = ModelResponseIterator(
        streaming_response=iter(lines),
        sync_stream=True,
        tool_name_reverse_map=reverse_map,
    )
    wrapper = CustomStreamWrapper(
        completion_stream=iterator,
        model=MODEL,
        custom_llm_provider="anthropic",
        logging_obj=Logging(
            model=MODEL,
            messages=[{"role": "user", "content": "stream"}],
            stream=True,
            call_type="completion",
            start_time=None,
            litellm_call_id="diff-stream",
            function_id="diff-stream",
        ),
    )
    return [chunk.model_dump() for chunk in wrapper]


def _v2_chunks(lines: list) -> list:
    parsed = parse_request(dict(_REQUEST))
    assert parsed.is_ok(), parsed.error.summary
    reverse = reverse_names(parsed.ok)
    folded = fold_lines(lines, lambda line: parse_sse_line(line, reverse))
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-X").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list) -> str:
    return json.dumps(
        [{**chunk, "id": "chatcmpl-X"} for chunk in chunks], sort_keys=True, default=str
    )


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_matches_v1(name: str) -> None:
    lines = STREAMS[name]
    assert _norm(_v2_chunks(lines)) == _norm(_v1_chunks(lines))
