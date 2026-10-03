"""
Regression tests for AnthropicPassthroughGuardrailHandler SSE rewriting.

Pins the two Greptile P1s from PR #42585:
- CRLF-framed streams must still reach post-call rewriting
- Multi-index text blocks must keep their own rewrites (not merge into index 0)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest

from litellm.llms.anthropic.passthrough.guardrail_translation.handler import (
    AnthropicPassthroughGuardrailHandler,
    _parse_sse_blocks,
    _processed_texts,
    _text_delta,
    _with_text,
)


def _text_delta_frame(index: int, text: str, sep: bytes = b"\n\n", line_end: bytes | None = None) -> bytes:
    if line_end is None:
        line_end = sep[: len(sep) // 2] or b"\n"
    payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return (
        b"event: content_block_delta" + line_end + b"data: " + json.dumps(payload, separators=(",", ":")).encode() + sep
    )


def _message_stop_frame(sep: bytes = b"\n\n", line_end: bytes | None = None) -> bytes:
    if line_end is None:
        line_end = sep[: len(sep) // 2] or b"\n"
    return b"event: message_stop" + line_end + b'data: {"type":"message_stop"}' + sep


def _frame_payloads(body: bytes) -> list[dict]:
    payloads: list[dict] = []
    for block in _parse_sse_blocks(body):
        for line in block.decode().splitlines():
            if line.startswith("data:"):
                payloads.append(json.loads(line[5:].strip()))
                break
    return payloads


class TestParseSseBlocks:
    def test_splits_lf_crlf_and_cr_blank_lines(self):
        body = b"event: a\ndata: 1\n\nevent: b\r\ndata: 2\r\n\r\nevent: c\rdata: 3\r\r"
        blocks = _parse_sse_blocks(body)
        assert len(blocks) == 3
        assert blocks[0].endswith(b"\n\n")
        assert blocks[1].endswith(b"\r\n\r\n")
        assert blocks[2].endswith(b"\r\r")


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\n\r\n", b"\r\r", b""])
def test_text_rewrite_preserves_index_metadata_and_framing(separator: bytes) -> None:
    line_end = separator[: len(separator) // 2] or b"\n"
    payload = {
        "type": "content_block_delta",
        "index": 3,
        "delta": {"type": "text_delta", "text": "<PERSON_1>", "metadata": {"source": "guardrail"}},
        "extra": [True, None, {"value": 1}],
    }
    frame = b"event: content_block_delta" + line_end + b"data: " + json.dumps(payload).encode() + separator

    rewritten = _with_text(frame, "Alice")

    assert rewritten == (
        b"event: content_block_delta"
        + line_end
        + b"data: "
        + json.dumps({**payload, "delta": {**payload["delta"], "text": "Alice"}}, separators=(",", ":")).encode()
        + separator
    )


@pytest.mark.parametrize(
    "frame",
    [
        b"event: content_block_delta\ndata: {invalid}\n\n",
        b"event: content_block_delta\ndata: []\n\n",
        b"event: content_block_delta\ndata: null\n\n",
        b"event: content_block_delta\ndata: \xff\n\n",
        b'event: content_block_delta\ndata: {"index":0,"delta":"text"}\n\n',
        _message_stop_frame(),
    ],
)
def test_invalid_or_non_delta_frames_stay_unchanged(frame: bytes) -> None:
    assert _text_delta(frame) is None
    assert _with_text(frame, "Alice") == frame


@pytest.mark.parametrize(
    "processed, expected",
    [
        ({"content": [{"text": "Alice"}, {"text": "Bob"}]}, ("Alice", "Bob")),
        ({"content": [{"text": "Alice"}]}, None),
        ({"content": [{"text": "Alice"}, {"text": 42}]}, None),
        ({"content": [{"text": "Alice"}, {"type": "text"}]}, None),
        ({"content": None}, None),
    ],
)
def test_guardrail_texts_require_a_complete_string_rewrite(
    processed: Mapping[str, object], expected: tuple[str, ...] | None
) -> None:
    assert _processed_texts(processed, 2) == expected


class TestDeAnonymizeEventStream:
    def _proxy(self, mock_hook) -> MagicMock:
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_success_hook = mock_hook
        return proxy_logging_obj

    @pytest.mark.asyncio
    @pytest.mark.parametrize("line_end", [b"\n", b"\r\n", b"\r"])
    async def test_multiline_data_reaches_guardrail(self, line_end: bytes):
        frame = line_end.join(
            (
                b"event: content_block_delta",
                b'data: {"type":"content_block_delta","index":3,',
                b'data: "delta":{"type":"text_delta","text":"<PERSON_1>"}}',
                b"",
                b"",
            )
        )
        stop = _message_stop_frame(sep=line_end * 2)

        async def hook(data, user_api_key_dict, response):
            assert response["content"] == [{"type": "text", "text": "<PERSON_1>"}]
            return {**response, "content": [{"type": "text", "text": "Alice"}]}

        result = await AnthropicPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=frame + stop,
            proxy_logging_obj=self._proxy(hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        assert _text_delta(_parse_sse_blocks(result)[0]) == (3, "Alice")
        assert b"<PERSON_1>" not in result
        assert result.endswith(stop)

    @pytest.mark.asyncio
    async def test_crlf_framed_stream_still_invokes_guardrail(self):
        """P1: CRLF frames must not merge so message_stop wins and deltas skip rewriting."""
        sse = _text_delta_frame(0, "<PERSON_1>", sep=b"\r\n\r\n") + _message_stop_frame(sep=b"\r\n\r\n")
        hook_calls: list[dict] = []

        async def mock_hook(data, user_api_key_dict, response):
            hook_calls.append(response)
            response = dict(response)
            response["content"] = [{"type": "text", "text": "Alice"}]
            return response

        # Precondition: a naive LF-only split would leave one merged block.
        assert len(sse.split(b"\n\n")) == 1

        result = await AnthropicPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=sse,
            proxy_logging_obj=self._proxy(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        assert len(hook_calls) == 1
        assert hook_calls[0]["content"][0]["text"] == "<PERSON_1>"
        assert b"<PERSON_1>" not in result
        assert b"Alice" in result
        assert result.endswith(b'data: {"type":"message_stop"}\r\n\r\n')

    @pytest.mark.asyncio
    async def test_multi_index_text_blocks_keep_their_own_rewrites(self):
        """P1: rewrites must stay on their content-block index around tool/thinking events."""
        tool_delta = (
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"input_json_delta","partial_json":"{}"}}\n\n'
        )
        sse = (
            _text_delta_frame(0, "<PERSON_")
            + _text_delta_frame(0, "1> said")
            + tool_delta
            + _text_delta_frame(2, "bye <PERSON_2>")
        )
        seen: dict[str, list[str]] = {}

        async def mock_hook(data, user_api_key_dict, response):
            seen["texts"] = [block["text"] for block in response["content"]]
            response = dict(response)
            response["content"] = [
                {"type": "text", "text": "Alice said"},
                {"type": "text", "text": "bye Bob"},
            ]
            return response

        result = await AnthropicPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=sse,
            proxy_logging_obj=self._proxy(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        assert seen["texts"] == ["<PERSON_1> said", "bye <PERSON_2>"]
        payloads = _frame_payloads(result)
        assert [(p["index"], p["delta"].get("text")) for p in payloads] == [
            (0, "Alice said"),
            (0, ""),
            (1, None),
            (2, "bye Bob"),
        ]
        # Tool frame between text blocks must stay untouched and in order.
        assert payloads[2]["delta"]["type"] == "input_json_delta"
        assert payloads[2]["delta"]["partial_json"] == "{}"
