"""Bounded Anthropic SSE rewriting for the public model alias."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.common_utils.sse_keepalive import split_complete_sse_frames

MAX_MESSAGE_START_BUFFER_BYTES: Final = 1024 * 1024
_SSE_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _restamp_message_start_data_line(line: bytes, requested_model: str) -> tuple[bytes, bool]:
    """Replace the model in one Anthropic message_start data line."""
    body: Final = line.rstrip(b"\r\n")
    line_ending: Final = line[len(body) :]
    if not body.startswith(b"data:"):
        return line, False
    try:
        payload: Final = _SSE_OBJECT_ADAPTER.validate_json(body[len(b"data:") :].strip())
    except ValidationError:
        return line, False
    if payload.get("type") != "message_start":
        return line, False
    try:
        message: Final = _SSE_OBJECT_ADAPTER.validate_python(payload.get("message"))
    except ValidationError:
        return line, False
    if "model" not in message:
        return line, False
    restamped_payload: Final = {**payload, "message": {**message, "model": requested_model}}
    rewritten: Final = b"data: " + json.dumps(restamped_payload, separators=(",", ":")).encode() + line_ending
    return rewritten, True


def _restamp_message_start_frames(frames: bytes, requested_model: str) -> tuple[bytes, bool]:
    """Restore the alias in complete frames and report when message_start was found."""
    output: list[bytes] = []
    restamped = False
    for line in frames.splitlines(keepends=True):
        rewritten_line, line_restamped = _restamp_message_start_data_line(line, requested_model)
        output.append(rewritten_line)
        restamped = restamped or line_restamped
    return b"".join(output), restamped


class AnthropicMessageStartModelRewriter:
    """Reassemble only the first Anthropic frames with a bounded pending buffer."""

    __slots__ = ("_finished", "_pending", "_requested_model")

    def __init__(self, requested_model: str) -> None:
        self._requested_model: Final = requested_model
        self._pending = b""
        self._finished = False

    def feed(self, chunk: bytes) -> bytes:
        """Return rewritten frames, or bounded pass-through for a malformed stream."""
        if self._finished:
            return chunk

        buffered: Final = self._pending + chunk
        if len(buffered) > MAX_MESSAGE_START_BUFFER_BYTES:
            self._pending = b""
            self._finished = True
            return buffered

        complete_frames, self._pending = split_complete_sse_frames(buffered)
        rewritten_frames, restamped = _restamp_message_start_frames(
            complete_frames,
            self._requested_model,
        )
        if not restamped:
            return rewritten_frames

        tail: Final = self._pending
        self._pending = b""
        self._finished = True
        return rewritten_frames + tail

    def flush(self) -> bytes:
        """Return the final unterminated tail without dropping provider bytes."""
        tail: Final = self._pending
        self._pending = b""
        return tail


async def restamp_anthropic_streaming_response_model(
    response: AsyncIterator[object],
    requested_model: str,
) -> AsyncGenerator[object, None]:
    """Restamp message_start while preserving arbitrary transport chunk boundaries."""
    rewriter: Final = AnthropicMessageStartModelRewriter(requested_model)
    async for chunk in response:
        if isinstance(chunk, (bytes, bytearray)):
            if rewritten_frames := rewriter.feed(bytes(chunk)):
                yield rewritten_frames
            continue
        if buffered_tail := rewriter.flush():
            yield buffered_tail
        yield chunk
    final_tail: Final = rewriter.flush()
    if final_tail:
        yield final_tail
