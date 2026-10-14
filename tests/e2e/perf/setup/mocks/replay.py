"""Real provider responses, captured once and replayed by the mocks.

`captures/<provider>/<route>_<shape>_<json|stream>.json` holds what OpenAI, Anthropic and Bedrock answered
to one request of each shape built by `setup/corpus/session.py`: the body, or the raw stream chunks with the
time each arrived, plus the request size and the usage the provider reported. A mock picks the capture for
its route and stream mode whose request size is closest to the request it received, swaps the provider's
ids for per-request ones, and sends the bytes back on the captured clock (scaled by the `p` knob). Nothing
in the reply is invented, so the gateway sees the provider's own usage, cache fields, stop reasons, tool
calls, reasoning items, chunk boundaries and pacing."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import AsyncIterator, Iterator
from functools import cache
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

CAPTURES_DIR: Final = Path(__file__).resolve().parent / "captures"
Provider = Literal["openai", "anthropic", "bedrock"]
_ID: Final = re.compile(r"\b(chatcmpl-|resp_|rs_|fc_|msg_|toolu_|call_|srvtoolu_)([A-Za-z0-9]{8,})")


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    t: float
    data: str


class Capture(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    route: str
    shape: str
    marker: str
    captured_at: str
    stream: bool
    model: str
    request_bytes: int
    request_tokens_estimate: int
    status: int
    ttfb_s: float
    total_s: float
    body: dict[str, object] | None
    chunks: tuple[Chunk, ...]
    events: tuple[dict[str, object], ...]

    @property
    def body_text(self) -> str:
        return json.dumps(self.body, separators=(",", ":"))

    @property
    def text(self) -> str:
        """The visible reply text, what a client should see once the gateway has translated the body."""
        return "".join(_texts(self))

    @property
    def usage(self) -> dict[str, object]:
        if self.body is not None:
            found: Final = self.body.get("usage")
            return found if isinstance(found, dict) else {}
        return _stream_usage(self)


def _texts(capture: Capture) -> Iterator[str]:
    if capture.body is None:
        yield from _stream_texts(capture)
        return
    body: Final = capture.body
    choices: Final = body.get("choices")
    if isinstance(choices, list) and choices:
        message: Final = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content: Final = message.get("content") if isinstance(message, dict) else None
        yield content if isinstance(content, str) else ""
        return
    output: Final = body.get("output")
    if isinstance(output, dict):
        yield from _bedrock_texts(output)
        return
    for block in output if isinstance(output, list) else body.get("content", []):
        yield from _block_texts(block)


def _bedrock_texts(output: dict[str, object]) -> Iterator[str]:
    message: Final = output.get("message")
    content: Final = message.get("content", []) if isinstance(message, dict) else []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            yield str(block["text"])


def _block_texts(block: object) -> Iterator[str]:
    if not isinstance(block, dict):
        return
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        yield str(block["text"])
    if block.get("type") == "message":
        for part in block.get("content", []) if isinstance(block.get("content"), list) else []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield str(part["text"])


def _sse_payloads(capture: Capture) -> Iterator[dict[str, object]]:
    joined: Final = "".join(chunk.data for chunk in capture.chunks)
    for line in joined.split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            parsed: Final = json.loads(line[6:])
            if isinstance(parsed, dict):
                yield parsed


def _event_texts(capture: Capture) -> Iterator[str]:
    for event in capture.events:
        payload: Final = event.get("payload")
        delta: Final = payload.get("delta") if isinstance(payload, dict) else None
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            yield str(delta["text"])
        elif isinstance(payload, dict) and payload.get("type") == "content_block_delta":
            inner: Final = payload.get("delta", {})
            if isinstance(inner, dict) and inner.get("type") == "text_delta":
                yield str(inner.get("text", ""))


def _stream_texts(capture: Capture) -> Iterator[str]:
    if capture.provider == "bedrock":
        yield from _event_texts(capture)
        return
    for payload in _sse_payloads(capture):
        choices: Final = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta: Final = choices[0].get("delta", {})
            text: Final = delta.get("content") if isinstance(delta, dict) else None
            yield text if isinstance(text, str) else ""
        elif payload.get("type") == "response.output_text.delta":
            yield str(payload.get("delta", ""))
        elif payload.get("type") == "content_block_delta":
            delta_block: Final = payload.get("delta", {})
            if isinstance(delta_block, dict) and delta_block.get("type") == "text_delta":
                yield str(delta_block.get("text", ""))


def _stream_usage(capture: Capture) -> dict[str, object]:
    if capture.provider == "bedrock":
        for event in capture.events:
            payload: Final = event.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
                return dict(payload["usage"])
        return {}
    for payload in _sse_payloads(capture):
        usage: Final = payload.get("usage")
        if isinstance(usage, dict) and usage:
            return usage
        response: Final = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            return dict(response["usage"])
    return {}


@cache
def library(provider: Provider) -> tuple[Capture, ...]:
    paths: Final = sorted((CAPTURES_DIR / provider).glob("*.json"))
    assert paths, f"no captures under {CAPTURES_DIR / provider}"
    return tuple(Capture.model_validate_json(path.read_text()) for path in paths)


def pick(provider: Provider, route: str, stream: bool, prompt_tokens: int) -> Capture:
    """The capture for this route and mode whose request was closest in size to the one just received."""
    matching: Final = tuple(c for c in library(provider) if c.route == route and c.stream == stream)
    assert matching, f"no {provider} capture for route={route} stream={stream}"
    return min(matching, key=lambda c: abs(c.request_tokens_estimate - prompt_tokens))


def fresh_ids(text: str, seed: str) -> str:
    """The captured provider ids replaced by ids derived from the request, same prefix and length."""

    def replace(match: re.Match[str]) -> str:
        digest: Final = hashlib.sha1(f"{seed}:{match.group(2)}".encode()).hexdigest()
        return match.group(1) + digest[: len(match.group(2))]

    return _ID.sub(replace, text)


def body_bytes(capture: Capture, seed: str) -> bytes:
    return fresh_ids(capture.body_text, seed).encode()


def chunk_bytes(capture: Capture, chunk: Chunk, seed: str) -> bytes:
    if capture.provider == "bedrock":
        return base64.b64decode(chunk.data)
    return fresh_ids(chunk.data, seed).encode()


async def paced(capture: Capture, seed: str, scale: float) -> AsyncIterator[bytes]:
    """The captured chunks, each sent at its captured offset times `scale`."""
    loop: Final = asyncio.get_running_loop()
    start: Final = loop.time()
    for chunk in capture.chunks:
        wait: Final = start + chunk.t * scale - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        yield chunk_bytes(capture, chunk, seed)
