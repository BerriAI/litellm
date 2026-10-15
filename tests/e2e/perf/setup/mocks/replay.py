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
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

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
    stream: bool
    request_tokens_estimate: int
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


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _items(value: object) -> tuple[object, ...]:
    return tuple(cast(list[object], value)) if isinstance(value, list) else ()


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _texts(capture: Capture) -> Iterator[str]:
    if capture.body is None:
        yield from _stream_texts(capture)
        return
    body: Final = capture.body
    choices: Final = _items(body.get("choices"))
    if choices:
        yield _str(_mapping(_mapping(choices[0]).get("message")).get("content"))
        return
    output: Final = body.get("output")
    if isinstance(output, dict):
        yield from _bedrock_texts(_mapping(body.get("output")))
        return
    for block in _items(body.get("output")) if isinstance(output, list) else _items(body.get("content")):
        yield from _block_texts(block)


def _bedrock_texts(output: dict[str, object]) -> Iterator[str]:
    for block in _items(_mapping(output.get("message")).get("content")):
        yield _str(_mapping(block).get("text"))


def _block_texts(block: object) -> Iterator[str]:
    mapping: Final = _mapping(block)
    if mapping.get("type") == "text":
        yield _str(mapping.get("text"))
    if mapping.get("type") == "message":
        for part in _items(mapping.get("content")):
            yield _str(_mapping(part).get("text"))


_PAYLOAD: Final = TypeAdapter(dict[str, object])


def _payload(line: str) -> dict[str, object]:
    return _PAYLOAD.validate_json(line[6:])


def _sse_payloads(capture: Capture) -> Iterator[dict[str, object]]:
    joined: Final = "".join(chunk.data for chunk in capture.chunks)
    for line in joined.split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            yield _payload(line)


def _sse_text(payload: dict[str, object]) -> str:
    choices: Final = _items(payload.get("choices"))
    if choices:
        return _str(_mapping(_mapping(choices[0]).get("delta")).get("content"))
    if payload.get("type") == "response.output_text.delta":
        return _str(payload.get("delta"))
    delta: Final = _mapping(payload.get("delta"))
    if payload.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
        return _str(delta.get("text"))
    return ""


def _event_text(event: dict[str, object]) -> str:
    payload: Final = _mapping(event.get("payload"))
    delta: Final = _mapping(payload.get("delta"))
    if payload.get("type") == "content_block_delta":
        return _str(delta.get("text")) if delta.get("type") == "text_delta" else ""
    return _str(delta.get("text"))


def _stream_texts(capture: Capture) -> Iterator[str]:
    if capture.provider == "bedrock":
        yield from (_event_text(event) for event in capture.events)
        return
    yield from (_sse_text(payload) for payload in _sse_payloads(capture))


@cache
def library(provider: Provider) -> tuple[Capture, ...]:
    paths: Final = sorted((CAPTURES_DIR / provider).glob("*.json"))
    assert paths, f"no captures under {CAPTURES_DIR / provider}"
    return tuple(Capture.model_validate_json(path.read_text()) for path in paths)


def _matching(provider: Provider, route: str, stream: bool) -> tuple[Capture, ...]:
    matching: Final = tuple(c for c in library(provider) if c.route == route and c.stream == stream)
    assert matching, f"no {provider} capture for route={route} stream={stream}"
    return matching


def pick(provider: Provider, route: str, stream: bool, prompt_tokens: int) -> Capture:
    """The capture for this route and mode whose request was closest in size to the one just received."""
    return min(_matching(provider, route, stream), key=lambda c: abs(c.request_tokens_estimate - prompt_tokens))


def first_byte_s(provider: Provider, route: str, stream: bool, prompt_tokens: int) -> float:
    """Seconds the real provider took to its first byte for a request of this size: linear between the two
    captures nearest in size, the smallest or largest capture's own time outside the captured range."""
    by_size: Final = sorted(_matching(provider, route, stream), key=lambda c: c.request_tokens_estimate)
    below: Final = tuple(c for c in by_size if c.request_tokens_estimate <= prompt_tokens)
    above: Final = tuple(c for c in by_size if c.request_tokens_estimate >= prompt_tokens)
    if not below:
        return by_size[0].ttfb_s
    if not above:
        return by_size[-1].ttfb_s
    low, high = below[-1], above[0]
    if high.request_tokens_estimate == low.request_tokens_estimate:
        return low.ttfb_s
    fraction: Final = (prompt_tokens - low.request_tokens_estimate) / (
        high.request_tokens_estimate - low.request_tokens_estimate
    )
    return low.ttfb_s + fraction * (high.ttfb_s - low.ttfb_s)


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


async def paced(capture: Capture, seed: str, scale: float, first_byte_s: float) -> AsyncIterator[bytes]:
    """The captured chunks, the first at `first_byte_s`, the rest at their captured spacing times `scale`."""
    loop: Final = asyncio.get_running_loop()
    start: Final = loop.time()
    shift: Final = first_byte_s - capture.ttfb_s * scale
    for chunk in capture.chunks:
        if (wait := start + shift + chunk.t * scale - loop.time()) > 0:
            await asyncio.sleep(wait)
        yield chunk_bytes(capture, chunk, seed)
