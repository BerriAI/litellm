"""
Stateful Presidio PII unmasking for Anthropic ``/v1/messages`` SSE streams.

Claude splits tokens like ``<EMAIL_ADDRESS_2>`` across deltas, and byte chunks
need not end on an event boundary, so unmasking each chunk on its own misses
them. This reassembles events, holds back a trailing partial token per content
block until it completes, and flushes held text before ``content_block_stop``.

``thinking_delta`` is left alone: thinking blocks are signed, and rewriting
them would invalidate the signature when the client sends them back.
"""

import codecs
import json
import re
from collections.abc import Mapping
from functools import reduce
from typing import Final

_EVENT_SPLIT: Final = re.compile(r"(\r?\n\r?\n)")

# delta.type -> (field holding the text, whether values land inside a JSON string)
_UNMASKABLE_DELTAS: Final[Mapping[str, tuple[str, bool]]] = {
    "text_delta": ("text", False),
    "input_json_delta": ("partial_json", True),
}

_FLUSH_ALL_EVENTS: Final = frozenset({"message_delta", "message_stop", "error"})

_MAX_BUFFERED_CHARS: Final = 1_000_000


class AnthropicSSEUnmasker:
    """Feed raw SSE byte chunks in order, then call ``flush`` once at end of stream."""

    def __init__(self, pii_tokens: Mapping[str, str]) -> None:
        self._tokens: Final = dict(pii_tokens)
        self._max_token_len: Final = max((len(t) for t in self._tokens), default=0)
        self._decoder: Final = codecs.getincrementaldecoder("utf-8")(errors="surrogateescape")
        self._buffer: str = ""
        self._held: dict[int, tuple[str, str]] = {}  # mutable-ok: per-block stream state, index -> (delta type, text)

    def feed(self, chunk: bytes) -> bytes:
        pieces: Final = _EVENT_SPLIT.split(self._buffer + self._decoder.decode(chunk))
        rest: Final = pieces[-1]
        processed: Final = "".join(
            tuple(self._process_event(event, terminator) for event, terminator in zip(pieces[:-1:2], pieces[1::2]))
        )
        overflow: Final = len(rest) >= _MAX_BUFFERED_CHARS
        self._buffer = "" if overflow else rest
        return _encode(processed + (rest if overflow else ""))

    def flush(self) -> bytes:
        held: Final = self._flush_all_held()
        tail: Final = self._buffer + self._decoder.decode(b"", final=True)
        self._buffer = ""
        return _encode(held + tail)

    def _process_event(self, event_text: str, terminator: str) -> str:
        original: Final = event_text + terminator
        lines: Final = tuple(event_text.split("\n"))
        data_indexes: Final = tuple(i for i, line in enumerate(lines) if line.startswith("data:"))
        if len(data_indexes) != 1:
            return original
        data_index: Final = data_indexes[0]
        line_ending: Final = "\r" if lines[data_index].endswith("\r") else ""
        try:
            event: Final[object] = json.loads(lines[data_index][5:].rstrip("\r").removeprefix(" "))
        except ValueError:
            return original
        if not isinstance(event, dict):
            return original

        match event.get("type"):
            case "content_block_stop" if isinstance(event.get("index"), int):
                return self._flush_held(event["index"]) + original
            case event_type if event_type in _FLUSH_ALL_EVENTS:
                return self._flush_all_held() + original
            case "content_block_delta":
                rewritten: Final = self._unmask_delta(event)
                if rewritten is None:
                    return original
                data_line: Final = "data: " + json.dumps(rewritten, ensure_ascii=False) + line_ending
                return "\n".join((*lines[:data_index], data_line, *lines[data_index + 1 :])) + terminator
            case _:
                return original

    def _unmask_delta(self, event: dict[str, object]) -> dict[str, object] | None:
        """Return the rewritten event, or None to pass the original through untouched."""
        index: Final = event.get("index")
        delta: Final = event.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            return None
        delta_type: Final = delta.get("type")
        spec: Final = _UNMASKABLE_DELTAS.get(delta_type) if isinstance(delta_type, str) else None
        if spec is None or not isinstance(delta.get(spec[0]), str):
            return None
        field, escape = spec
        text: Final[str] = delta[field]

        _, previously_held = self._held.pop(index, ("", ""))
        emit, hold = self._split_partial_token(previously_held + text)
        if hold:
            self._held[index] = (delta_type, hold)
        unmasked: Final = self._replace_tokens(emit, escape)
        if unmasked == text:
            return None
        return {**event, "delta": {**delta, field: unmasked}}

    def _split_partial_token(self, text: str) -> tuple[str, str]:
        """Split off a trailing prefix of a known token, e.g. ``"<EMAIL_ADD"``."""
        start: Final = text.rfind("<")
        suffix: Final = text[start:] if start != -1 else ""
        is_partial_token: Final = (
            bool(suffix)
            and ">" not in suffix
            and len(suffix) < self._max_token_len
            and any(token.startswith(suffix) for token in self._tokens)
        )
        return (text[:start], suffix) if is_partial_token else (text, "")

    def _replace_tokens(self, text: str, escape: bool) -> str:
        return reduce(
            lambda acc, item: acc.replace(item[0], _render(item[1], escape)),
            self._tokens.items(),
            text,
        )

    def _flush_all_held(self) -> str:
        return "".join(tuple(self._flush_held(index) for index in sorted(self._held)))

    def _flush_held(self, index: int) -> str:
        held: Final = self._held.pop(index, None)
        if held is None:
            return ""
        delta_type, text = held
        field, escape = _UNMASKABLE_DELTAS[delta_type]
        # Block ended mid-token (e.g. max_tokens): same truncation rule as
        # _OPTIONAL_PresidioPIIMasking._unmask_pii_text.
        truncated_value: Final = next(
            (
                value
                for token, value in self._tokens.items()
                if token.startswith(text) and len(text) >= min(20, len(token) // 2)
            ),
            None,
        )
        final_text: Final = text if truncated_value is None else _render(truncated_value, escape)
        event: Final = {"type": "content_block_delta", "index": index, "delta": {"type": delta_type, field: final_text}}
        return "event: content_block_delta\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _render(value: str, escape: bool) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1] if escape else value


def _encode(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")
