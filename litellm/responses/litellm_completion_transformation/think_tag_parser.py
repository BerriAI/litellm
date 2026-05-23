"""Inline ``<think>...</think>`` extraction for chat-completions content streams.

Used by the Responses API bridge when a chat-completions provider emits its
reasoning as ``<think>...</think>`` blocks embedded in the regular ``content``
field instead of populating the canonical ``reasoning_content`` field. Known
offenders include sglang's ``--reasoning-parser minimax-append-think``
(broken per sgl-project/sglang#15508), some MiniMax / DeepSeek-R1 deployments,
Fireworks AI streaming, and Kimi K2 via NVIDIA NIM.

The parser feeds character chunks (as they arrive from upstream stream
deltas) through a small state machine and yields a list of typed segments
the caller can map to OpenAI Responses API stream events:

    ("text", "...")              -> response.output_text.delta
    ("reasoning_open", None)     -> response.output_item.added (type=reasoning)
    ("reasoning", "...")         -> response.reasoning_summary_text.delta
    ("reasoning_close", None)    -> response.output_item.done (type=reasoning)

Robustness contract:

- Tags may be split across feeds (e.g. ``<thi`` then ``nk>``); the parser
  buffers only the minimum trailing bytes that are themselves a valid prefix
  of the next-expected tag.
- A ``<think>`` without a closing ``</think>`` flushes the remaining buffer
  as reasoning when ``flush()`` is called at stream end.
- A buffer ending mid-open-tag (e.g. ``"hello <thi"``) flushes as plain
  text - partial open tags do not transition state.
- Unmatched ``</think>`` outside of an open block is plain text.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"

Segment = Tuple[str, Optional[str]]


def _partial_prefix_len(buf: str, target: str) -> int:
    """Length of the longest suffix of ``buf`` that is a strict prefix of ``target``.

    Returns 0 when no overlap. Used to hold only the bytes that *could* become
    a split tag on the next feed; everything earlier is safe to emit.
    """
    max_check = min(len(buf), len(target) - 1)
    for k in range(max_check, 0, -1):
        if buf.endswith(target[:k]):
            return k
    return 0


class ThinkTagParser:
    def __init__(self) -> None:
        self._buf: str = ""
        self._in_think: bool = False

    def feed(self, chunk: str) -> List[Segment]:
        """Process a content chunk, return ordered segments to emit."""
        if not chunk:
            return []
        self._buf += chunk
        return self._drain(final=False)

    def flush(self) -> List[Segment]:
        """Drain anything still buffered at stream end.

        If we are still inside a ``<think>`` block, the unclosed remainder is
        emitted as reasoning (and a synthetic close event is fired) so the
        stream stays well-formed even if the model truncated.
        """
        return self._drain(final=True)

    def _drain(self, final: bool) -> List[Segment]:
        out: List[Segment] = []
        while True:
            if self._in_think:
                idx = self._buf.find(CLOSE_TAG)
                if idx >= 0:
                    if idx > 0:
                        out.append(("reasoning", self._buf[:idx]))
                    self._buf = self._buf[idx + len(CLOSE_TAG) :]
                    self._in_think = False
                    out.append(("reasoning_close", None))
                    continue
                if final:
                    if self._buf:
                        out.append(("reasoning", self._buf))
                        self._buf = ""
                    out.append(("reasoning_close", None))
                    return out
                hold = _partial_prefix_len(self._buf, CLOSE_TAG)
                if len(self._buf) > hold:
                    emit = self._buf[: len(self._buf) - hold]
                    self._buf = self._buf[len(self._buf) - hold :]
                    if emit:
                        out.append(("reasoning", emit))
                return out
            else:
                idx = self._buf.find(OPEN_TAG)
                if idx >= 0:
                    if idx > 0:
                        out.append(("text", self._buf[:idx]))
                    self._buf = self._buf[idx + len(OPEN_TAG) :]
                    self._in_think = True
                    out.append(("reasoning_open", None))
                    continue
                if final:
                    if self._buf:
                        out.append(("text", self._buf))
                        self._buf = ""
                    return out
                hold = _partial_prefix_len(self._buf, OPEN_TAG)
                if len(self._buf) > hold:
                    emit = self._buf[: len(self._buf) - hold]
                    self._buf = self._buf[len(self._buf) - hold :]
                    if emit:
                        out.append(("text", emit))
                return out


def extract_think_from_text(content: str) -> Tuple[str, str]:
    """Non-streaming helper: split an entire content string into reasoning
    (concatenated across all ``<think>`` blocks) and remaining message text.

    Used by ``transform_chat_completion_response_to_responses_api_response``
    when the bridge processes a non-streamed completion.
    """
    p = ThinkTagParser()
    segments = p.feed(content) + p.flush()
    reasoning_parts: List[str] = []
    text_parts: List[str] = []
    for kind, value in segments:
        if kind == "reasoning" and value:
            reasoning_parts.append(value)
        elif kind == "text" and value:
            text_parts.append(value)
    return "".join(reasoning_parts), "".join(text_parts)
