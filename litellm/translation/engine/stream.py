"""The ONE stream accumulator: provider wire lines -> IR events -> chunks.

Async-first: ``chunk_stream`` is the production shape (an async line stream
in, OpenAI chunk bodies out); ``fold_lines`` is the synchronous fold over a
recorded stream that the differential tests replay, and ``fold_events`` is
the same fold pinned at the parsed-event seam for providers whose wire
framing is botocore plumbing (bedrock). All compose an injected provider
parser with the inbound chunk fold, so adding a provider or an inbound schema
never touches this file; the optional ``start`` state carries the request
model and chunk dialect for providers whose stream never names the model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from typing import TypeVar

from expression import Error, Ok, Result
from expression.collections import Block

from ..errors import TranslationError
from ..inbound.openai_chat.stream import StreamState, initial_state, step
from ..ir import Body, PlainJson, StreamEvent

ParseLine = Callable[[str], Result[StreamEvent | None, TranslationError]]
ParseEvent = Callable[[PlainJson], Result[StreamEvent | None, TranslationError]]

_TWire = TypeVar("_TWire")


def _fold(
    items: Iterable[_TWire],
    parse: Callable[[_TWire], Result[StreamEvent | None, TranslationError]],
    start: StreamState | None,
) -> Result[Block[Body], TranslationError]:
    state = start if start is not None else initial_state()
    chunks: list[Body] = []
    for item in items:
        match parse(item):
            case Result(tag="ok", ok=event):
                if event is None:
                    continue
                state, emitted = step(state, event)
                chunks.extend(emitted)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(chunks))


def fold_lines(
    lines: Iterable[str], parse_line: ParseLine, start: StreamState | None = None
) -> Result[Block[Body], TranslationError]:
    return _fold(lines, parse_line, start)


def fold_events(
    events: Iterable[PlainJson],
    parse_event: ParseEvent,
    start: StreamState | None = None,
) -> Result[Block[Body], TranslationError]:
    """Recorded parsed events (the characterization corpus seam) -> chunks."""
    return _fold(events, parse_event, start)


async def chunk_stream(
    lines: AsyncIterator[str],
    parse_line: ParseLine,
    start: StreamState | None = None,
) -> AsyncIterator[Result[Body, TranslationError]]:
    """Async-first form: yields one Result per outbound chunk; the first
    error ends the stream (the seam surfaces it as a provider error)."""
    state = start if start is not None else initial_state()
    async for line in lines:
        match parse_line(line):
            case Result(tag="ok", ok=event):
                if event is None:
                    continue
                state, emitted = step(state, event)
                for chunk in emitted:
                    yield Ok(chunk)
            case Result(error=err):
                yield Error(err)
                return
