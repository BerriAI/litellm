"""The ONE stream accumulator: provider wire lines -> IR events -> chunks.

Async-first: ``chunk_stream`` is the production shape (an async line stream
in, OpenAI chunk bodies out); ``fold_lines`` is the synchronous fold over a
recorded stream that the differential tests replay. Both compose an injected
provider line parser with the inbound chunk fold, so adding a provider or an
inbound schema never touches this file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable

from expression import Error, Ok, Result
from expression.collections import Block

from ..errors import TranslationError
from ..inbound.openai_chat.stream import StreamState, initial_state, step
from ..ir import Body, StreamEvent

ParseLine = Callable[[str], Result[StreamEvent | None, TranslationError]]


def fold_lines(
    lines: Iterable[str], parse_line: ParseLine
) -> Result[Block[Body], TranslationError]:
    state: StreamState = initial_state()
    chunks: list[Body] = []
    for line in lines:
        match parse_line(line):
            case Result(tag="ok", ok=event):
                if event is None:
                    continue
                state, emitted = step(state, event)
                chunks.extend(emitted)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(chunks))


async def chunk_stream(
    lines: AsyncIterator[str], parse_line: ParseLine
) -> AsyncIterator[Result[Body, TranslationError]]:
    """Async-first form: yields one Result per outbound chunk; the first
    error ends the stream (the seam surfaces it as a provider error)."""
    state: StreamState = initial_state()
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
