"""
Split a decoded Server-Sent Events body into lines the way the SSE spec does.

httpx's `iter_lines()` / `aiter_lines()` follow `str.splitlines()`, which also breaks on U+0085, U+2028,
U+2029, \\x0b, \\x0c and \\x1c-\\x1e. JSON leaves U+0085, U+2028 and U+2029 unescaped inside strings, so a
`data:` line carrying one of them gets cut in half. SSE lines end only at CRLF, LF or CR
(https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream)
"""

import io
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from typing import Final

_SSE_LINE_BREAK: Final = re.compile(r"\r\n|\r|\n")


class _SSELineSplitter:
    def __init__(self) -> None:
        self._partial_line: io.StringIO = io.StringIO()
        self._after_cr: bool = False

    def feed(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        fresh: Final = text[1:] if self._after_cr and text.startswith("\n") else text
        self._after_cr = fresh.endswith("\r")
        *ended, tail = fresh.split("\n") if "\r" not in fresh else _SSE_LINE_BREAK.split(fresh)
        if not ended:
            _ = self._partial_line.write(tail)
            return ()
        first: Final = self._partial_line.getvalue() + ended[0]
        self._partial_line = io.StringIO()
        _ = self._partial_line.write(tail)
        return (first, *ended[1:])

    def flush(self) -> tuple[str, ...]:
        rest: Final = self._partial_line.getvalue()
        self._partial_line = io.StringIO()
        return (rest,) if rest else ()


def iter_sse_lines(text_chunks: Iterable[str]) -> Iterator[str]:
    splitter: Final = _SSELineSplitter()
    for text in text_chunks:
        yield from splitter.feed(text)
    yield from splitter.flush()


def split_sse_lines(text: str) -> tuple[str, ...]:
    return tuple(iter_sse_lines((text,)))


async def aiter_sse_lines(text_chunks: AsyncIterable[str]) -> AsyncIterator[str]:
    splitter: Final = _SSELineSplitter()
    async for text in text_chunks:
        for line in splitter.feed(text):
            yield line
    for line in splitter.flush():
        yield line
