from collections.abc import AsyncIterator, Sequence
from typing import Final

import pytest

from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines, iter_sse_lines

CASES: Final = (
    pytest.param(("data: a b\r\n\r\n",), ("data: a b", ""), id="u2028-stays-inside-the-line"),
    pytest.param(
        ("data: \u0085 \x0b\x0c\x1c\x1d\x1e\n",),
        ("data: \u0085 \x0b\x0c\x1c\x1d\x1e",),
        id="no-splitlines-only-separator-breaks-a-line",
    ),
    pytest.param(("a\r\nb\nc\rd\n",), ("a", "b", "c", "d"), id="crlf-lf-and-cr-each-end-a-line"),
    pytest.param(("a\r", "\nb\n"), ("a", "b"), id="crlf-split-across-chunks-is-one-break"),
    pytest.param(("a\r", "\n\r\n"), ("a", ""), id="crlf-split-then-blank-line"),
    pytest.param(("a\r", "", "\nb\n"), ("a", "b"), id="empty-chunk-between-cr-and-lf"),
    pytest.param(("a\r", "\r"), ("a", ""), id="cr-then-cr-is-two-breaks"),
    pytest.param(("da", "ta: x", "y\n"), ("data: xy",), id="line-spanning-chunks-is-joined"),
    pytest.param(("data: x",), ("data: x",), id="unterminated-last-line-is-kept"),
    pytest.param(("",), (), id="empty-body"),
)


async def _async_chunks(chunks: Sequence[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


@pytest.mark.parametrize(("chunks", "expected"), CASES)
def test_iter_sse_lines_splits_only_on_cr_lf_and_crlf(chunks: Sequence[str], expected: Sequence[str]) -> None:
    assert tuple(iter_sse_lines(chunks)) == tuple(expected)


@pytest.mark.parametrize(("chunks", "expected"), CASES)
async def test_aiter_sse_lines_splits_only_on_cr_lf_and_crlf(chunks: Sequence[str], expected: Sequence[str]) -> None:
    assert tuple([line async for line in aiter_sse_lines(_async_chunks(chunks))]) == tuple(expected)
