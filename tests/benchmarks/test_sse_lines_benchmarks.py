from collections.abc import Sequence
from typing import Final

import pytest

from litellm.llms.custom_httpx.sse_lines import iter_sse_lines

EVENT: Final = 'data: {"choices":[{"delta":{"content":"hello world"}}]}'


def _chunks(line_break: str) -> Sequence[str]:
    body: Final = "".join(f"{EVENT}{line_break}{line_break}" for _ in range(2000))
    return tuple(body[i : i + 512] for i in range(0, len(body), 512))


LF_CHUNKS: Final = _chunks("\n")
CRLF_CHUNKS: Final = _chunks("\r\n")


@pytest.mark.benchmark
def test_iter_sse_lines_lf_stream() -> None:
    assert sum(1 for _ in iter_sse_lines(LF_CHUNKS)) == 4000


@pytest.mark.benchmark
def test_iter_sse_lines_crlf_stream() -> None:
    assert sum(1 for _ in iter_sse_lines(CRLF_CHUNKS)) == 4000
