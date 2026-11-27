import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.presidio import _coalesce_first_sse_frame

LEADING_NON_OBJECT_READS: Final = tuple(b"data: 5\n\n" for _ in range(6000)) + (b'data: {"type": "ping"}\n\n',)
LARGE_FIRST_FRAME: Final = b'data: {"type": "message_start", "pad": "' + b"x" * 48_000 + b'"}\n\n'
LARGE_FIRST_FRAME_READS: Final = tuple(LARGE_FIRST_FRAME[i : i + 16] for i in range(0, len(LARGE_FIRST_FRAME), 16))


async def _first_sse_frame(reads: Sequence[bytes]) -> list[object]:
    async def upstream() -> AsyncIterator[object]:
        for chunk in reads:
            yield chunk

    return [chunk async for chunk in _coalesce_first_sse_frame(upstream())]


@pytest.mark.benchmark
def test_presidio_first_frame_wait_over_many_leading_non_object_reads() -> None:
    assert asyncio.run(_first_sse_frame(LEADING_NON_OBJECT_READS)) == [b"".join(LEADING_NON_OBJECT_READS)]


@pytest.mark.benchmark
def test_presidio_first_frame_wait_over_a_frame_split_into_many_reads() -> None:
    assert asyncio.run(_first_sse_frame(LARGE_FIRST_FRAME_READS)) == [LARGE_FIRST_FRAME]
