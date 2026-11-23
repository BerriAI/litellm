from typing import Final

import pytest
from starlette.requests import Request
from starlette.types import Message

from litellm.proxy.mcp_server.events.webhooks.receiver import MAX_BODY_BYTES, _read_body


@pytest.mark.parametrize("extra_byte", (False, True), ids=("at-limit", "oversized-body"))
@pytest.mark.asyncio
async def test_body_limit_counts_bytes_across_chunks(extra_byte: bool) -> None:
    first: Final = b"x" * (MAX_BODY_BYTES - 1)
    last: Final = b"yz" if extra_byte else b"y"
    chunks: Final = iter((first, last))

    async def receive() -> Message:
        chunk: Final = next(chunks)
        return {"type": "http.request", "body": chunk, "more_body": chunk is first}

    request: Final = Request({"type": "http"}, receive=receive)
    assert await _read_body(request) == (None if extra_byte else first + last)
