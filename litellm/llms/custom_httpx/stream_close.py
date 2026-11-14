from collections.abc import AsyncGenerator, AsyncIterator
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


async def aclose_if_supported(stream: object) -> None:
    if isinstance(stream, SupportsAclose):
        await stream.aclose()


async def _yield_then_close(chunks: AsyncIterator[bytes], response: httpx.Response) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in chunks:
            yield chunk
    finally:
        await response.aclose()


def aiter_bytes_then_close(response: httpx.Response, *, chunk_size: int | None = None) -> AsyncGenerator[bytes, None]:
    return _yield_then_close(response.aiter_bytes(chunk_size=chunk_size), response)
