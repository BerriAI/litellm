from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Final

import httpx


class OnEndAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, on_end: Callable[[Exception | None], Awaitable[None]]) -> None:
        self._stream: Final = stream
        self._on_end: Final = on_end
        self._ended = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except Exception as error:
            await self._end(error)
            raise
        await self._end(None)

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            await self._end(None)

    async def _end(self, error: Exception | None) -> None:
        if self._ended:
            return
        self._ended = True
        await self._on_end(error)


class OnEndSyncByteStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, on_end: Callable[[Exception | None], None]) -> None:
        self._stream: Final = stream
        self._on_end: Final = on_end
        self._ended = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._stream
        except Exception as error:
            self._end(error)
            raise
        self._end(None)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._end(None)

    def _end(self, error: Exception | None) -> None:
        if self._ended:
            return
        self._ended = True
        self._on_end(error)
