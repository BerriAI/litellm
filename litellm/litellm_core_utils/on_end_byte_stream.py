from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Final, TypeAlias

import httpx


@dataclass(frozen=True, slots=True)
class StreamCompleted:
    pass


@dataclass(frozen=True, slots=True)
class StreamClosedEarly:
    pass


@dataclass(frozen=True, slots=True)
class StreamFailed:
    error: Exception


StreamEnd: TypeAlias = StreamCompleted | StreamClosedEarly | StreamFailed


class OnEndAsyncByteStream(httpx.AsyncByteStream):
    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        map_error: Callable[[httpx.TransportError], Exception],
        on_end: Callable[[StreamEnd], Awaitable[None]],
    ) -> None:
        self._stream: Final = stream
        self._map_error: Final = map_error
        self._on_end: Final = on_end
        self._ended = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except httpx.TransportError as error:
            mapped: Final = self._map_error(error)
            await self._end(StreamFailed(mapped))
            raise mapped from error
        except Exception as error:
            await self._end(StreamFailed(error))
            raise
        await self._end(StreamCompleted())

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            await self._end(StreamClosedEarly())

    async def _end(self, end: StreamEnd) -> None:
        if self._ended:
            return
        self._ended = True
        await self._on_end(end)


class OnEndSyncByteStream(httpx.SyncByteStream):
    def __init__(
        self,
        stream: httpx.SyncByteStream,
        map_error: Callable[[httpx.TransportError], Exception],
        on_end: Callable[[StreamEnd], None],
    ) -> None:
        self._stream: Final = stream
        self._map_error: Final = map_error
        self._on_end: Final = on_end
        self._ended = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._stream
        except httpx.TransportError as error:
            mapped: Final = self._map_error(error)
            self._end(StreamFailed(mapped))
            raise mapped from error
        except Exception as error:
            self._end(StreamFailed(error))
            raise
        self._end(StreamCompleted())

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._end(StreamClosedEarly())

    def _end(self, end: StreamEnd) -> None:
        if self._ended:
            return
        self._ended = True
        self._on_end(end)
