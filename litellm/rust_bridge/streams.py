from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.rust_bridge import lifecycle
from litellm.rust_bridge.lifecycle import Execution

Await: Final = lifecycle.Await
Complete: Final = lifecycle.Complete
Open: Final = lifecycle.Open
Yield: Final = lifecycle.Yield

_HEADERS: Final = TypeAdapter(Mapping[str, object])


async def drive(execution: Execution) -> object:
    return await lifecycle.drive(execution, Stream)


class Stream(AsyncIterator[object]):
    def __init__(self, execution: Execution, hidden_params: object = None) -> None:
        self._stream: Final = lifecycle.Stream(execution, hidden_params)
        self._hidden_params: dict[str, object] = dict(_headers(hidden_params))  # mutable-ok: header writers mutate it

    def __aiter__(self) -> Stream:
        return self

    async def __anext__(self) -> object:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        await self._stream.aclose()


class SyncStream(Iterator[object]):
    def __init__(self, execution: Execution, hidden_params: object = None) -> None:
        self._stream: Final = lifecycle.SyncStream(execution, hidden_params)
        self._hidden_params: dict[str, object] = dict(_headers(hidden_params))  # mutable-ok: header writers mutate it

    def __iter__(self) -> SyncStream:
        return self

    def __next__(self) -> object:
        return next(self._stream)

    def close(self) -> None:
        self._stream.close()


def _headers(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    return _HEADERS.validate_python(value)
