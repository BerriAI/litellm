from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from typing import Final, cast  # noqa: TID251  # trusts the native stub at the binding boundary

from pydantic import TypeAdapter

from litellm.rust_bridge import lifecycle
from litellm.rust_bridge.lifecycle import Execution
from litellm.rust_bridge.public_call import NativeCall

_HEADERS: Final = TypeAdapter(Mapping[str, object])


def synchronous(native: Callable[[NativeCall], object]) -> Callable[[NativeCall], object]:
    return lambda call: lifecycle.run(_execution(native(call)), SyncStream)


def asynchronous(native: Callable[[NativeCall], object]) -> Callable[[NativeCall], Awaitable[object]]:
    return lambda call: lifecycle.drive(_execution(native(call)), AsyncStream)


def _execution(value: object) -> Execution:
    return cast("Execution", value)  # cast-ok: native entrypoints return Execution per _native.pyi


class AsyncStream(AsyncIterator[object]):
    def __init__(self, execution: Execution, head: object) -> None:
        self._stream: Final = lifecycle.AsyncStream(execution, head)
        self._hidden_params: dict[str, object] = dict(_headers(head))  # mutable-ok: header writers mutate it

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> object:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        await self._stream.aclose()


class SyncStream(Iterator[object]):
    def __init__(self, execution: Execution, head: object) -> None:
        self._stream: Final = lifecycle.SyncStream(execution, head)
        self._hidden_params: dict[str, object] = dict(_headers(head))  # mutable-ok: header writers mutate it

    def __iter__(self) -> SyncStream:
        return self

    def __next__(self) -> object:
        return next(self._stream)

    def close(self) -> None:
        self._stream.close()


def _headers(head: object) -> Mapping[str, object]:
    return _HEADERS.validate_python(head)
