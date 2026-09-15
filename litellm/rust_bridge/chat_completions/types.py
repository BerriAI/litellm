from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol


class RustChatCompletions(Protocol):
    def __call__(
        self,
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> object:
        raise NotImplementedError


class RustAchatCompletions(Protocol):
    def __call__(
        self,
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[object]:
        raise NotImplementedError
