from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RustTokenizer:
    kind: str | None
    encoding: str
    disabled: bool
    legacy_accounting: bool


class RustTokenCounter(Protocol):
    def __call__(
        self,
        body: bytes,
        kind: str | None,
        encoding: str,
        disabled: bool,
        legacy_accounting: bool,
        resource_loader: Callable[[str], str],
    ) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True)
class InputTokenCount:
    model: str | None
    input_tokens: int
