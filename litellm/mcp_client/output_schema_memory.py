from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

from mcp.types import Tool


@dataclass(frozen=True, slots=True)
class _Remembered:
    tools: tuple[Tool, ...]
    expires_at: float


def _schema_only(tool: Tool) -> Tool:
    return Tool(name=tool.name, input_schema={"type": "object"}, output_schema=tool.output_schema)


class OutputSchemaMemory:
    def __init__(
        self,
        *,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries: Final = max_entries
        self._clock: Final = clock
        self._entries: dict[str, _Remembered] = {}  # mutable-ok: bounded process-local store

    def recall(self, key: str) -> tuple[Tool, ...]:
        entry: Final = self._entries.get(key)
        if entry is None:
            return ()
        if entry.expires_at <= self._clock():
            self._entries.pop(key, None)
            return ()
        return entry.tools

    def remember(self, key: str, tools: Iterable[Tool], *, ttl_seconds: float) -> None:
        kept: Final = {tool.name: _schema_only(tool) for tool in tools}
        self._entries.pop(key, None)
        while len(self._entries) >= self._max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = _Remembered(tools=tuple(kept.values()), expires_at=self._clock() + ttl_seconds)

    def forget(self, key: str) -> None:
        self._entries.pop(key, None)
