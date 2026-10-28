"""
In-memory buffer for tool registry upserts, aggregated by tool_name so one flush carries one row per
tool with the number of times it was invoked since the previous flush.
"""

from collections.abc import Iterable
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ToolDiscoveryQueueItem

_ITEM: Final = TypeAdapter(ToolDiscoveryQueueItem)


def _merged(existing: ToolDiscoveryQueueItem | None, item: ToolDiscoveryQueueItem) -> ToolDiscoveryQueueItem:
    """The first sighting keeps its origin and caller fields, later sightings only add to call_count."""
    increment: Final = item.get("call_count", 1)
    if existing is None:
        return _ITEM.validate_python({**item, "call_count": increment})
    return _ITEM.validate_python({**existing, "call_count": existing.get("call_count", 1) + increment})


def aggregate_tool_discoveries(items: Iterable[ToolDiscoveryQueueItem]) -> tuple[ToolDiscoveryQueueItem, ...]:
    """Collapse items to one per tool_name, summing call_count, in first-seen order."""
    by_name: Final[dict[str, ToolDiscoveryQueueItem]] = {}  # mutable-ok: single pass fold keyed by tool_name
    for item in items:
        if not (tool_name := item.get("tool_name", "")):
            continue
        by_name[tool_name] = _merged(by_name.get(tool_name), item)
    return tuple(by_name.values())


class ToolDiscoveryQueue:
    """In-memory buffer for tool registry upserts, one pending row per tool_name with a running call_count."""

    def __init__(self) -> None:
        self._pending: dict[str, ToolDiscoveryQueueItem] = {}  # mutable-ok: folded per add_update, cleared on flush

    def add_update(self, item: ToolDiscoveryQueueItem) -> None:
        """Count one more invocation of item's tool_name, keeping the first sighting's origin and caller fields."""
        tool_name: Final = item.get("tool_name", "")
        if not tool_name:
            return
        self._pending[tool_name] = _merged(self._pending.get(tool_name), item)
        verbose_proxy_logger.debug(
            "ToolDiscoveryQueue: tool %s call_count=%s (origin=%s)",
            tool_name,
            self._pending[tool_name].get("call_count"),
            item.get("origin"),
        )

    def flush(self) -> list[ToolDiscoveryQueueItem]:
        """Return and clear all pending rows."""
        items, self._pending = list(self._pending.values()), {}
        return items
