"""
In-memory buffer for tool registry upserts.

Unlike SpendUpdateQueue (which aggregates increments), ToolDiscoveryQueue
uses set-deduplication: each unique tool_name is only queued once per pod
lifetime, so DB upserts stop entirely after warmup.
"""

from typing import List, Set

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ToolDiscoveryQueueItem


class ToolDiscoveryQueue:
    """
    In-memory buffer for tool registry upserts.

    Deduplicates by tool_name — once a tool has been seen in this process,
    it is never enqueued again (the DB row already exists or will be created
    during the current flush cycle).
    """

    def __init__(self) -> None:
        self._seen_tool_names: Set[str] = set()
        self._pending: List[ToolDiscoveryQueueItem] = []

    def add_update(self, item: ToolDiscoveryQueueItem) -> None:
        """Enqueue a tool discovery item if tool_name has not been seen before."""
        tool_name = item.get("tool_name", "")
        if not tool_name:
            return
        if tool_name in self._seen_tool_names:
            verbose_proxy_logger.debug(
                "ToolDiscoveryQueue: skipping already-seen tool %s", tool_name
            )
            return
        self._seen_tool_names.add(tool_name)
        self._pending.append(item)
        verbose_proxy_logger.debug(
            "ToolDiscoveryQueue: queued new tool %s (origin=%s)",
            tool_name,
            item.get("origin"),
        )

    def flush(self) -> List[ToolDiscoveryQueueItem]:
        """Return and clear all pending items."""
        items, self._pending = self._pending, []
        return items
