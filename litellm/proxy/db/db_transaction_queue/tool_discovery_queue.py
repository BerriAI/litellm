"""
In-memory buffer for tool registry upserts.

Unlike SpendUpdateQueue (which aggregates increments), ToolDiscoveryQueue
uses set-deduplication: each unique tool_name is only queued once per flush
cycle (~30s). The seen-set is cleared on every flush so that call_count
increments in subsequent cycles rather than stopping after the first flush.
"""

from typing import List, Set

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ToolCallLogItem, ToolDiscoveryQueueItem


class ToolDiscoveryQueue:
    """
    In-memory buffer for tool registry upserts and call logs.

    Registry items deduplicate by tool_name within each flush cycle: a tool is
    only queued once per ~30s batch, so call_count increments once per flush
    cycle the tool appears in. The seen-set is cleared on flush so subsequent
    batches can re-count the same tool.

    Call log items are NOT deduplicated — one entry is written per invocation.
    They reference LiteLLM_SpendLogs via request_id so callers can look up args.
    """

    def __init__(self) -> None:
        self._seen_tool_names: Set[str] = set()
        self._pending: List[ToolDiscoveryQueueItem] = []
        self._call_log_pending: List[ToolCallLogItem] = []

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

    def add_call_log(self, item: ToolCallLogItem) -> None:
        """Enqueue one call log entry (not deduplicated)."""
        if not item.get("tool_name"):
            return
        self._call_log_pending.append(item)

    def flush(self) -> List[ToolDiscoveryQueueItem]:
        """Return and clear all pending registry items. Resets seen-set so the
        next flush cycle can re-count the same tools."""
        items, self._pending = self._pending, []
        self._seen_tool_names.clear()
        return items

    def flush_call_logs(self) -> List[ToolCallLogItem]:
        """Return and clear all pending call log items."""
        items, self._call_log_pending = self._call_log_pending, []
        return items
