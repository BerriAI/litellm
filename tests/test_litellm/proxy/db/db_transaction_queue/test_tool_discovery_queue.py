"""
Unit tests for ToolDiscoveryQueue.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import (
    ToolDiscoveryQueue,
)


@pytest.fixture
def queue():
    return ToolDiscoveryQueue()


def test_add_single_tool(queue):
    queue.add_update({"tool_name": "my_tool", "origin": "user_defined"})
    items = queue.flush()
    assert len(items) == 1
    assert items[0]["tool_name"] == "my_tool"
    assert items[0]["origin"] == "user_defined"


def test_deduplication_same_name(queue):
    """Adding the same tool_name twice should only keep the first."""
    queue.add_update({"tool_name": "tool_a", "origin": "mcp_server"})
    queue.add_update({"tool_name": "tool_a", "origin": "user_defined"})
    items = queue.flush()
    assert len(items) == 1
    assert items[0]["origin"] == "mcp_server"  # first wins


def test_deduplication_different_names(queue):
    queue.add_update({"tool_name": "tool_a"})
    queue.add_update({"tool_name": "tool_b"})
    items = queue.flush()
    assert len(items) == 2
    names = {i["tool_name"] for i in items}
    assert names == {"tool_a", "tool_b"}


def test_flush_clears_pending(queue):
    queue.add_update({"tool_name": "tool_x"})
    items1 = queue.flush()
    assert len(items1) == 1
    items2 = queue.flush()
    assert len(items2) == 0


def test_seen_names_reset_after_flush(queue):
    """Seen-set is cleared on flush so the same tool can re-enter the next cycle."""
    queue.add_update({"tool_name": "tool_a"})
    queue.flush()
    queue.add_update({"tool_name": "tool_a"})  # same tool, new cycle
    items = queue.flush()
    assert len(items) == 1
    assert items[0]["tool_name"] == "tool_a"


def test_empty_tool_name_ignored(queue):
    queue.add_update({"tool_name": ""})
    queue.add_update({"tool_name": None})  # type: ignore[arg-type]
    items = queue.flush()
    assert len(items) == 0


def test_flush_returns_list(queue):
    result = queue.flush()
    assert isinstance(result, list)
