"""
Unit tests for ToolDiscoveryQueue.
"""


import pytest

from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import (
    ToolDiscoveryQueue,
    aggregate_tool_discoveries,
)


@pytest.fixture
def queue():
    return ToolDiscoveryQueue()


def test_a_tool_seen_once_flushes_with_call_count_one(queue):
    queue.add_update({"tool_name": "my_tool", "origin": "user_defined"})
    assert queue.flush() == [{"tool_name": "my_tool", "origin": "user_defined", "call_count": 1}]


def test_repeated_sightings_fold_into_one_row_counting_every_call(queue):
    queue.add_update({"tool_name": "tool_a", "origin": "mcp_server", "created_by": "alice"})
    queue.add_update({"tool_name": "tool_a", "origin": "user_defined", "created_by": "bob"})
    queue.add_update({"tool_name": "tool_a"})
    assert queue.flush() == [{"tool_name": "tool_a", "origin": "mcp_server", "created_by": "alice", "call_count": 3}]


def test_distinct_tools_each_get_their_own_row(queue):
    queue.add_update({"tool_name": "tool_a"})
    queue.add_update({"tool_name": "tool_b"})
    queue.add_update({"tool_name": "tool_a"})
    assert queue.flush() == [
        {"tool_name": "tool_a", "call_count": 2},
        {"tool_name": "tool_b", "call_count": 1},
    ]


def test_flush_clears_pending_and_the_next_cycle_counts_from_zero(queue):
    queue.add_update({"tool_name": "tool_x"})
    queue.add_update({"tool_name": "tool_x"})
    assert queue.flush() == [{"tool_name": "tool_x", "call_count": 2}]
    assert queue.flush() == []
    queue.add_update({"tool_name": "tool_x"})
    assert queue.flush() == [{"tool_name": "tool_x", "call_count": 1}]


def test_empty_tool_name_ignored(queue):
    queue.add_update({"tool_name": ""})
    queue.add_update({"tool_name": None})  # type: ignore[arg-type]
    assert queue.flush() == []


def test_aggregate_sums_call_counts_across_workers_flushes():
    worker_a = [{"tool_name": "get_weather", "origin": "user_defined", "call_count": 3}]
    worker_b = [
        {"tool_name": "get_weather", "origin": "mcp_server", "call_count": 2},
        {"tool_name": "search", "call_count": 1},
    ]
    assert aggregate_tool_discoveries([*worker_a, *worker_b]) == (
        {"tool_name": "get_weather", "origin": "user_defined", "call_count": 5},
        {"tool_name": "search", "call_count": 1},
    )
