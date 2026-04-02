import asyncio

import pytest

from litellm.proxy.middleware.auto_queue_middleware import (
    ModelQueue,
    _QueueWakeReason,
    _WakeState,
)


def test_wake_next_skips_removed_head_and_wakes_next_valid():
    queue = ModelQueue(max_depth=5)

    first = _WakeState()
    second = _WakeState()
    third = _WakeState()

    queue.add("req-1", first, priority=10)
    queue.add("req-2", second, priority=10)
    queue.add("req-3", third, priority=10)

    # Remove the head entry (req-1, lowest seq)
    queue.remove("req-1")

    # wake_next should skip the removed entry and wake req-2
    assert queue.wake_next() is True
    assert second.is_set is True
    assert second.reason == _QueueWakeReason.TRANSFERRED
    assert first.is_set is False  # removed, not woken by wake_next
    assert third.is_set is False
    assert queue.depth == 1


def test_remove_missing_request_is_noop():
    queue = ModelQueue(max_depth=5)

    state = _WakeState()
    queue.add("req-1", state, priority=10)

    # Remove a non-existent request id
    queue.remove("nonexistent")

    assert queue.depth == 1
    assert state.is_set is False


def test_wake_all_after_lazy_deletes_clears_heap_and_entries():
    queue = ModelQueue(max_depth=5)

    states = [_WakeState() for _ in range(3)]
    queue.add("req-1", states[0], priority=10)
    queue.add("req-2", states[1], priority=10)
    queue.add("req-3", states[2], priority=10)

    # Lazily remove one entry
    queue.remove("req-2")

    # wake_all should set events for remaining entries and clear everything
    queue.wake_all()

    # req-1 and req-3 should be woken
    assert states[0].is_set is True
    assert states[2].is_set is True
    # req-2 was removed, wake_all iterates _entries which no longer has req-2
    assert queue.depth == 0
    assert queue._heap == []
    assert queue._entries == {}


def test_priority_ordering():
    queue = ModelQueue(max_depth=5)

    low = _WakeState()
    high = _WakeState()
    medium = _WakeState()

    queue.add("req-low", low, priority=10)
    queue.add("req-high", high, priority=1)
    queue.add("req-medium", medium, priority=5)

    # wake_next should wake highest priority (lowest number) first
    assert queue.wake_next() is True
    assert high.is_set is True
    assert low.is_set is False
    assert medium.is_set is False

    assert queue.wake_next() is True
    assert medium.is_set is True

    assert queue.wake_next() is True
    assert low.is_set is True

    assert queue.depth == 0


def test_max_depth_exceeded():
    queue = ModelQueue(max_depth=2)

    queue.add("req-1", _WakeState(), priority=10)
    queue.add("req-2", _WakeState(), priority=10)

    assert queue.is_full is True
    assert queue.depth == 2


def test_fifo_within_same_priority():
    queue = ModelQueue(max_depth=5)

    first = _WakeState()
    second = _WakeState()

    queue.add("req-1", first, priority=10)
    queue.add("req-2", second, priority=10)

    # wake_next should wake the first entry added (FIFO tie-break)
    assert queue.wake_next() is True
    assert first.is_set is True
    assert second.is_set is False

    assert queue.depth == 1
