"""Regression for #31059.

``ResponsesAPIStreamingIterator`` scheduled the success-logging coroutine via a
bare ``asyncio.create_task(...)`` whose returned task was never referenced. The
event loop keeps only a weak reference to tasks, so an un-referenced logging
task can be garbage-collected before it finishes — silently dropping the
streaming Responses-API spend-log row.

These tests cover the fix: a module-level ``_BACKGROUND_TASKS`` set holds a
strong reference until completion (surviving GC), failures are still surfaced,
and the success-logging call site registers its task with the tracker.
"""

import asyncio
import gc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesWebSocketStreaming,
    _BACKGROUND_TASKS,
    _track_background_task,
)


@pytest.fixture(autouse=True)
def _clear_background_tasks():
    """Keep the module-level set isolated between tests."""
    _BACKGROUND_TASKS.clear()
    yield
    _BACKGROUND_TASKS.clear()


@pytest.mark.asyncio
async def test_track_background_task_holds_strong_ref_then_discards():
    started = asyncio.Event()
    release = asyncio.Event()

    async def _work():
        started.set()
        await release.wait()

    task = asyncio.create_task(_work())
    _track_background_task(task, task_name="unit task")

    await started.wait()
    # Strong reference retained while the task runs.
    assert task in _BACKGROUND_TASKS

    release.set()
    await task
    # Discarded once complete so the set does not grow unbounded.
    assert task not in _BACKGROUND_TASKS


@pytest.mark.asyncio
async def test_tracked_task_survives_gc_while_pending():
    """The actual bug: without a strong reference, the task can be collected
    mid-flight. The tracker must keep it alive across a GC cycle."""
    done = asyncio.Event()
    ran = {"value": False}

    async def _work():
        await asyncio.sleep(0.05)
        ran["value"] = True
        done.set()

    _track_background_task(asyncio.create_task(_work()), task_name="gc-sensitive task")
    # Drop every local reference the caller might have held, then force GC.
    gc.collect()

    await asyncio.wait_for(done.wait(), timeout=2.0)
    assert ran["value"] is True
    # The discard done-callback fires a tick after the task completes.
    for _ in range(5):
        await asyncio.sleep(0)
        if not _BACKGROUND_TASKS:
            break
    assert len(_BACKGROUND_TASKS) == 0


@pytest.mark.asyncio
async def test_tracked_task_failure_is_logged_and_discarded():
    async def _boom():
        raise RuntimeError("logging blew up")

    task = asyncio.create_task(_boom())
    with patch("litellm.responses.streaming_iterator.verbose_logger") as mock_logger:
        _track_background_task(task, task_name="failing task")
        # Await the underlying coroutine result without raising into the test.
        with pytest.raises(RuntimeError):
            await task
        # done-callbacks run on the next loop tick.
        await asyncio.sleep(0)

    assert task not in _BACKGROUND_TASKS
    mock_logger.error.assert_called_once()
    assert "failing task" in str(mock_logger.error.call_args)


@pytest.mark.asyncio
async def test_log_completed_response_async_path_tracks_task():
    """The async success-logging call site must register its task with the
    tracker so it cannot be GC'd before the spend-log row is written."""
    iterator = object.__new__(BaseResponsesAPIStreamingIterator)
    iterator._completed_response_logged = False
    iterator._persist_completed_response_before_logging = False
    iterator.completed_response = None
    iterator.start_time = SimpleNamespace()
    iterator._completed_response_cache_hit = False

    handler_started = asyncio.Event()
    handler_release = asyncio.Event()

    async def _fake_dispatch_success_handlers(*args, **kwargs):
        handler_started.set()
        await handler_release.wait()

    iterator.logging_obj = MagicMock()
    iterator.logging_obj.dispatch_success_handlers = _fake_dispatch_success_handlers

    iterator._log_completed_response(is_async=True)

    await asyncio.wait_for(handler_started.wait(), timeout=2.0)
    assert len(_BACKGROUND_TASKS) == 1

    handler_release.set()
    # Let the tracked task finish and its done-callback discard it.
    for _ in range(5):
        await asyncio.sleep(0)
        if not _BACKGROUND_TASKS:
            break
    assert len(_BACKGROUND_TASKS) == 0


@pytest.mark.asyncio
async def test_websocket_log_messages_tracks_task():
    """The WebSocket success-logging call site (_log_messages) must register its
    task with the tracker too — the issue reported both sites."""
    ws = object.__new__(ResponsesWebSocketStreaming)
    ws.input_messages = None
    ws.messages = [{"role": "assistant", "content": "hi"}]

    handler_started = asyncio.Event()
    handler_release = asyncio.Event()

    async def _fake_dispatch_success_handlers(messages, **kwargs):
        handler_started.set()
        await handler_release.wait()

    ws.logging_obj = MagicMock()
    ws.logging_obj.dispatch_success_handlers = _fake_dispatch_success_handlers

    await ws._log_messages()

    await asyncio.wait_for(handler_started.wait(), timeout=2.0)
    assert len(_BACKGROUND_TASKS) == 1

    handler_release.set()
    for _ in range(5):
        await asyncio.sleep(0)
        if not _BACKGROUND_TASKS:
            break
    assert len(_BACKGROUND_TASKS) == 0
