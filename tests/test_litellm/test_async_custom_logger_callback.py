"""
Tests for #8842 — async CustomLogger callbacks must fire without
requiring the caller to yield to the event loop (e.g., asyncio.sleep).

The root cause was that `wrapper_async` dispatched async callbacks via a
fire-and-forget `asyncio.create_task`, which could be cancelled before
running. The fix enqueues the coroutine synchronously to
GLOBAL_LOGGING_WORKER.
"""

import asyncio
from typing import List

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger


class _EventTracker(CustomLogger):
    """Records which callback events fire."""

    def __init__(self):
        super().__init__()
        self.events: List[str] = []

    def log_pre_api_call(self, model, messages, kwargs):
        self.events.append("pre_api_call")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        self.events.append("post_api_call")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.events.append("sync_success")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.events.append("async_success")


@pytest.mark.asyncio
async def test_acompletion_fires_async_callback():
    """Direct litellm.acompletion must trigger async_log_success_event."""
    tracker = _EventTracker()
    old_callbacks = list(litellm.callbacks)
    litellm.callbacks = [tracker]
    try:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi!",
        )

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5.0)

        assert "async_success" in tracker.events, (
            f"async_log_success_event was NOT called. Events: {tracker.events}"
        )
    finally:
        litellm.callbacks = old_callbacks


@pytest.mark.asyncio
async def test_router_acompletion_fires_async_callback():
    """Router.acompletion must trigger async_log_success_event."""
    tracker = _EventTracker()
    old_callbacks = list(litellm.callbacks)
    litellm.callbacks = [tracker]
    try:
        router = litellm.Router(
            model_list=[
                {"model_name": "default", "litellm_params": {"model": "gpt-3.5-turbo"}},
            ],
        )
        await router.acompletion(
            model="default",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi!",
        )

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5.0)

        assert "async_success" in tracker.events, (
            f"async_log_success_event was NOT called via router. Events: {tracker.events}"
        )
    finally:
        litellm.callbacks = old_callbacks


@pytest.mark.asyncio
async def test_no_double_async_callback():
    """async_log_success_event must fire exactly once per completion."""
    tracker = _EventTracker()
    old_callbacks = list(litellm.callbacks)
    litellm.callbacks = [tracker]
    try:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi!",
        )

        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5.0)

        count = tracker.events.count("async_success")
        assert count == 1, (
            f"async_log_success_event called {count} times (expected 1). Events: {tracker.events}"
        )
    finally:
        litellm.callbacks = old_callbacks
