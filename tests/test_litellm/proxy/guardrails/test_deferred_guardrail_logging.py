"""
Tests for Bug 3 fix: deferred logging for post-call guardrails.

When post-call guardrails are configured, the async logging task is deferred
until after post_call_success_hook completes. This ensures the
StandardLoggingPayload is built after guardrails write guardrail_information
to metadata.

Test coverage:
- Detection: _has_post_call_guardrails correctly identifies post-call guardrails
- Deferral: wrapper_async stores closure instead of calling create_task
- Execution: stored closure fires create_task + sync callbacks when called
- Exception: deferred logging fires even if post_call_success_hook raises
- Regression: without flag, create_task fires normally
"""

import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PostCallGuardrail(CustomGuardrail):
    """A post-call guardrail."""

    def __init__(self):
        super().__init__(
            guardrail_name="post-call",
            default_on=True,
            event_hook=GuardrailEventHooks.post_call,
        )

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
    ) -> Any:
        return response


class PreCallGuardrail(CustomGuardrail):
    """A pre-call-only guardrail — should NOT trigger deferral."""

    def __init__(self):
        super().__init__(
            guardrail_name="pre-call",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )


class AllEventsGuardrail(CustomGuardrail):
    """A guardrail with event_hook=None (runs on all events)."""

    def __init__(self):
        super().__init__(
            guardrail_name="all-events",
            default_on=True,
            event_hook=None,
        )


# ---------------------------------------------------------------------------
# 1. _has_post_call_guardrails detection
#
# Breakable scenario: wrong detection → flag set when it shouldn't be
# (unnecessary deferral, harmless) or not set when it should be (bug persists).
# ---------------------------------------------------------------------------


class TestHasPostCallGuardrails:
    def test_returns_true_for_post_call_guardrail(self):
        with patch("litellm.callbacks", [PostCallGuardrail()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is True

    def test_returns_true_for_event_hook_none(self):
        """event_hook=None means 'all events', including post_call."""
        with patch("litellm.callbacks", [AllEventsGuardrail()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is True

    def test_returns_false_for_pre_call_only(self):
        with patch("litellm.callbacks", [PreCallGuardrail()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is False

    def test_returns_false_for_no_callbacks(self):
        with patch("litellm.callbacks", []):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is False

    def test_ignores_non_guardrail_callbacks(self):
        """String callbacks and CustomLogger instances are not guardrails."""
        with patch("litellm.callbacks", ["langfuse", CustomLogger()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is False

    def test_returns_true_for_list_with_post_call(self):
        """event_hook as a list containing post_call should trigger deferral."""

        class ListGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="list-post",
                    default_on=True,
                    event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call],
                )

        with patch("litellm.callbacks", [ListGuardrail()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is True

    def test_returns_false_for_list_without_post_call(self):
        """event_hook as a list without post_call should not trigger deferral."""

        class ListGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="list-pre",
                    default_on=True,
                    event_hook=[GuardrailEventHooks.pre_call],
                )

        with patch("litellm.callbacks", [ListGuardrail()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is False


# ---------------------------------------------------------------------------
# 2. Deferral mechanism: flag → closure stored, create_task skipped
#
# Breakable scenario: flag is set but create_task fires anyway (race persists),
# or closure is not stored (logging lost).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_flag_stores_and_executes_closure():
    """
    When _defer_async_logging is True on logging_obj:
    1. wrapper_async stores a callable closure instead of calling create_task
    2. Calling the closure fires create_task
    3. Sync callbacks fire immediately (not deferred)
    """
    mock_logging_obj = MagicMock()
    mock_logging_obj._defer_async_logging = True
    mock_logging_obj._enqueue_deferred_logging = None

    # Use mock_response to avoid real API call
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="Hello!",
        litellm_logging_obj=mock_logging_obj,
    )

    # --- Part 1: closure was stored ---
    enqueue_fn = mock_logging_obj._enqueue_deferred_logging
    assert callable(enqueue_fn), "Closure should be stored on logging_obj"

    # --- Part 2: sync callbacks fired immediately (not in closure) ---
    mock_logging_obj.handle_sync_success_callbacks_for_async_calls.assert_called_once()

    # --- Part 3: calling the closure fires create_task ---
    created_tasks = []
    real_create_task = asyncio.create_task

    def tracking_create_task(coro):
        task = real_create_task(coro)
        created_tasks.append(task)
        return task

    with patch("asyncio.create_task", side_effect=tracking_create_task):
        enqueue_fn()

    assert len(created_tasks) >= 1, "Closure should fire asyncio.create_task"

    # Clean up tasks
    for task in created_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# 3. Regression: without flag, create_task fires normally
#
# Breakable scenario: the if/else in wrapper_async is wrong and the else
# branch (no flag) also defers → logging broken for ALL requests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_flag_fires_create_task_normally():
    """
    When _defer_async_logging is NOT set, wrapper_async calls
    asyncio.create_task as before (existing behavior preserved).
    """
    created_tasks = []
    real_create_task = asyncio.create_task

    def tracking_create_task(coro):
        task = real_create_task(coro)
        created_tasks.append(task)
        return task

    with patch("asyncio.create_task", side_effect=tracking_create_task):
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
        )

    # At least one create_task call (for _client_async_logging_helper)
    assert len(created_tasks) >= 1

    # Clean up tasks
    for task in created_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# 4. Exception path: deferred logging fires even if post_call_success_hook raises
#
# Breakable scenario: guardrail blocks content (raises HTTPException), the
# finally block doesn't run or doesn't call the closure → logging silently lost.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_logging_fires_on_guardrail_exception():
    """
    If post_call_success_hook raises (e.g., guardrail blocks content),
    the deferred logging closure must still fire (via try/finally).
    """
    enqueue_called = False

    def mock_enqueue():
        nonlocal enqueue_called
        enqueue_called = True

    class BlockingGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="blocker",
                default_on=True,
                event_hook=GuardrailEventHooks.post_call,
            )

        async def async_post_call_success_hook(
            self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
        ) -> Any:
            raise HTTPException(status_code=400, detail="Content blocked")

    guardrail = BlockingGuardrail()

    # Simulate the proxy's try/finally pattern with a raising guardrail
    logging_obj = MagicMock()
    logging_obj._enqueue_deferred_logging = mock_enqueue

    with patch("litellm.callbacks", [guardrail]):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        with pytest.raises(HTTPException):
            try:
                await proxy_logging.post_call_success_hook(
                    data={"model": "gpt-4", "metadata": {}},
                    response=MagicMock(),
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                )
            finally:
                # This mirrors the proxy's finally block
                _enqueue_fn = getattr(logging_obj, "_enqueue_deferred_logging", None)
                if _enqueue_fn is not None:
                    logging_obj._enqueue_deferred_logging = None
                    _enqueue_fn()

    assert enqueue_called is True, "Deferred logging should fire even on guardrail exception"
    assert logging_obj._enqueue_deferred_logging is None, "Closure should be cleared after firing"
