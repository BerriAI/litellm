"""
Tests for deferred logging with post-call guardrails.

When post-call guardrails are configured, the async logging task is deferred
until after guardrails complete.  This ensures the StandardLoggingPayload
is built with guardrail_information populated.

Non-streaming: create_task in wrapper_async is replaced by a closure that
    the proxy fires in a try/finally after post_call_success_hook.

Streaming: a closure on logging_obj is called by CSW.__anext__ at stream end.
    The closure runs ONLY guardrail hooks (not all callbacks), then fires
    both logging handlers.
"""

import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.exceptions import HTTPException

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
# 2. Non-streaming: deferral flag → closure stored, create_task skipped
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

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="Hello!",
        litellm_logging_obj=mock_logging_obj,
    )

    # Closure was stored
    enqueue_fn = mock_logging_obj._enqueue_deferred_logging
    assert callable(enqueue_fn), "Closure should be stored on logging_obj"

    # Sync callbacks fired immediately
    mock_logging_obj.handle_sync_success_callbacks_for_async_calls.assert_called_once()

    # Calling the closure fires create_task
    created_tasks = []
    real_create_task = asyncio.create_task

    def tracking_create_task(coro):
        task = real_create_task(coro)
        created_tasks.append(task)
        return task

    with patch("asyncio.create_task", side_effect=tracking_create_task):
        enqueue_fn()

    assert len(created_tasks) >= 1, "Closure should fire asyncio.create_task"

    for task in created_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# 3. Non-streaming regression: without flag, create_task fires normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_flag_fires_create_task_normally():
    """Without _defer_async_logging, wrapper_async calls create_task as before."""
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

    assert len(created_tasks) >= 1

    for task in created_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# 4. Non-streaming: deferred logging fires even if guardrail raises
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
                # Mirrors the proxy's finally block
                _enqueue_fn = getattr(logging_obj, "_enqueue_deferred_logging", None)
                if _enqueue_fn is not None:
                    logging_obj._enqueue_deferred_logging = None
                    _enqueue_fn()

    assert enqueue_called is True
    assert logging_obj._enqueue_deferred_logging is None


# ---------------------------------------------------------------------------
# 5. Streaming: closure defers logging at stream end
# ---------------------------------------------------------------------------


class TestDeferredStreamingClosure:
    @pytest.mark.asyncio
    async def test_streaming_closure_defers_logging(self):
        """When _on_deferred_stream_complete is set, CSW calls the closure
        instead of firing async_success_handler directly."""
        mock_logging_obj = MagicMock()
        callback_called = False
        callback_args = {}

        async def mock_callback(assembled_response, cache_hit):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = {"response": assembled_response, "cache_hit": cache_hit}

        mock_logging_obj._on_deferred_stream_complete = mock_callback

        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
            stream=True,
            litellm_logging_obj=mock_logging_obj,
        )
        async for _ in resp:
            pass

        await asyncio.sleep(0)

        assert callback_called is True, "Closure should be called at stream end"
        assert callback_args["response"] is not None
        assert mock_logging_obj._on_deferred_stream_complete is None

    @pytest.mark.asyncio
    async def test_streaming_no_closure_fires_normally(self):
        """Regression: without closure, CSW fires logging immediately."""
        created_tasks = []
        real_create_task = asyncio.create_task

        def tracking_create_task(coro):
            task = real_create_task(coro)
            created_tasks.append(task)
            return task

        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
            stream=True,
        )
        with patch("asyncio.create_task", side_effect=tracking_create_task):
            async for _ in resp:
                pass

        assert len(created_tasks) >= 1
        for task in created_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    @pytest.mark.asyncio
    async def test_closure_runs_only_guardrail_hooks(self):
        """The closure must call only CustomGuardrail hooks, not all callbacks.
        This is the key v2 change — PR #23929 called post_call_success_hook
        which ran ALL callbacks, causing behavioral changes for streaming."""
        guardrail_called = False
        logger_called = False

        class TrackingGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="tracker",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                nonlocal guardrail_called
                guardrail_called = True
                return response

        class TrackingLogger(CustomLogger):
            async def async_post_call_success_hook(
                self, user_api_key_dict, data, response
            ):
                nonlocal logger_called
                logger_called = True
                return response

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        tracking_guardrail = TrackingGuardrail()
        tracking_logger = TrackingLogger()

        # Use the real production static method via a thin closure
        _captured_data = {"model": "gpt-4", "metadata": {}}
        _captured_user_api_key_dict = UserAPIKeyAuth(api_key="test")

        async def _on_deferred_stream_complete(assembled_response, cache_hit):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=_captured_data,
                captured_user_api_key_dict=_captured_user_api_key_dict,
                captured_logging_obj=mock_logging_obj,
                assembled_response=assembled_response,
                cache_hit=cache_hit,
            )

        mock_logging_obj._on_deferred_stream_complete = _on_deferred_stream_complete

        with patch("litellm.callbacks", [tracking_guardrail, tracking_logger]):
            resp = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="Hello!",
                stream=True,
                litellm_logging_obj=mock_logging_obj,
            )
            async for _ in resp:
                pass

            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert guardrail_called is True, "Guardrail hook should be called"
        assert logger_called is False, "Non-guardrail logger should NOT be called by closure"

    @pytest.mark.asyncio
    async def test_closure_passes_guardrail_modified_response_to_logging(self):
        """The closure passes the guardrail-modified response to logging handlers."""
        mock_logging_obj = MagicMock()
        modified_response = MagicMock()
        logged_response = None

        async def mock_async_success(*args, **kwargs):
            nonlocal logged_response
            logged_response = args[0] if args else None

        mock_logging_obj.async_success_handler = mock_async_success

        async def closure(assembled_response, cache_hit):
            # Simulate guardrail modifying the response
            asyncio.create_task(
                mock_logging_obj.async_success_handler(
                    modified_response, cache_hit=cache_hit, start_time=None, end_time=None
                )
            )

        mock_logging_obj._on_deferred_stream_complete = closure

        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
            stream=True,
            litellm_logging_obj=mock_logging_obj,
        )
        async for _ in resp:
            pass

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert logged_response is modified_response

    @pytest.mark.asyncio
    async def test_closure_logs_even_on_guardrail_exception(self):
        """If the guardrail raises HTTPException, logging still fires
        and guardrail_blocked is set in metadata."""
        logging_called = False

        async def mock_async_success(*args, **kwargs):
            nonlocal logging_called
            logging_called = True

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}
        mock_logging_obj.async_success_handler = mock_async_success

        async def closure(assembled_response, cache_hit):
            _response = assembled_response
            try:
                raise HTTPException(status_code=400, detail="Blocked")
            except Exception as e:
                if isinstance(e, HTTPException) and hasattr(
                    mock_logging_obj, "model_call_details"
                ):
                    mock_logging_obj.model_call_details.setdefault(
                        "metadata", {}
                    )["guardrail_blocked"] = True

            asyncio.create_task(
                mock_logging_obj.async_success_handler(
                    _response, cache_hit=cache_hit, start_time=None, end_time=None
                )
            )

        mock_logging_obj._on_deferred_stream_complete = closure

        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
            stream=True,
            litellm_logging_obj=mock_logging_obj,
        )
        async for _ in resp:
            pass

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert logging_called is True
        assert mock_logging_obj.model_call_details["metadata"].get(
            "guardrail_blocked"
        ) is True

    @pytest.mark.asyncio
    async def test_transient_error_does_not_set_guardrail_blocked(self):
        """Transient errors (not HTTPException) should NOT set guardrail_blocked."""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def mock_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = mock_async_success

        async def closure(assembled_response, cache_hit):
            try:
                raise ConnectionError("Network timeout")
            except Exception as e:
                if isinstance(e, HTTPException) and hasattr(
                    mock_logging_obj, "model_call_details"
                ):
                    mock_logging_obj.model_call_details.setdefault(
                        "metadata", {}
                    )["guardrail_blocked"] = True

            asyncio.create_task(
                mock_logging_obj.async_success_handler(
                    assembled_response, cache_hit=cache_hit, start_time=None, end_time=None
                )
            )

        mock_logging_obj._on_deferred_stream_complete = closure

        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
            stream=True,
            litellm_logging_obj=mock_logging_obj,
        )
        async for _ in resp:
            pass

        await asyncio.sleep(0)

        assert mock_logging_obj.model_call_details["metadata"].get(
            "guardrail_blocked"
        ) is not True

    @pytest.mark.asyncio
    async def test_production_closure_integration(self):
        """Integration test: calls the real _run_deferred_stream_guardrails
        static method and verifies it calls guardrail hooks and passes
        the modified response to logging."""
        hook_called = False
        logged_response = None
        modified_response = MagicMock()

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            nonlocal logged_response
            logged_response = args[0] if args else None

        mock_logging_obj.async_success_handler = track_async_success

        class TestGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="test",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                nonlocal hook_called
                hook_called = True
                return modified_response

        guardrail = TestGuardrail()

        async def _on_deferred_stream_complete(assembled_response, cache_hit):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=assembled_response,
                cache_hit=cache_hit,
            )

        mock_logging_obj._on_deferred_stream_complete = _on_deferred_stream_complete

        with patch("litellm.callbacks", [guardrail]):
            resp = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="Hello!",
                stream=True,
                litellm_logging_obj=mock_logging_obj,
            )
            async for _ in resp:
                pass

            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert hook_called is True, \
            "Production closure must call guardrail hook"
        assert logged_response is modified_response, \
            "Production closure must pass guardrail-modified response to logging"

    @pytest.mark.asyncio
    async def test_apply_guardrail_path_uses_unified_guardrail(self):
        """Guardrails that define apply_guardrail should be dispatched through
        UnifiedLLMGuardrails.async_post_call_success_hook via the real
        _run_deferred_stream_guardrails static method."""
        from litellm.types.utils import GenericGuardrailAPIInputs

        unified_hook_called = False

        class ApplyGuardrailType(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="apply-type",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def apply_guardrail(
                self, inputs, request_data, input_type, logging_obj=None
            ) -> GenericGuardrailAPIInputs:
                nonlocal unified_hook_called
                unified_hook_called = True
                return inputs

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}
        logged_response = None

        async def track_async_success(*args, **kwargs):
            nonlocal logged_response
            logged_response = args[0] if args else None

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = ApplyGuardrailType()

        async def _on_deferred_stream_complete(assembled_response, cache_hit):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=assembled_response,
                cache_hit=cache_hit,
            )

        mock_logging_obj._on_deferred_stream_complete = _on_deferred_stream_complete

        with patch("litellm.callbacks", [guardrail]):
            resp = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="Hello!",
                stream=True,
                litellm_logging_obj=mock_logging_obj,
            )
            async for _ in resp:
                pass

            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert unified_hook_called is True, \
            "apply_guardrail guardrails must be dispatched through UnifiedLLMGuardrails"
        assert logged_response is not None, \
            "Logging must fire after unified guardrail path"
