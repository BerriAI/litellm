"""
Tests for deferred logging with post-call guardrails.

When post-call guardrails are configured, the async logging task is deferred
until after guardrails complete.  This ensures the StandardLoggingPayload
is built with guardrail_information populated.

Non-streaming: create_task in wrapper_async is replaced by a closure that
    the proxy fires in a try/finally after post_call_success_hook.

Streaming: CSW.__anext__ stores args on logging_obj at stream end.
    ProxyLogging._fire_deferred_stream_logging fires the closure AFTER all
    guardrail end-of-stream blocks complete. apply_guardrail guardrails are
    skipped (they already ran in unified_guardrail's streaming iterator).
"""

import asyncio
import contextlib
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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

    def test_returns_false_for_event_hook_none(self):
        """event_hook=None is not an explicit post_call registration for deferral."""
        with patch("litellm.callbacks", [AllEventsGuardrail()]):
            assert ProxyBaseLLMRequestProcessing._has_post_call_guardrails() is False

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
                    event_hook=[
                        GuardrailEventHooks.pre_call,
                        GuardrailEventHooks.post_call,
                    ],
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
# 4. Non-streaming: deferred success log is suppressed when guardrail raises
# ---------------------------------------------------------------------------


def test_flush_deferred_async_logging_fires_on_success():
    """
    Happy path: with no exception, the production flush helper invokes the
    deferred async-success closure and clears the slot.
    """
    enqueue_called = False

    def mock_enqueue():
        nonlocal enqueue_called
        enqueue_called = True

    logging_obj = MagicMock()
    logging_obj._enqueue_deferred_logging = mock_enqueue

    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
        logging_obj=logging_obj,
        exception_raised=False,
    )

    assert enqueue_called is True
    assert logging_obj._enqueue_deferred_logging is None


def test_flush_deferred_async_logging_suppressed_on_exception():
    """
    Regression: when post_call_success_hook raises (e.g. a post-call guardrail
    blocks the response), the production flush helper MUST NOT fire the
    deferred async-success closure. The proxy's error path invokes
    post_call_failure_hook which writes its own failure spend log via
    async_failure_handler; firing both produced a duplicate (Success +
    Failure) entry per request, with the Success row exposing the blocked
    LLM response.

    This test exercises the real helper in
    `ProxyBaseLLMRequestProcessing._flush_deferred_async_logging` so a
    regression that re-fires the closure on exception (or removes the
    `exception_raised` gate) is caught.
    """
    enqueue_called = False

    def mock_enqueue():
        nonlocal enqueue_called
        enqueue_called = True

    logging_obj = MagicMock()
    logging_obj._enqueue_deferred_logging = mock_enqueue

    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
        logging_obj=logging_obj,
        exception_raised=True,
    )

    assert enqueue_called is False, (
        "Deferred success log must not fire when the post-call hook raised — "
        "post_call_failure_hook writes its own failure log."
    )
    # Slot is still cleared so a follow-up flush does not double-fire.
    assert logging_obj._enqueue_deferred_logging is None


def test_flush_deferred_async_logging_noop_when_no_closure_stored():
    """
    Streaming early-returns and non-deferred paths never store a closure;
    flushing must be a no-op (no AttributeError, no firing).
    """

    class _Bare:
        pass

    logging_obj = _Bare()  # no _enqueue_deferred_logging attribute

    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
        logging_obj=logging_obj,
        exception_raised=False,
    )
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
        logging_obj=logging_obj,
        exception_raised=True,
    )

    # Helper must not create the attribute as a side effect.
    assert not hasattr(logging_obj, "_enqueue_deferred_logging")


def test_proxy_finally_block_routes_through_flush_helper():
    """
    Source-level contract: the proxy's `base_process_llm_request` finally
    block must delegate to `_flush_deferred_async_logging` rather than
    inlining the gating logic. Inlining is what allowed the duplicate
    Success+Failure spend log to slip in originally — this guards the
    refactor.
    """
    import inspect

    src = inspect.getsource(ProxyBaseLLMRequestProcessing.base_process_llm_request)
    assert "_flush_deferred_async_logging" in src, (
        "base_process_llm_request must call _flush_deferred_async_logging "
        "from its finally block — do not inline the gating logic."
    )
    # Belt-and-braces: the inlined `_enqueue_deferred_logging = None` reset
    # was the symptom of the duplicate-log bug; assert it stays inside the
    # helper, not in the request-processing function.
    assert "_enqueue_deferred_logging = None" not in src, (
        "Reset of _enqueue_deferred_logging must live inside "
        "_flush_deferred_async_logging, not in base_process_llm_request."
    )


def test_flush_deferred_async_logging_swallows_closure_errors():
    """
    The flush helper must not propagate exceptions from the deferred closure —
    a logging failure must not break the request lifecycle for the caller.
    """

    def boom():
        raise RuntimeError("logger failure")

    logging_obj = MagicMock()
    logging_obj._enqueue_deferred_logging = boom

    # Should not raise.
    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
        logging_obj=logging_obj,
        exception_raised=False,
    )
    assert logging_obj._enqueue_deferred_logging is None


# ---------------------------------------------------------------------------
# 5. Streaming: closure defers logging at stream end
# ---------------------------------------------------------------------------


class TestDeferredStreamingClosure:
    @pytest.mark.asyncio
    async def test_streaming_stores_deferred_args(self):
        """When _on_deferred_stream_complete is set, CSW stores the assembled
        response args on logging_obj instead of calling the closure directly."""
        mock_logging_obj = MagicMock()
        mock_logging_obj._on_deferred_stream_complete = MagicMock()

        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="Hello!",
            stream=True,
            litellm_logging_obj=mock_logging_obj,
        )
        async for _ in resp:
            pass

        # CSW should store args, NOT call the closure
        assert hasattr(mock_logging_obj, "_deferred_stream_complete_args")
        args = mock_logging_obj._deferred_stream_complete_args
        assert args is not None, "Deferred args should be stored"
        assert len(args) == 2, "Should be (assembled_response, cache_hit)"
        assert args[0] is not None, "Assembled response should not be None"

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

            # CSW stored args; now simulate what ProxyLogging does
            request_data = {"litellm_logging_obj": mock_logging_obj}
            ProxyLogging._fire_deferred_stream_logging(request_data)

            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert guardrail_called is True, "Guardrail hook should be called"
        assert (
            logger_called is False
        ), "Non-guardrail logger should NOT be called by closure"

    @pytest.mark.asyncio
    async def test_closure_passes_guardrail_modified_response_to_logging(self):
        """The production _run_deferred_stream_guardrails must pass the
        guardrail-modified response to async_success_handler."""
        logged_response = None
        modified_response = MagicMock()

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            nonlocal logged_response
            logged_response = args[0] if args else None

        mock_logging_obj.async_success_handler = track_async_success

        class ModifyingGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="modifier",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                return modified_response

        guardrail = ModifyingGuardrail()

        with patch("litellm.callbacks", [guardrail]):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert (
            logged_response is modified_response
        ), "Logging must receive the guardrail-modified response"

    @pytest.mark.asyncio
    async def test_closure_logs_even_on_guardrail_exception(self):
        """If a guardrail raises HTTPException, the production
        _run_deferred_stream_guardrails must still fire logging
        and set guardrail_blocked in metadata."""
        from fastapi import HTTPException  # noqa: local import for test isolation

        logging_called = False

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
                raise HTTPException(status_code=400, detail="Blocked")

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            nonlocal logging_called
            logging_called = True

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = BlockingGuardrail()

        with patch("litellm.callbacks", [guardrail]):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert (
            logging_called is True
        ), "Logging must fire even when guardrail raises HTTPException"
        assert (
            mock_logging_obj.model_call_details["metadata"].get("guardrail_blocked")
            is True
        ), "guardrail_blocked must be set for HTTPException"

    @pytest.mark.asyncio
    async def test_transient_error_does_not_set_guardrail_blocked(self):
        """Transient errors (not HTTPException) should NOT set
        guardrail_blocked. Uses the production _run_deferred_stream_guardrails."""

        class TransientErrorGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="transient",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                raise ConnectionError("Network timeout")

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = TransientErrorGuardrail()

        with patch("litellm.callbacks", [guardrail]):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)

        assert (
            mock_logging_obj.model_call_details["metadata"].get("guardrail_blocked")
            is not True
        ), "guardrail_blocked must NOT be set for transient errors"

    @pytest.mark.asyncio
    async def test_production_closure_integration(self):
        """Integration test: CSW stores args, then _fire_deferred_stream_logging
        fires the closure which calls _run_deferred_stream_guardrails."""
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

            # CSW stored args; now simulate what ProxyLogging does
            request_data = {"litellm_logging_obj": mock_logging_obj}
            ProxyLogging._fire_deferred_stream_logging(request_data)

            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert hook_called is True, "Production closure must call guardrail hook"
        assert (
            logged_response is modified_response
        ), "Production closure must pass guardrail-modified response to logging"

    @pytest.mark.asyncio
    async def test_apply_guardrail_skipped_in_deferred_path(self):
        """Guardrails that define apply_guardrail should be SKIPPED in
        _run_deferred_stream_guardrails (they already ran via unified_guardrail's
        streaming end-of-stream block)."""
        from litellm.types.utils import GenericGuardrailAPIInputs

        apply_guardrail_called = False

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
                nonlocal apply_guardrail_called
                apply_guardrail_called = True
                return inputs

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = ApplyGuardrailType()

        with patch("litellm.callbacks", [guardrail]):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)

        assert (
            apply_guardrail_called is False
        ), "apply_guardrail guardrails must be SKIPPED in deferred path"

    @pytest.mark.asyncio
    async def test_streaming_iterator_hook_skipped_in_deferred_path(self):
        """regression test: guardrails that define async_post_call_streaming_iterator_hook must be SKIPPED in _run_deferred_stream_guardrails.
        The iterator hook already scanned the assembled response in the streaming
        pipeline"""
        success_hook_called = False

        class IteratorHookGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="iterator-hook",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_streaming_iterator_hook(
                self, user_api_key_dict, response, request_data
            ):
                async for chunk in response:
                    yield chunk

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                nonlocal success_hook_called
                success_hook_called = True
                return response

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = IteratorHookGuardrail()

        with patch("litellm.callbacks", [guardrail]):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)

        assert success_hook_called is False, (
            "Guardrails that implement async_post_call_streaming_iterator_hook "
            "must be SKIPPED in deferred path — the iterator hook already ran"
        )

    @pytest.mark.asyncio
    async def test_hooks_receive_merged_guardrail_data(self):
        """Hooks must receive guardrail_data (the merged dict from
        _check_and_merge_model_level_guardrails), not the original
        captured_data.  This ensures model-level non-default guardrails
        are visible to any inner should_run_guardrail re-checks.

        Uses a deep-copy mock to break the shallow-copy side-effect that
        would otherwise mask the bug — verifying the code is explicitly
        correct, not correct-by-accident."""
        import copy

        hook_received_data = None

        class InspectingGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="inspector",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                nonlocal hook_received_data
                hook_received_data = data
                return response

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = InspectingGuardrail()

        captured_data = {"model": "gpt-4", "metadata": {"existing_key": "value"}}

        def mock_merge(data, llm_router):
            """Return a fully independent dict (deep copy) so the original
            captured_data is NOT mutated.  This simulates a correct merge
            implementation and proves _run_deferred_stream_guardrails uses
            the return value, not the original data."""
            merged = copy.deepcopy(data)
            merged["metadata"]["guardrails"] = ["model-guardrail"]
            merged["_merged_marker"] = True
            return merged

        with (
            patch("litellm.callbacks", [guardrail]),
            patch(
                "litellm.proxy.utils._check_and_merge_model_level_guardrails",
                side_effect=mock_merge,
            ),
        ):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=captured_data,
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        assert hook_received_data is not None, "Guardrail hook must be called"
        assert (
            hook_received_data.get("_merged_marker") is True
        ), "Hook must receive guardrail_data (merged), not original captured_data"
        assert "model-guardrail" in hook_received_data.get("metadata", {}).get(
            "guardrails", []
        ), "Hook data must contain model-level guardrails"

    @pytest.mark.asyncio
    async def test_multiple_guardrails_all_receive_merged_data(self):
        """When multiple guardrails are configured, ALL of them must receive
        guardrail_data (merged), not just the first one."""
        import copy

        received_data_per_guardrail = {}

        class TaggedGuardrail(CustomGuardrail):
            def __init__(self, name):
                super().__init__(
                    guardrail_name=name,
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                received_data_per_guardrail[self.guardrail_name] = data
                return response

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        guardrail_a = TaggedGuardrail("guardrail-a")
        guardrail_b = TaggedGuardrail("guardrail-b")

        captured_data = {"model": "gpt-4", "metadata": {}}

        def mock_merge(data, llm_router):
            merged = copy.deepcopy(data)
            merged["metadata"]["guardrails"] = ["guardrail-a", "guardrail-b"]
            merged["_merged_marker"] = True
            return merged

        with (
            patch("litellm.callbacks", [guardrail_a, guardrail_b]),
            patch(
                "litellm.proxy.utils._check_and_merge_model_level_guardrails",
                side_effect=mock_merge,
            ),
        ):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=captured_data,
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        for name in ("guardrail-a", "guardrail-b"):
            assert name in received_data_per_guardrail, f"{name} must be called"
            assert (
                received_data_per_guardrail[name].get("_merged_marker") is True
            ), f"{name} must receive guardrail_data (merged), not captured_data"

    @pytest.mark.asyncio
    async def test_logging_fires_even_if_guardrail_init_raises(self):
        """If _check_and_merge_model_level_guardrails raises during
        initialization, logging must still fire via the try/finally guard.
        This prevents silent logging loss on transient init errors."""
        logging_called = False

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            nonlocal logging_called
            logging_called = True

        mock_logging_obj.async_success_handler = track_async_success

        def exploding_merge(data, llm_router):
            raise RuntimeError("Simulated init failure")

        with patch(
            "litellm.proxy.utils._check_and_merge_model_level_guardrails",
            side_effect=exploding_merge,
        ):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data={"model": "gpt-4", "metadata": {}},
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert (
            logging_called is True
        ), "Logging must fire even when guardrail initialization raises"


# ---------------------------------------------------------------------------
# 7. _fire_deferred_stream_logging
# ---------------------------------------------------------------------------


class TestFireDeferredStreamLogging:
    @pytest.mark.asyncio
    async def test_fires_callback_with_stored_args(self):
        """_fire_deferred_stream_logging should call the deferred callback
        with the stored args."""
        callback_called = False
        callback_args = {}

        async def mock_callback(assembled_response, cache_hit):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = {"response": assembled_response, "cache_hit": cache_hit}

        mock_logging_obj = MagicMock()
        mock_logging_obj._on_deferred_stream_complete = mock_callback
        mock_logging_obj._deferred_stream_complete_args = ("test_response", True)

        request_data = {"litellm_logging_obj": mock_logging_obj}
        ProxyLogging._fire_deferred_stream_logging(request_data)

        await asyncio.sleep(0)

        assert callback_called is True
        assert callback_args["response"] == "test_response"
        assert callback_args["cache_hit"] is True
        # Attributes should be cleared
        assert mock_logging_obj._on_deferred_stream_complete is None
        assert mock_logging_obj._deferred_stream_complete_args is None

    @pytest.mark.asyncio
    async def test_noop_when_no_deferred_args(self):
        """_fire_deferred_stream_logging should be a no-op when no deferred
        args are stored."""
        mock_logging_obj = MagicMock()
        mock_logging_obj._on_deferred_stream_complete = None

        request_data = {"litellm_logging_obj": mock_logging_obj}
        # Should not raise
        ProxyLogging._fire_deferred_stream_logging(request_data)

    @pytest.mark.asyncio
    async def test_noop_when_no_logging_obj(self):
        """_fire_deferred_stream_logging should be a no-op when
        litellm_logging_obj is missing from request_data."""
        request_data = {}
        # Should not raise
        ProxyLogging._fire_deferred_stream_logging(request_data)

    @pytest.mark.asyncio
    async def test_short_stream_guardrail_info_populated(self):
        """Verify that _run_deferred_stream_guardrails populates
        guardrail_information for guardrails using async_post_call_success_hook
        (non-apply_guardrail path) even with short streams."""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        logged_response = None

        async def track_async_success(*args, **kwargs):
            nonlocal logged_response
            logged_response = args[0] if args else None

        mock_logging_obj.async_success_handler = track_async_success

        class InfoWritingGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="info-writer",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def async_post_call_success_hook(
                self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any
            ) -> Any:
                # Simulate writing guardrail_information
                metadata = data.setdefault("metadata", {})
                info_list = metadata.setdefault(
                    "standard_logging_guardrail_information", []
                )
                info_list.append({"guardrail_name": "info-writer", "status": "success"})
                return response

        guardrail = InfoWritingGuardrail()
        captured_data = {"model": "gpt-4", "metadata": {}}

        with patch("litellm.callbacks", [guardrail]):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=captured_data,
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        info = captured_data["metadata"].get("standard_logging_guardrail_information")
        assert info is not None, "guardrail_information should be populated"
        assert len(info) == 1
        assert info[0]["guardrail_name"] == "info-writer"


class _FakeCSW:
    """Stands in for CustomStreamWrapper. ``aclose()`` mirrors real CSW: if
    a deferred-streaming closure is registered and no args have been set yet,
    persist partial args from accumulated chunks so the outer hook can fire."""

    def __init__(self, chunks, logging_obj, raise_after=None):
        self._chunks, self._logging_obj, self._raise_after = (
            list(chunks),
            logging_obj,
            raise_after,
        )
        self._idx, self._closed = 0, False
        self.aclose_calls: list = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        if self._raise_after is not None and self._idx >= self._raise_after:
            raise RuntimeError("inner stream blew up")
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        v = self._chunks[self._idx]
        self._idx += 1
        return v

    async def aclose(self):
        self.aclose_calls.append(self._idx)
        self._closed = True
        lo = self._logging_obj
        if (
            getattr(lo, "_on_deferred_stream_complete", None) is not None
            and getattr(lo, "_deferred_stream_complete_args", None) is None
        ):
            lo._deferred_stream_complete_args = (f"partial_after_{self._idx}", False)


async def _exception_wrapper(inner):
    """Mimics _wrap_streaming_iterator_with_enrichment. ``except: raise`` is
    what marks the wrapper's generator frame completed when an exception
    propagates — making any aclose() on the wrapper a no-op."""
    try:
        async for c in inner:
            yield c
    except Exception:
        raise


@contextlib.contextmanager
def _hook_under_test(capture_fire):
    """Patches stay alive for the lifetime of the generator (the generator's
    finally runs ``_fire_deferred_stream_logging`` so the patch must outlive
    iteration)."""
    with (
        patch.object(
            ProxyLogging, "_fire_deferred_stream_logging", staticmethod(capture_fire)
        ),
        patch("litellm.callbacks", []),
    ):
        yield ProxyLogging(user_api_key_cache=DualCache())


class TestStreamingDisconnectFiresDeferredLogging:
    """AAr6bXKP regression suite for ``async_post_call_streaming_iterator_hook``.

    The hook is bound to the client connection. Without ``try/finally`` +
    aclose-on-inner, deferred logging is skipped on disconnect / exception
    paths — the caller consumes LLM tokens unbilled, bypassing budgets and
    PII / moderation / policy guardrails. The fix:
      (1) wraps iteration in try/finally;
      (2) snapshots the inner ``response`` BEFORE the wrap chain and drives
          ``aclose()`` on the snapshot directly (the wrapper's aclose is a
          no-op after an exception completes its generator frame);
      (3) ``CSW.aclose()`` persists partial args from accumulated chunks.
    """

    @pytest.mark.parametrize(
        "termination,should_fire",
        [
            ("normal_completion", True),
            ("client_disconnect", True),  # partial usage still attributed
            # Starlette / anyio cancel the streaming task on client
            # disconnect — surfaces as ``asyncio.CancelledError``, which is
            # a ``BaseException`` not a regular ``Exception``. Must still
            # fire deferred logging (same financial-integrity property as
            # GeneratorExit disconnect).
            ("cancellederror_disconnect", True),
            # Iterator-raised exceptions (post-call guardrail blocked the
            # response, upstream provider error mid-stream, etc.) must NOT
            # fire deferred success — that would persist a blocked /
            # errored response. The upstream caller's
            # post_call_failure_hook handles the failure side.
            ("exception_mid_stream", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_deferred_logging_fires_only_on_clean_or_disconnect(
        self, termination, should_fire
    ):
        fired_with: list = []

        async def _normal():
            yield "only"

        async def _raises():
            yield "c1"
            raise RuntimeError("guardrail blocked")

        async def _three():
            yield "c1"
            yield "c2"
            yield "c3"

        async def _two_then_cancel():
            # Simulates an upstream iterator that gets cancelled while it's
            # awaiting the next chunk — mirrors what Starlette / anyio do
            # when the client disconnects mid-stream.
            yield "c1"
            yield "c2"
            raise asyncio.CancelledError()

        response = {
            "normal_completion": _normal(),
            "client_disconnect": _three(),
            "cancellederror_disconnect": _two_then_cancel(),
            "exception_mid_stream": _raises(),
        }[termination]
        request_data = {"litellm_logging_obj": MagicMock()}

        with _hook_under_test(lambda r: fired_with.append(r)) as pl:
            gen = pl.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                response=response,
                request_data=request_data,
            )
            if termination == "client_disconnect":
                chunks_seen = []
                async for chunk in gen:
                    chunks_seen.append(chunk)
                    if len(chunks_seen) == 2:
                        await gen.aclose()
                        break
            elif termination == "cancellederror_disconnect":
                with pytest.raises(asyncio.CancelledError):
                    async for _ in gen:
                        pass
            elif termination == "exception_mid_stream":
                with pytest.raises(RuntimeError, match="guardrail blocked"):
                    async for _ in gen:
                        pass
            else:
                async for _ in gen:
                    pass

        if should_fire:
            assert fired_with == [request_data]
        else:
            assert (
                fired_with == []
            ), "deferred success logging must NOT fire when iterator raises"

    @pytest.mark.asyncio
    async def test_disconnect_drives_aclose_but_skips_partial_log(self):
        """``aclose()`` is driven down the wrapper chain (HTTP / upstream
        cleanup), but ``_fire_deferred_stream_logging`` no-ops because args
        weren't populated. Veria-flagged: populating args from partial
        chunks would persist raw unguardrailed content to SpendLogs and
        external logging callbacks. Awaiting a usage-only logging path
        before re-enabling disconnect-time spend attribution."""
        fired: list = []
        lo = MagicMock(
            _on_deferred_stream_complete=MagicMock(),
            _deferred_stream_complete_args=None,
        )
        # _FakeCSW.aclose populates args (mimics CSW pre-Veria-revert) —
        # but in the real hook flow we use the REAL CSW which no longer
        # populates. Override _FakeCSW behaviour here:
        csw = _FakeCSW(["c1", "c2", "c3"], lo)

        async def _real_aclose():
            csw.aclose_calls.append(csw._idx)
            csw._closed = True
            # Intentionally do NOT populate args — matches the real
            # post-Veria CSW.aclose contract.

        csw.aclose = _real_aclose  # type: ignore[assignment]

        with _hook_under_test(
            lambda r: fired.append(
                r["litellm_logging_obj"]._deferred_stream_complete_args is not None
            )
        ) as pl:
            gen = pl.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                response=csw,
                request_data={"litellm_logging_obj": lo},
            )
            chunks_seen = []
            async for chunk in gen:
                chunks_seen.append(chunk)
                if len(chunks_seen) == 2:
                    await gen.aclose()
                    break

        assert csw.aclose_calls, "CSW.aclose() must still run for cleanup"
        assert fired == [False], (
            "_fire_deferred_stream_logging must be invoked, but with args "
            "unset on disconnect — the helper itself no-ops internally so "
            "no body export happens"
        )

    @pytest.mark.asyncio
    async def test_aclose_reaches_inner_when_wrapper_closed_by_exception(self):
        """Greptile P1: wrapper.aclose() is a no-op after exception completes
        its generator frame. The snapshot-before-wrap fix closes the inner
        CSW directly on every termination path — including iterator
        exceptions — so HTTP-connection / upstream-stream cleanup runs
        deterministically. (Veria follow-up: success-deferred logging is
        explicitly skipped on the exception path; we assert aclose still
        runs but the deferred fire is NOT called.)"""
        fired: list = []
        lo = MagicMock(
            _on_deferred_stream_complete=MagicMock(),
            _deferred_stream_complete_args=None,
        )
        csw = _FakeCSW(["a", "b", "c"], lo, raise_after=2)

        with _hook_under_test(
            lambda r: fired.append(
                r["litellm_logging_obj"]._deferred_stream_complete_args
            )
        ) as pl:
            gen = pl.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                response=csw,
                request_data={"litellm_logging_obj": lo},
            )
            with pytest.raises(RuntimeError, match="inner stream blew up"):
                async for _ in gen:
                    pass

        assert csw.aclose_calls, "inner CSW.aclose() must run on exception path"
        assert fired == [], (
            "success-deferred logging must NOT fire on the exception path; "
            "the upstream post_call_failure_hook handles the failure side"
        )
