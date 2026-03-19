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

        assert logged_response is modified_response, \
            "Logging must receive the guardrail-modified response"

    @pytest.mark.asyncio
    async def test_closure_logs_even_on_guardrail_exception(self):
        """If a guardrail raises HTTPException, the production
        _run_deferred_stream_guardrails must still fire logging
        and set guardrail_blocked in metadata."""
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

        assert logging_called is True, \
            "Logging must fire even when guardrail raises HTTPException"
        assert mock_logging_obj.model_call_details["metadata"].get(
            "guardrail_blocked"
        ) is True, "guardrail_blocked must be set for HTTPException"

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

        assert mock_logging_obj.model_call_details["metadata"].get(
            "guardrail_blocked"
        ) is not True, "guardrail_blocked must NOT be set for transient errors"

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

        with patch("litellm.callbacks", [guardrail]), \
             patch(
                "litellm.proxy.utils._check_and_merge_model_level_guardrails",
                side_effect=mock_merge,
             ):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=captured_data,
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        assert hook_received_data is not None, "Guardrail hook must be called"
        assert hook_received_data.get("_merged_marker") is True, \
            "Hook must receive guardrail_data (merged), not original captured_data"
        assert "model-guardrail" in hook_received_data.get("metadata", {}).get(
            "guardrails", []
        ), "Hook data must contain model-level guardrails"

    @pytest.mark.asyncio
    async def test_apply_guardrail_path_receives_merged_guardrail_data(self):
        """The apply_guardrail path (through UnifiedLLMGuardrails) must also
        receive guardrail_data so that the inner should_run_guardrail re-check
        inside UnifiedLLMGuardrails sees model-level guardrails.

        This is the specific scenario Greptile flagged: a default_on=False
        guardrail configured at the model level would pass the outer gate but
        be silently skipped at execution time if captured_data (unmerged) were
        passed instead of guardrail_data (merged)."""
        import copy
        from litellm.types.utils import GenericGuardrailAPIInputs

        unified_received_data = None

        class ModelLevelApplyGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="model-apply-guardrail",
                    default_on=True,
                    event_hook=GuardrailEventHooks.post_call,
                )

            async def apply_guardrail(
                self, inputs, request_data, input_type, logging_obj=None
            ) -> GenericGuardrailAPIInputs:
                return inputs

        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {"metadata": {}}

        async def track_async_success(*args, **kwargs):
            pass

        mock_logging_obj.async_success_handler = track_async_success

        guardrail = ModelLevelApplyGuardrail()
        captured_data = {"model": "gpt-4", "metadata": {}}

        def mock_merge(data, llm_router):
            merged = copy.deepcopy(data)
            merged["metadata"]["guardrails"] = ["model-apply-guardrail"]
            merged["_merged_marker"] = True
            return merged

        # Capture what UnifiedLLMGuardrails.async_post_call_success_hook receives
        original_unified_hook = None
        from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
            UnifiedLLMGuardrails,
        )
        original_unified_hook = UnifiedLLMGuardrails.async_post_call_success_hook

        async def tracking_unified_hook(self, user_api_key_dict, data, response):
            nonlocal unified_received_data
            unified_received_data = data
            return response

        with patch("litellm.callbacks", [guardrail]), \
             patch(
                "litellm.proxy.utils._check_and_merge_model_level_guardrails",
                side_effect=mock_merge,
             ), \
             patch.object(
                UnifiedLLMGuardrails,
                "async_post_call_success_hook",
                tracking_unified_hook,
             ):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=captured_data,
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        assert unified_received_data is not None, \
            "UnifiedLLMGuardrails must be called for apply_guardrail guardrails"
        assert unified_received_data.get("_merged_marker") is True, \
            "UnifiedLLMGuardrails must receive guardrail_data (merged), not captured_data"
        assert "model-apply-guardrail" in unified_received_data.get(
            "metadata", {}
        ).get("guardrails", []), \
            "UnifiedLLMGuardrails data must contain model-level guardrails"

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

        with patch("litellm.callbacks", [guardrail_a, guardrail_b]), \
             patch(
                "litellm.proxy.utils._check_and_merge_model_level_guardrails",
                side_effect=mock_merge,
             ):
            await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                captured_data=captured_data,
                captured_user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                captured_logging_obj=mock_logging_obj,
                assembled_response=MagicMock(),
                cache_hit=False,
            )

        for name in ("guardrail-a", "guardrail-b"):
            assert name in received_data_per_guardrail, \
                f"{name} must be called"
            assert received_data_per_guardrail[name].get("_merged_marker") is True, \
                f"{name} must receive guardrail_data (merged), not captured_data"
