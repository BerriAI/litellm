"""
Unit tests for LIT-3314: non-streaming v1/messages overhead optimizations.

Covers the five optimization areas:
  1. _CallbackCapabilities: new flags has_post_call_guardrail,
     has_pre_request_hook, has_websearch_interceptor
  2. _has_post_call_guardrails() uses the capabilities cache
  3. _is_streaming_response() module-level imports (no inline imports)
  4. transform_anthropic_messages_response() short-circuit (cast, no copy)
  5. _has_agentic_completion_hook() fast empty-list exit
"""

import sys
import os
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks

# ---------------------------------------------------------------------------
# Guardrail helpers
# ---------------------------------------------------------------------------


class _PostCallGuardrail(CustomGuardrail):
    def __init__(self):
        super().__init__(
            guardrail_name="post-call",
            default_on=True,
            event_hook=GuardrailEventHooks.post_call,
        )


class _PreCallGuardrail(CustomGuardrail):
    def __init__(self):
        super().__init__(
            guardrail_name="pre-call",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )


class _AllEventsGuardrail(CustomGuardrail):
    def __init__(self):
        super().__init__(
            guardrail_name="all-events",
            default_on=True,
            event_hook=None,  # matches all events — NOT a post_call explicit registration
        )


class _ListGuardrailWithPostCall(CustomGuardrail):
    def __init__(self):
        super().__init__(
            guardrail_name="list-post",
            default_on=True,
            event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call],
        )


# ---------------------------------------------------------------------------
# CustomLogger with async_pre_request_hook override
# ---------------------------------------------------------------------------


class _PreRequestHookLogger(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: List, kwargs: Dict
    ) -> Optional[Dict]:
        return None


# ---------------------------------------------------------------------------
# Logger that inherits async_pre_request_hook from _PreRequestHookLogger
# without re-declaring it in its own __dict__.  Used to verify MRO detection.
# ---------------------------------------------------------------------------


class _InheritedPreRequestHookLogger(_PreRequestHookLogger):
    """Inherits async_pre_request_hook from parent — not in own __dict__."""

    pass


# ---------------------------------------------------------------------------
# WebSearch interceptor duck-type (has try_short_circuit_search)
# ---------------------------------------------------------------------------


class _WebSearchLogger(CustomLogger):
    async def try_short_circuit_search(self, **kwargs):
        return None


# ---------------------------------------------------------------------------
# 1. _CallbackCapabilities — new flag detection
# ---------------------------------------------------------------------------


class TestCallbackCapabilitiesNewFlags:
    """Verify the three new flags added to _CallbackCapabilities."""

    def _caps(self, callbacks):
        ProxyLogging._callback_capabilities_cache.clear()
        with patch("litellm.callbacks", callbacks):
            return ProxyLogging._callback_capabilities()

    # has_post_call_guardrail ---------------------------------------------------

    def test_has_post_call_guardrail_true(self):
        caps = self._caps([_PostCallGuardrail()])
        assert caps.has_post_call_guardrail is True

    def test_has_post_call_guardrail_list_event_hook(self):
        caps = self._caps([_ListGuardrailWithPostCall()])
        assert caps.has_post_call_guardrail is True

    def test_has_post_call_guardrail_false_for_pre_call(self):
        caps = self._caps([_PreCallGuardrail()])
        assert caps.has_post_call_guardrail is False

    def test_has_post_call_guardrail_false_for_event_hook_none(self):
        """event_hook=None should NOT set has_post_call_guardrail."""
        caps = self._caps([_AllEventsGuardrail()])
        assert caps.has_post_call_guardrail is False

    def test_has_post_call_guardrail_false_for_empty(self):
        caps = self._caps([])
        assert caps.has_post_call_guardrail is False

    def test_has_post_call_guardrail_false_for_non_guardrail_logger(self):
        caps = self._caps([CustomLogger()])
        assert caps.has_post_call_guardrail is False

    # has_pre_request_hook -----------------------------------------------------

    def test_has_pre_request_hook_true(self):
        caps = self._caps([_PreRequestHookLogger()])
        assert caps.has_pre_request_hook is True

    def test_has_pre_request_hook_false_for_base_logger(self):
        """Base CustomLogger does not override async_pre_request_hook."""
        caps = self._caps([CustomLogger()])
        assert caps.has_pre_request_hook is False

    def test_has_pre_request_hook_false_for_empty(self):
        caps = self._caps([])
        assert caps.has_pre_request_hook is False

    def test_has_pre_request_hook_true_for_inherited_hook(self):
        """MRO traversal must detect hooks inherited from an intermediate parent."""
        caps = self._caps([_InheritedPreRequestHookLogger()])
        assert caps.has_pre_request_hook is True

    # has_websearch_interceptor ------------------------------------------------

    def test_has_websearch_interceptor_true(self):
        caps = self._caps([_WebSearchLogger()])
        assert caps.has_websearch_interceptor is True

    def test_has_websearch_interceptor_false_for_plain_logger(self):
        caps = self._caps([CustomLogger()])
        assert caps.has_websearch_interceptor is False

    def test_has_websearch_interceptor_false_for_empty(self):
        caps = self._caps([])
        assert caps.has_websearch_interceptor is False


# ---------------------------------------------------------------------------
# 2. _has_post_call_guardrails() uses the capabilities cache
# ---------------------------------------------------------------------------


class TestHasPostCallGuardrailsUsesCache:
    """
    _has_post_call_guardrails() must return the same truth as the old
    per-loop implementation AND must not iterate litellm.callbacks
    directly (it delegates to ProxyLogging._callback_capabilities()).
    """

    def _check(self, callbacks, expected: bool):
        ProxyLogging._callback_capabilities_cache.clear()
        with patch("litellm.callbacks", callbacks):
            result = ProxyBaseLLMRequestProcessing._has_post_call_guardrails()
        assert result is expected

    def test_post_call_guardrail(self):
        self._check([_PostCallGuardrail()], True)

    def test_pre_call_only(self):
        self._check([_PreCallGuardrail()], False)

    def test_event_hook_none(self):
        self._check([_AllEventsGuardrail()], False)

    def test_empty(self):
        self._check([], False)

    def test_non_guardrail_callbacks_ignored(self):
        self._check([CustomLogger()], False)

    def test_list_event_hook_with_post_call(self):
        self._check([_ListGuardrailWithPostCall()], True)

    def test_delegates_to_cache_not_direct_loop(self):
        """Verify _has_post_call_guardrails uses ProxyLogging._callback_capabilities()."""
        ProxyLogging._callback_capabilities_cache.clear()
        with patch("litellm.callbacks", [_PostCallGuardrail()]):
            with patch.object(
                ProxyLogging,
                "_callback_capabilities",
                wraps=ProxyLogging._callback_capabilities,
            ) as mock_caps:
                ProxyBaseLLMRequestProcessing._has_post_call_guardrails()
                mock_caps.assert_called_once()


# ---------------------------------------------------------------------------
# 3. _is_streaming_response() — no inline imports
# ---------------------------------------------------------------------------


class TestIsStreamingResponseNoInlineImports:
    """
    The optimised _is_streaming_response() must use module-level imports
    (inspect, _AsyncIterator, _AsyncGenerator) rather than importing inside
    the method body.
    """

    def test_inspect_is_module_level(self):
        import inspect as _inspect
        import litellm.proxy.common_request_processing as crp

        assert crp.inspect is _inspect

    def test_async_iterator_is_module_level(self):
        from collections.abc import AsyncIterator
        import litellm.proxy.common_request_processing as crp

        assert crp._AsyncIterator is AsyncIterator

    def test_async_generator_is_module_level(self):
        from collections.abc import AsyncGenerator
        import litellm.proxy.common_request_processing as crp

        assert crp._AsyncGenerator is AsyncGenerator

    def test_async_gen_detected(self):
        proc = ProxyBaseLLMRequestProcessing(data={})

        async def _gen():
            yield 1

        assert proc._is_streaming_response(_gen()) is True

    def test_sync_response_not_detected(self):
        proc = ProxyBaseLLMRequestProcessing(data={})
        assert proc._is_streaming_response({"type": "message"}) is False

    def test_async_iterator_detected(self):
        from collections.abc import AsyncIterator as _AI

        class _FakeIter(_AI):
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        proc = ProxyBaseLLMRequestProcessing(data={})
        assert proc._is_streaming_response(_FakeIter()) is True


# ---------------------------------------------------------------------------
# 4. transform_anthropic_messages_response — short-circuit (cast, no copy)
# ---------------------------------------------------------------------------


class TestTransformAnthropicMessagesResponseShortCircuit:
    """
    The optimised transform must not create an unnecessary dict copy.
    The returned object IS the same dict that raw_response.json() returns.
    """

    def _make_raw_response(self, payload: dict):
        resp = MagicMock()
        resp.json.return_value = payload
        resp.text = ""
        resp.status_code = 200
        return resp

    def test_returns_same_dict_object(self):
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        payload = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        config = AnthropicMessagesConfig()
        raw_resp = self._make_raw_response(payload)
        result = config.transform_anthropic_messages_response(
            model="claude-test",
            raw_response=raw_resp,
            logging_obj=MagicMock(),
        )
        # Short-circuit: the returned object is the SAME dict (no copy)
        assert result is payload

    def test_returns_correct_fields(self):
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        payload = {
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
        config = AnthropicMessagesConfig()
        result = config.transform_anthropic_messages_response(
            model="claude-3",
            raw_response=self._make_raw_response(payload),
            logging_obj=MagicMock(),
        )
        assert result["id"] == "msg_x"
        assert result["content"][0]["text"] == "hello"

    def test_raises_on_json_parse_failure(self):
        from litellm.llms.anthropic.common_utils import AnthropicError
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        resp = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        resp.text = "bad response"
        resp.status_code = 200

        config = AnthropicMessagesConfig()
        with pytest.raises(AnthropicError):
            config.transform_anthropic_messages_response(
                model="claude-test",
                raw_response=resp,
                logging_obj=MagicMock(),
            )


# ---------------------------------------------------------------------------
# 5. _has_agentic_completion_hook — fast empty-list exit
# ---------------------------------------------------------------------------


class TestHasAgenticCompletionHookFastPath:
    """
    When both litellm.callbacks and dynamic_success_callbacks are empty,
    _has_agentic_completion_hook must return False without building a
    concatenated list.
    """

    def test_returns_false_for_empty_static_and_no_dynamic(self):
        with patch("litellm.callbacks", []):
            logging_obj = MagicMock()
            logging_obj.dynamic_success_callbacks = None
            from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

            assert BaseLLMHTTPHandler._has_agentic_completion_hook(logging_obj) is False

    def test_returns_false_for_empty_static_and_empty_dynamic(self):
        with patch("litellm.callbacks", []):
            logging_obj = MagicMock()
            logging_obj.dynamic_success_callbacks = []
            from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

            assert BaseLLMHTTPHandler._has_agentic_completion_hook(logging_obj) is False

    def test_returns_false_when_no_override(self):
        """Base CustomLogger does not override async_should_run_agentic_loop."""
        with patch("litellm.callbacks", [CustomLogger()]):
            logging_obj = MagicMock()
            logging_obj.dynamic_success_callbacks = None
            from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

            assert BaseLLMHTTPHandler._has_agentic_completion_hook(logging_obj) is False

    def test_returns_true_when_override_present(self):
        class _AgenticLogger(CustomLogger):
            async def async_should_run_agentic_loop(self, *args, **kwargs):
                return False, {}

        with patch("litellm.callbacks", [_AgenticLogger()]):
            logging_obj = MagicMock()
            logging_obj.dynamic_success_callbacks = None
            from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

            assert BaseLLMHTTPHandler._has_agentic_completion_hook(logging_obj) is True


# ---------------------------------------------------------------------------
# 6. Handler module-level capability cache
# ---------------------------------------------------------------------------


class TestHandlerCapabilityCache:
    """
    The _handler_capability_cache in messages/handler.py should correctly
    detect pre_request_hook and websearch interceptors.
    """

    def test_no_hooks_with_empty_callbacks(self):
        with patch("litellm.callbacks", []):
            from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
                _get_handler_capabilities,
                _handler_capability_cache,
            )

            _handler_capability_cache.clear()
            has_pre, has_ws = _get_handler_capabilities()
        assert has_pre is False
        assert has_ws is False

    def test_detects_pre_request_hook(self):
        with patch("litellm.callbacks", [_PreRequestHookLogger()]):
            from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
                _get_handler_capabilities,
                _handler_capability_cache,
            )

            _handler_capability_cache.clear()
            has_pre, has_ws = _get_handler_capabilities()
        assert has_pre is True
        assert has_ws is False

    def test_detects_websearch_interceptor(self):
        with patch("litellm.callbacks", [_WebSearchLogger()]):
            from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
                _get_handler_capabilities,
                _handler_capability_cache,
            )

            _handler_capability_cache.clear()
            has_pre, has_ws = _get_handler_capabilities()
        assert has_pre is False
        assert has_ws is True

    def test_detects_inherited_pre_request_hook(self):
        """MRO traversal must detect hooks inherited from intermediate parents."""
        with patch("litellm.callbacks", [_InheritedPreRequestHookLogger()]):
            from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
                _get_handler_capabilities,
                _handler_capability_cache,
            )

            _handler_capability_cache.clear()
            has_pre, has_ws = _get_handler_capabilities()
        assert has_pre is True
        assert has_ws is False

    def test_cache_hit_on_same_callbacks(self):
        cb = CustomLogger()
        with patch("litellm.callbacks", [cb]):
            from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
                _get_handler_capabilities,
                _handler_capability_cache,
            )

            _handler_capability_cache.clear()
            result1 = _get_handler_capabilities()
            result2 = _get_handler_capabilities()
        assert result1 is result2  # same tuple object from cache


# ---------------------------------------------------------------------------
# 7. ProxyLogging static accessors for new flags
# ---------------------------------------------------------------------------


class TestProxyLoggingNewAccessors:
    def _clear_and_patch(self, callbacks):
        ProxyLogging._callback_capabilities_cache.clear()
        return patch("litellm.callbacks", callbacks)

    def test_has_post_call_guardrail_callbacks(self):
        with self._clear_and_patch([_PostCallGuardrail()]):
            assert ProxyLogging.has_post_call_guardrail_callbacks() is True

    def test_has_pre_request_hook_callbacks(self):
        with self._clear_and_patch([_PreRequestHookLogger()]):
            assert ProxyLogging.has_pre_request_hook_callbacks() is True

    def test_has_websearch_interceptor_callbacks(self):
        with self._clear_and_patch([_WebSearchLogger()]):
            assert ProxyLogging.has_websearch_interceptor_callbacks() is True

    def test_all_false_for_empty(self):
        with self._clear_and_patch([]):
            assert ProxyLogging.has_post_call_guardrail_callbacks() is False
            assert ProxyLogging.has_pre_request_hook_callbacks() is False
            assert ProxyLogging.has_websearch_interceptor_callbacks() is False
