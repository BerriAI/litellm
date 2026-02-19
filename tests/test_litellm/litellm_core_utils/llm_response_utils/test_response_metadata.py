"""
Tests for litellm.litellm_core_utils.llm_response_utils.response_metadata

Covers the callback_duration_ms timing metric that flows from the Logging object
through _hidden_params to the x-litellm-callback-duration-ms response header.
"""

import datetime
from unittest.mock import MagicMock

import litellm.litellm_core_utils.llm_response_utils.response_metadata as response_metadata_mod
import litellm.proxy.common_request_processing as common_request_processing_mod
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    ResponseMetadata,
    update_response_metadata,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.utils import ModelResponse


class TestCallbackDurationMs:
    """Tests for the callback_duration_ms metric in ResponseMetadata."""

    def _make_logging_obj(self, callback_duration_ms=None, llm_api_duration_ms=None):
        """Build a minimal mock logging object."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        if llm_api_duration_ms is not None:
            logging_obj.model_call_details["llm_api_duration_ms"] = llm_api_duration_ms
        logging_obj.caching_details = None
        if callback_duration_ms is not None:
            logging_obj.callback_duration_ms = callback_duration_ms
        else:
            # Simulate a Logging object that has no callback_duration_ms
            del logging_obj.callback_duration_ms
        return logging_obj

    def test_callback_duration_ms_set_in_hidden_params(self):
        """When logging_obj has callback_duration_ms, it should appear in _hidden_params."""
        result = ModelResponse()
        logging_obj = self._make_logging_obj(callback_duration_ms=12.3456)

        metadata = ResponseMetadata(result)
        start = datetime.datetime(2025, 1, 1, 0, 0, 0)
        end = datetime.datetime(2025, 1, 1, 0, 0, 1)
        metadata.set_timing_metrics(start, end, logging_obj)
        metadata.apply()

        hidden = result._hidden_params
        assert hidden.get("callback_duration_ms") == 12.3456

    def test_callback_duration_ms_absent_when_not_on_logging_obj(self):
        """When logging_obj lacks callback_duration_ms, hidden_params should not have it."""
        result = ModelResponse()
        logging_obj = self._make_logging_obj(callback_duration_ms=None)

        metadata = ResponseMetadata(result)
        start = datetime.datetime(2025, 1, 1, 0, 0, 0)
        end = datetime.datetime(2025, 1, 1, 0, 0, 1)
        metadata.set_timing_metrics(start, end, logging_obj)
        metadata.apply()

        hidden = result._hidden_params
        assert hidden.get("callback_duration_ms") is None

    def test_update_response_metadata_includes_callback_duration(self):
        """End-to-end: update_response_metadata should propagate callback_duration_ms."""
        result = ModelResponse()
        logging_obj = self._make_logging_obj(
            callback_duration_ms=5.5, llm_api_duration_ms=800.0
        )
        logging_obj._response_cost_calculator = MagicMock(return_value=0.001)
        logging_obj.litellm_call_id = "test-call-id"

        start = datetime.datetime(2025, 1, 1, 0, 0, 0)
        end = datetime.datetime(2025, 1, 1, 0, 0, 1)

        update_response_metadata(
            result=result,
            logging_obj=logging_obj,
            model="gpt-4",
            kwargs={},
            start_time=start,
            end_time=end,
        )

        hidden = result._hidden_params
        assert hidden.get("callback_duration_ms") == 5.5
        # overhead should also be set
        assert hidden.get("litellm_overhead_time_ms") is not None


class TestCallbackDurationInCustomHeaders:
    """Test that callback_duration_ms flows into get_custom_headers."""

    def test_header_present_when_callback_duration_in_hidden_params(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        hidden_params = {
            "_response_ms": 1000.0,
            "litellm_overhead_time_ms": 50.0,
            "callback_duration_ms": 7.25,
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            hidden_params=hidden_params,
        )

        assert "x-litellm-callback-duration-ms" in headers
        assert headers["x-litellm-callback-duration-ms"] == "7.25"

    def test_header_absent_when_no_callback_duration(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        hidden_params = {
            "_response_ms": 1000.0,
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            hidden_params=hidden_params,
        )

        # Should be excluded because value is "None" which is in exclude_values
        assert "x-litellm-callback-duration-ms" not in headers


class TestDetailedTiming:
    """Tests for detailed per-phase timing headers behind LITELLM_DETAILED_TIMING."""

    def _make_logging_obj(
        self,
        llm_api_duration_ms=500.0,
        message_copy_duration_ms=2.5,
        api_call_start_time=None,
    ):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "llm_api_duration_ms": llm_api_duration_ms,
        }
        if api_call_start_time is not None:
            logging_obj.model_call_details["api_call_start_time"] = api_call_start_time
        logging_obj.caching_details = None
        logging_obj.callback_duration_ms = 1.0
        logging_obj.message_copy_duration_ms = message_copy_duration_ms
        return logging_obj

    def test_detailed_timing_headers_present_when_enabled(self, monkeypatch):
        """When LITELLM_DETAILED_TIMING is true, detailed timing keys appear in hidden_params."""
        monkeypatch.setattr(response_metadata_mod, "LITELLM_DETAILED_TIMING", True)

        result = ModelResponse()
        start = datetime.datetime(2025, 1, 1, 0, 0, 0)
        api_call_start = datetime.datetime(2025, 1, 1, 0, 0, 0, 20000)  # +20ms
        end = datetime.datetime(2025, 1, 1, 0, 0, 0, 530000)  # +530ms total

        logging_obj = self._make_logging_obj(
            llm_api_duration_ms=500.0,
            message_copy_duration_ms=2.5,
            api_call_start_time=api_call_start,
        )

        metadata = ResponseMetadata(result)
        metadata.set_timing_metrics(start, end, logging_obj)
        metadata.apply()

        hidden = result._hidden_params
        assert hidden.get("timing_llm_api_ms") == 500.0
        assert hidden.get("timing_message_copy_ms") == 2.5
        assert hidden.get("timing_pre_processing_ms") == 20.0
        assert hidden.get("timing_post_processing_ms") == 10.0  # 530 - 20 - 500

    def test_detailed_timing_absent_when_disabled(self, monkeypatch):
        """When LITELLM_DETAILED_TIMING is false, no detailed timing keys."""
        monkeypatch.setattr(response_metadata_mod, "LITELLM_DETAILED_TIMING", False)

        result = ModelResponse()
        start = datetime.datetime(2025, 1, 1, 0, 0, 0)
        end = datetime.datetime(2025, 1, 1, 0, 0, 1)
        logging_obj = self._make_logging_obj()

        metadata = ResponseMetadata(result)
        metadata.set_timing_metrics(start, end, logging_obj)
        metadata.apply()

        hidden = result._hidden_params
        assert hidden.get("timing_llm_api_ms") is None
        assert hidden.get("timing_pre_processing_ms") is None

    def test_detailed_timing_headers_in_custom_headers(self, monkeypatch):
        """When LITELLM_DETAILED_TIMING is true, headers flow to get_custom_headers."""
        monkeypatch.setattr(common_request_processing_mod, "LITELLM_DETAILED_TIMING", True)

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        hidden_params = {
            "_response_ms": 530.0,
            "timing_llm_api_ms": 500.0,
            "timing_pre_processing_ms": 20.0,
            "timing_post_processing_ms": 10.0,
            "timing_message_copy_ms": 2.5,
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            hidden_params=hidden_params,
        )

        assert headers["x-litellm-timing-llm-api-ms"] == "500.0"
        assert headers["x-litellm-timing-pre-processing-ms"] == "20.0"
        assert headers["x-litellm-timing-post-processing-ms"] == "10.0"
        assert headers["x-litellm-timing-message-copy-ms"] == "2.5"

    def test_detailed_timing_headers_absent_when_disabled(self, monkeypatch):
        """When LITELLM_DETAILED_TIMING is false, no timing headers emitted."""
        monkeypatch.setattr(common_request_processing_mod, "LITELLM_DETAILED_TIMING", False)

        user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
        hidden_params = {
            "_response_ms": 530.0,
            "timing_llm_api_ms": 500.0,
        }

        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            hidden_params=hidden_params,
        )

        assert "x-litellm-timing-llm-api-ms" not in headers
        assert "x-litellm-timing-pre-processing-ms" not in headers


class TestLoggingInitCallbackDuration:
    """Test that Logging.__init__ tracks deep copy time in callback_duration_ms."""

    def test_logging_init_sets_callback_duration_ms(self):
        obj = Logging(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello " * 100}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.datetime.now(),
            litellm_call_id="test-123",
            function_id="func-123",
        )

        # callback_duration_ms should be set and non-negative
        assert hasattr(obj, "callback_duration_ms")
        assert obj.callback_duration_ms >= 0

    def test_logging_init_callback_duration_zero_for_none_messages(self):
        obj = Logging(
            model="gpt-4",
            messages=None,
            stream=False,
            call_type="acompletion",
            start_time=datetime.datetime.now(),
            litellm_call_id="test-456",
            function_id="func-456",
        )

        # Should still be set (deep copy of None is essentially a no-op)
        assert hasattr(obj, "callback_duration_ms")
        assert obj.callback_duration_ms >= 0
