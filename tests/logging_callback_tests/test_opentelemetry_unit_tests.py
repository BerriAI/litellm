# What is this?
## Unit tests for opentelemetry integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from base_test import BaseLoggingCallbackTest
from litellm.types.utils import ModelResponse


class TestOpentelemetryUnitTests(BaseLoggingCallbackTest):
    def test_parallel_tool_calls(self, mock_response_obj: ModelResponse):
        tool_calls = mock_response_obj.choices[0].message.tool_calls
        from litellm.integrations.opentelemetry import OpenTelemetry
        from litellm.proxy._types import SpanAttributes

        kv_pair_dict = OpenTelemetry._tool_calls_kv_pair(tool_calls)

        assert kv_pair_dict == {
            f"{SpanAttributes.LLM_COMPLETIONS.value}.0.function_call.arguments": '{"city": "New York"}',
            f"{SpanAttributes.LLM_COMPLETIONS.value}.0.function_call.name": "get_weather",
            f"{SpanAttributes.LLM_COMPLETIONS.value}.1.function_call.arguments": '{"city": "New York"}',
            f"{SpanAttributes.LLM_COMPLETIONS.value}.1.function_call.name": "get_news",
        }

    @pytest.mark.asyncio
    async def test_opentelemetry_integration(self):
        """
        Unit test to confirm external parent otel spans are NOT ended by LiteLLM.

        External spans (passed via metadata) should be managed by their creators,
        not by LiteLLM. This prevents premature closure of spans from Langfuse,
        user code, or other external observability tools.
        """
        # Reset all callbacks to ensure clean state
        litellm.logging_callback_manager._reset_all_callbacks()

        parent_otel_span = MagicMock()
        litellm.callbacks = ["otel"]

        await litellm.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="Hey!",
            metadata={"litellm_parent_otel_span": parent_otel_span},
        )

        await asyncio.sleep(1)

        # Verify external span was NOT ended by LiteLLM
        # External spans should only be closed by their creators
        parent_otel_span.end.assert_not_called()

    def test_get_span_context_detects_active_span(self):
        """
        Unit test: _get_span_context() should auto-detect active spans from global context.

        Active spans should be automatically detected without explicit metadata
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from litellm.integrations.opentelemetry import OpenTelemetry

        # Setup: Create TracerProvider and tracer
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)

        # Create OpenTelemetry integration
        otel_integration = OpenTelemetry()

        # Act: Create an active span and test detection
        with tracer.start_as_current_span("test_parent") as parent_span:
            parent_span_context = parent_span.get_span_context()

            # Call _get_span_context without explicit parent in metadata
            kwargs = {"litellm_params": {"metadata": {}}}
            detected_context, detected_span = otel_integration._get_span_context(kwargs)

            # Assert: Should detect the active span
            assert (
                detected_span is not None
            ), "Should detect active span from global context"
            assert (
                detected_span is parent_span
            ), "Detected span should be the active parent span"

            detected_span_context = detected_span.get_span_context()
            assert (
                detected_span_context.trace_id == parent_span_context.trace_id
            ), "Detected span should have same trace_id as parent"
            assert (
                detected_span_context.span_id == parent_span_context.span_id
            ), "Detected span should have same span_id as parent"

    def test_record_exception_on_span(self):
        """
        Test that _record_exception_on_span properly records exception information.

        This test verifies that StandardLoggingPayloadErrorInformation is properly
        extracted and set as span attributes using ErrorAttributes constants.
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from litellm.integrations.opentelemetry import OpenTelemetry
        from litellm.integrations._types.open_inference import ErrorAttributes

        # Setup: Create TracerProvider and tracer
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)

        # Create OpenTelemetry integration
        otel_integration = OpenTelemetry()

        # Create a mock span
        mock_span = MagicMock()

        # Create test exception
        test_exception = ValueError("Test error message")

        # Create kwargs with exception and error_information
        kwargs = {
            "exception": test_exception,
            "standard_logging_object": {
                "error_information": {
                    "error_code": "500",
                    "error_class": "ValueError",
                    "llm_provider": "openai",
                    "traceback": "Traceback (most recent call last)...",
                    "error_message": "Test error message",
                },
                "error_str": "Test error message",
            },
        }

        # Act: Record exception on span
        otel_integration._record_exception_on_span(span=mock_span, kwargs=kwargs)

        # Assert: span.record_exception should be called with the exception
        mock_span.record_exception.assert_called_once_with(test_exception)

        # Assert: Error attributes should be set using ErrorAttributes constants
        expected_calls = [
            (ErrorAttributes.ERROR_CODE, "500"),
            (ErrorAttributes.ERROR_TYPE, "ValueError"),
            (ErrorAttributes.ERROR_MESSAGE, "Test error message"),
            (ErrorAttributes.ERROR_LLM_PROVIDER, "openai"),
            (ErrorAttributes.ERROR_STACK_TRACE, "Traceback (most recent call last)..."),
        ]

        # Check that set_attribute was called with expected values
        actual_calls = [call.args for call in mock_span.set_attribute.call_args_list]

        for expected_call in expected_calls:
            assert (
                expected_call in actual_calls
            ), f"Expected set_attribute call {expected_call} not found in actual calls: {actual_calls}"

    def test_record_exception_on_span_with_fallback(self):
        """
        Test that _record_exception_on_span falls back to error_str when error_information is None.
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from litellm.integrations.opentelemetry import OpenTelemetry
        from litellm.integrations._types.open_inference import ErrorAttributes

        # Setup: Create TracerProvider and tracer
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)

        # Create OpenTelemetry integration
        otel_integration = OpenTelemetry()

        # Create a mock span
        mock_span = MagicMock()

        # Create test exception
        test_exception = ValueError("Test error message")

        # Create kwargs without error_information (should fallback to error_str)
        kwargs = {
            "exception": test_exception,
            "standard_logging_object": {
                "error_information": None,
                "error_str": "Fallback error message",
            },
        }

        # Act: Record exception on span
        otel_integration._record_exception_on_span(span=mock_span, kwargs=kwargs)

        # Assert: span.record_exception should be called
        mock_span.record_exception.assert_called_once_with(test_exception)

        # Assert: error.message should be set from error_str using ErrorAttributes constant
        mock_span.set_attribute.assert_called_with(
            ErrorAttributes.ERROR_MESSAGE, "Fallback error message"
        )


class TestGenAiSystemNeverNone:
    """
    Regression pin for #36759.

    `gen_ai.system` previously reached the OTLP exporter as `None` from the
    metrics path (`_record_metrics`) and the semantic-log events path
    (`_emit_semantic_logs`, per-message and per-choice events). The OTLP
    protobuf encoder raises `Invalid type <class 'NoneType'> of value None`
    on every record, which the OTel SDK catches and logs at ERROR level
    with a full stack trace — one per span/metric/event. In production this
    was observed driving CloudWatch Logs ingestion from ~0.05 GB/day to
    100+ GB/day under normal traffic.

    The fix routes the `provider` value through `cast_as_primitive_value_type()`
    (already used by the safe span-attribute path) before it is assigned to
    `gen_ai.system` in the three unguarded call sites. The helper returns
    `""` for `None`, so the OTLP encoder never sees a `None`.
    """

    def _make_otel(self, metrics_disabled: bool = True):
        from litellm.integrations.opentelemetry import OpenTelemetry

        otel = OpenTelemetry()
        # Disable real histogram init / real logger init; the regression
        # is in attribute construction, not in the OTel SDK.
        if metrics_disabled:
            otel._operation_duration_histogram = None
            otel._token_usage_histogram = None
            otel._cost_histogram = None
            otel._time_to_first_token_histogram = None
            otel._time_per_output_token_histogram = None
        return otel

    def test_record_metrics_casts_none_provider_to_empty_string(self):
        from datetime import datetime

        from litellm.integrations.opentelemetry import OpenTelemetry

        otel = self._make_otel(metrics_disabled=False)
        captured: dict = {}

        def fake_record(_duration, attributes=None):
            captured["attributes"] = attributes

        otel._operation_duration_histogram = MagicMock()
        otel._operation_duration_histogram.record = fake_record
        otel._token_usage_histogram = None
        otel._cost_histogram = None

        kwargs = {
            "model": "gpt-4",
            "litellm_params": {"custom_llm_provider": None},
            "standard_logging_object": None,
        }
        response_obj: dict = {"usage": None}
        otel._record_metrics(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        attrs = captured["attributes"]
        assert attrs is not None
        # The regression pin: gen_ai.system must be a primitive, never None.
        # OTLP encoder raises on None and logs a stack trace per record.
        assert attrs["gen_ai.system"] is not None
        assert isinstance(attrs["gen_ai.system"], (str, bool, int, float))
        # And the helper's documented behaviour is the empty string for None.
        assert attrs["gen_ai.system"] == ""

    def test_emit_semantic_logs_casts_none_provider_for_prompt_and_completion_events(self):
        """
        Regression pin for #36759: the per-message (gen_ai.content.prompt)
        and per-choice (gen_ai.content.completion) event paths must route
        the `provider` value through the same `cast_as_primitive_value_type`
        guard as the metrics and span-attribute paths.
        """
        from datetime import datetime

        from litellm.integrations.opentelemetry import OpenTelemetry

        otel = self._make_otel(metrics_disabled=True)
        otel.config.enable_events = True

        captured_attrs: list = []

        def fake_emit(log_record):
            captured_attrs.append(log_record.attributes)

        class _FakeLogger:
            def emit(self, log_record):
                fake_emit(log_record)

        fake_logger_provider = MagicMock()
        fake_logger_provider.get_logger = MagicMock(return_value=_FakeLogger())
        otel._logger_provider = fake_logger_provider

        # Stub SdkLogRecord to a value object so `_emit_semantic_logs`
        # can build one without a real OTel SDK. Accepts and discards
        # keyword args we don't care about (trace_flags, severity_text, ...).
        class _FakeLogRecord:
            def __init__(self, timestamp, trace_id, span_id, severity_number, body, attributes, **kwargs):
                self.timestamp = timestamp
                self.trace_id = trace_id
                self.span_id = span_id
                self.severity_number = severity_number
                self.body = body
                self.attributes = attributes

        otel._otel_log_types = MagicMock(return_value=(_FakeLogRecord, type("S", (), {"INFO": "INFO"})()))

        # The legacy (non-experimental) codepath is the default and the one
        # the per-message + per-choice event branches live on.

        kwargs = {
            "model": "gpt-4",
            "litellm_params": {"custom_llm_provider": None},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        response_obj = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ]
        }
        fake_span = MagicMock()
        fake_span.get_span_context.return_value = MagicMock(trace_id=1, span_id=2)

        otel._emit_semantic_logs(
            kwargs=kwargs, response_obj=response_obj, span=fake_span
        )

        # Every emitted event must carry a primitive `gen_ai.system`,
        # never None. Before the fix the per-message and per-choice
        # events passed the raw `provider` (None) straight through.
        assert captured_attrs, "expected semantic-log events to be emitted"
        for attrs in captured_attrs:
            assert "gen_ai.system" in attrs
            assert attrs["gen_ai.system"] is not None, (
                f"gen_ai.system must not be None; got {attrs!r}"
            )
            assert isinstance(attrs["gen_ai.system"], (str, bool, int, float))
        # And at least one prompt event and one completion event were emitted.
        event_names = {a.get("event_name") for a in captured_attrs}
        assert "gen_ai.content.prompt" in event_names
        assert "gen_ai.content.completion" in event_names
