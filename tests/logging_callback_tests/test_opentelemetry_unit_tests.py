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
            model="gpt-3.5-turbo",
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
            assert detected_span is not None, "Should detect active span from global context"
            assert detected_span is parent_span, "Detected span should be the active parent span"

            detected_span_context = detected_span.get_span_context()
            assert detected_span_context.trace_id == parent_span_context.trace_id, (
                "Detected span should have same trace_id as parent"
            )
            assert detected_span_context.span_id == parent_span_context.span_id, (
                "Detected span should have same span_id as parent"
            )

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
            assert expected_call in actual_calls, (
                f"Expected set_attribute call {expected_call} not found in actual calls: {actual_calls}"
            )

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
        mock_span.set_attribute.assert_called_with(ErrorAttributes.ERROR_MESSAGE, "Fallback error message")
