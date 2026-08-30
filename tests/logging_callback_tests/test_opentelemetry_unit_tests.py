# What is this?
## Unit tests for opentelemetry integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from contextlib import contextmanager
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from base_test import BaseLoggingCallbackTest
from litellm.types.utils import ModelResponse


@contextmanager
def temporary_litellm_redaction(enabled: bool):
    original_flag = litellm.redact_user_api_key_info
    litellm.redact_user_api_key_info = enabled  # test-quality-ok: redaction flag is tested here
    try:
        yield
    finally:
        litellm.redact_user_api_key_info = original_flag  # test-quality-ok: restore test global


@contextmanager
def temporary_litellm_callbacks(callbacks: list[str]):
    original_callbacks = litellm.callbacks
    litellm.callbacks = callbacks  # test-quality-ok: callback registration is tested here
    try:
        yield
    finally:
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.callbacks = original_callbacks  # test-quality-ok: restore test global


class TestOpentelemetryRedaction:
    """Regression tests for user_api_key_info redaction in OTEL spans (#36758)."""

    @pytest.mark.asyncio
    async def test_otel_redacts_user_api_key_metadata_when_flag_enabled(self):
        """When redact_user_api_key_info is True the OTEL callback must not
        emit metadata.user_api_key_* span attributes.  Validates the live
        integration path by monkey-patching safe_set_attribute on the
        OpenTelemetry class and inspecting the attribute keys it receives
        during a real acompletion call."""
        from litellm.integrations.opentelemetry import OpenTelemetry

        litellm.logging_callback_manager._reset_all_callbacks()
        recorded_keys: list[str] = []
        original_set = OpenTelemetry.safe_set_attribute

        def recording_set(self_inner, span, key, value):
            recorded_keys.append(key)
            return original_set(self_inner, span=span, key=key, value=value)

        with (
            temporary_litellm_redaction(True),
            temporary_litellm_callbacks(["otel"]),
            patch.object(OpenTelemetry, "safe_set_attribute", recording_set),
        ):
            await litellm.acompletion(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": "test"}],
                mock_response="ok",
                metadata={
                    "user_api_key_hash": "hashed-secret",
                    "user_api_key_user_id": "uid-123",
                    "user_api_key_user_email": "user@example.com",
                    "user_api_key_team_id": "team-456",
                    "generation_name": "test-gen",
                },
            )
            await asyncio.sleep(1)

        metadata_keys = [k for k in recorded_keys if k.startswith("metadata.")]
        assert len(metadata_keys) > 0, "no metadata attributes were emitted at all"
        assert "metadata.user_api_key_hash" not in recorded_keys
        assert "metadata.user_api_key_user_id" not in recorded_keys
        assert "metadata.user_api_key_user_email" not in recorded_keys
        assert "metadata.user_api_key_team_id" not in recorded_keys

    def test_redact_user_api_key_info_filters_correctly(self):
        """Direct unit test for the redaction function used by the OTEL path."""
        from litellm.litellm_core_utils.redact_messages import (
            redact_user_api_key_info,
        )

        with temporary_litellm_redaction(True):
            metadata = {
                "user_api_key_hash": "secret-hash",
                "user_api_key_user_id": "uid",
                "user_api_key_user_email": "email@test.com",
                "user_api_key_team_id": "team",
                "user_api_key_org_id": "org",
                "user_api_key_alias": "my-key",
                "model": "gpt-5",
                "generation_name": "test",
            }
            result = redact_user_api_key_info(metadata=metadata)
            for key in list(metadata.keys()):
                if key.startswith("user_api_key"):
                    assert key not in result, f"{key} should be redacted"
                else:
                    assert key in result, f"{key} should be preserved"

    def test_redact_user_api_key_info_noop_when_disabled(self):
        """When the flag is off, metadata passes through unchanged."""
        from litellm.litellm_core_utils.redact_messages import (
            redact_user_api_key_info,
        )

        with temporary_litellm_redaction(False):
            metadata = {
                "user_api_key_hash": "secret-hash",
                "model": "gpt-5",
            }
            result = redact_user_api_key_info(metadata=metadata)
            assert result == metadata

    def test_team_attributes_are_not_added_to_auxiliary_spans_when_redacted(self):
        """Auxiliary spans must respect the same user_api_key redaction flag."""
        from litellm.integrations.opentelemetry import OpenTelemetry

        with temporary_litellm_redaction(True):
            otel_logger = OpenTelemetry()
            otel_logger.safe_set_attribute = MagicMock()

            otel_logger._set_team_attributes_on_span(
                span=MagicMock(),
                team_id="team-456",
                team_alias="team-alias",
            )

            otel_logger.safe_set_attribute.assert_not_called()

    def test_set_attributes_redacts_user_api_key_metadata_before_span_export(self):
        """The OTEL attribute path must filter user_api_key_* metadata."""
        from litellm.integrations.opentelemetry import OpenTelemetry

        with temporary_litellm_redaction(True):
            otel_logger = OpenTelemetry()
            otel_logger.safe_set_attribute = MagicMock()
            otel_logger._set_inference_identity_attributes = MagicMock()
            otel_logger._set_service_tier_attributes = MagicMock()
            otel_logger._capture_in_span = MagicMock(return_value=False)

            otel_logger.set_attributes(
                span=MagicMock(),
                kwargs={
                    "litellm_params": {"custom_llm_provider": "openai"},
                    "standard_logging_object": {
                        "call_type": "completion",
                        "metadata": {
                            "user_api_key_hash": "hashed-secret",
                            "user_api_key_user_id": "uid-123",
                            "user_api_key_user_email": "user@example.com",
                            "user_api_key_team_id": "team-456",
                            "generation_name": "test-gen",
                        },
                    },
                },
                response_obj=None,
            )

            attribute_keys = [call.kwargs["key"] for call in otel_logger.safe_set_attribute.call_args_list]

            assert "metadata.generation_name" in attribute_keys
            assert "metadata.user_api_key_hash" not in attribute_keys
            assert "metadata.user_api_key_user_id" not in attribute_keys
            assert "metadata.user_api_key_user_email" not in attribute_keys
            assert "metadata.user_api_key_team_id" not in attribute_keys


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
