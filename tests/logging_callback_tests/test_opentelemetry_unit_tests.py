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
        Unit test to confirm the parent otel span is ended
        """

        parent_otel_span = MagicMock()
        litellm.callbacks = ["otel"]

        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="Hey!",
            metadata={"litellm_parent_otel_span": parent_otel_span},
        )

        await asyncio.sleep(1)

        parent_otel_span.end.assert_called_once()

    def test_init_tracing_respects_existing_tracer_provider(self):
        """
        Unit test: _init_tracing() should respect existing TracerProvider.

        When a TracerProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from litellm.integrations.opentelemetry import OpenTelemetry

        # Setup: Create and set an existing TracerProvider
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        existing_provider = trace.get_tracer_provider()

        # Act: Initialize OpenTelemetry integration (should detect existing provider)
        otel_integration = OpenTelemetry()

        # Assert: The existing provider should still be active
        current_provider = trace.get_tracer_provider()
        assert current_provider is existing_provider, (
            "Existing TracerProvider should be respected and not overridden"
        )

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
