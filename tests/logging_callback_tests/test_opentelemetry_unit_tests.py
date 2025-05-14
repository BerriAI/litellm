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

    def test_no_duplicate_dynamic_span_processors(self):
        """
        Ensure that only one dynamic span processor is added even with multiple calls.
        """
        from litellm.integrations.opentelemetry import OpenTelemetry
        from unittest.mock import MagicMock

        otel = OpenTelemetry()
        otel._get_span_processor = MagicMock()
        provider = MagicMock()
        # Simulate the provider's _active_span_processor._span_processors
        provider._active_span_processor._span_processors = []
        # Patch trace.get_tracer_provider to return our mock provider
        import opentelemetry.trace
        orig_get_tracer_provider = opentelemetry.trace.get_tracer_provider
        opentelemetry.trace.get_tracer_provider = MagicMock(return_value=provider)
        try:
            # First call: should add
            otel._add_dynamic_span_processor_if_needed({
                "standard_callback_dynamic_params": {"arize_space_key": "abc"}
            })
            assert otel._get_span_processor.call_count == 1
            # Simulate processor with _litellm_dynamic already present
            processor = MagicMock()
            processor._litellm_dynamic = True
            provider._active_span_processor._span_processors.append(processor)
            # Second call: should not add
            otel._add_dynamic_span_processor_if_needed({
                "standard_callback_dynamic_params": {"arize_space_key": "abc"}
            })
            assert otel._get_span_processor.call_count == 1
        finally:
            opentelemetry.trace.get_tracer_provider = orig_get_tracer_provider

    def test_no_double_end_on_span(self):
        """
        Ensure that .end() is not called on a span that is not recording.
        """
        from litellm.integrations.opentelemetry import OpenTelemetry
        otel = OpenTelemetry()
        span = MagicMock()
        span.is_recording.return_value = False
        # Should not call .end() if not recording
        otel._handle_sucess({"litellm_params": {"metadata": {"litellm_parent_otel_span": span}}}, {}, datetime.now(), datetime.now())
        span.end.assert_not_called()
