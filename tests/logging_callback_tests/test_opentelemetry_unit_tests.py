# What is this?
## Unit tests for opentelemetry integration

# What is this?
## Unit test for presidio pii masking
import sys
from dotenv import load_dotenv

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import os
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm

from unittest.mock import MagicMock, patch
from base_test import BaseLoggingCallbackTest
from litellm.types.utils import ModelResponse


class TestOpentelemetryUnitTests(BaseLoggingCallbackTest):
    def test_parallel_tool_calls(self, mock_response_obj: ModelResponse):
        tool_calls = mock_response_obj.choices[0].message.tool_calls
        from litellm.integrations.opentelemetry import OpenTelemetry
        from litellm.proxy._types import SpanAttributes

        kv_pair_dict = OpenTelemetry._tool_calls_kv_pair(tool_calls)

        assert kv_pair_dict == {
            f"{SpanAttributes.LLM_COMPLETIONS}.0.function_call.arguments": '{"city": "New York"}',
            f"{SpanAttributes.LLM_COMPLETIONS}.0.function_call.name": "get_weather",
            f"{SpanAttributes.LLM_COMPLETIONS}.1.function_call.arguments": '{"city": "New York"}',
            f"{SpanAttributes.LLM_COMPLETIONS}.1.function_call.name": "get_news",
        }

    @patch("opentelemetry.trace")
    def test_sets_tracer_provider_when_none_exists(self, mock_trace):
        mock_trace.get_tracer_provider.return_value = None

        OpenTelemetry(config=OpenTelemetryConfig())

        mock_trace.set_tracer_provider.assert_called_once()

    @patch("opentelemetry.trace")
    def test_does_not_override_existing_tracer_provider(self, mock_trace):
        existing_tracer_provider = MagicMock()
        mock_trace.get_tracer_provider.return_value = existing_tracer_provider

        OpenTelemetry(config=OpenTelemetryConfig())

        mock_trace.set_tracer_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_opentelemetry_integration(self):
        """
        Unit test to confirm the parent otel span is ended
        """

        load_dotenv()

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


class TestOpenTelemetryConfigUnitTests:

    @pytest.mark.parametrize(
        "name, env_vars, expected",
        [
            (
                "default",
                {},
                OpenTelemetryConfig(exporter="console"),
            ),
            (
                "OTEL_ENDPOINT -> endpoint",
                {
                    "OTEL_EXPORTER": "otlp_http",
                    "OTEL_ENDPOINT": "http://localhost:4318/v1/traces"
                },
                OpenTelemetryConfig(exporter="otlp_http", endpoint="http://localhost:4318/v1/traces"),
            ),
            (
                "OTEL_EXPORTER=in_memory -> exporter=InMemorySpanExporter",
                {"OTEL_EXPORTER": "in_memory"},
                OpenTelemetryConfig(exporter=InMemorySpanExporter),
            ),
            (
                "OTEL_HEADERS -> headers",
                {
                    "OTEL_HEADERS": "Authorization=Bearer token123"
                },
                OpenTelemetryConfig(exporter="console", headers="Authorization=Bearer token123"),
            ),
            (
                "DEBUG_OTEL=TrUe -> debug=true",
                {"DEBUG_OTEL": "TrUe"},
                OpenTelemetryConfig(exporter="console", debug="true"),
            ),
        ],
    )
    def test_env_variable_prioritization(self, name, monkeypatch, env_vars, expected):
        # Clear all environment variables
        for var in os.environ:
            monkeypatch.delenv(var, raising=False)
        # Set test-specific environment variables
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        # Call the method under test
        config = OpenTelemetryConfig.from_env()

        # Validate the results
        if isinstance(expected.exporter, type):
            assert isinstance(config.exporter, expected.exporter)
        else:
            assert config.exporter == expected.exporter

        assert config.endpoint == expected.endpoint
        assert config.headers == expected.headers
