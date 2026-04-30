"""
Regression tests for OTel callback handling of non-standard response_obj shapes
and non-string message content.

Covers:
- #24516: response_obj can be a list (Usage AI chat flow)
- #24057: message.content can be list[dict] (multimodal)
- gen_ai.system set to None when custom_llm_provider is explicitly None
"""

import json
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig


class TestOtelNonDictResponseObj(unittest.TestCase):
    """Verify _handle_success does not crash when response_obj is a list."""

    def _make_otel(self, enable_events=False):
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        log_exporter = InMemoryLogExporter()
        logger_provider = OTLoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        config = OpenTelemetryConfig(enable_events=enable_events)
        otel = OpenTelemetry(
            config=config,
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        otel.tracer = tracer_provider.get_tracer(__name__)
        return otel, span_exporter

    def _make_kwargs(self, custom_llm_provider="openai"):
        return {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {
                "custom_llm_provider": custom_llm_provider,
                "proxy_server_request": None,
            },
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
                "hidden_params": {},
            },
        }

    @patch.dict(os.environ, {}, clear=True)
    def test_handle_success_with_list_response_obj(self):
        """response_obj as a list should not raise (Usage AI chat flow)."""
        otel, span_exporter = self._make_otel(enable_events=True)
        kwargs = self._make_kwargs()
        response_obj = [{"role": "assistant", "content": "Hi there"}]

        start = datetime.utcnow()
        end = start + timedelta(seconds=1)

        # Should not raise - covers both _handle_success and _emit_semantic_logs
        otel._handle_success(kwargs, response_obj, start, end)

        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span even with list response_obj")

    @patch.dict(os.environ, {}, clear=True)
    def test_handle_success_with_none_response_obj(self):
        """response_obj as None should not raise."""
        otel, span_exporter = self._make_otel(enable_events=True)
        kwargs = self._make_kwargs()

        start = datetime.utcnow()
        end = start + timedelta(seconds=1)

        otel._handle_success(kwargs, None, start, end)

        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span even with None response_obj")

    @patch.dict(os.environ, {}, clear=True)
    def test_set_attributes_with_list_response_obj(self):
        """set_attributes should not crash when response_obj is a list."""
        otel = OpenTelemetry(config=OpenTelemetryConfig())
        mock_span = MagicMock()
        kwargs = self._make_kwargs()

        # Should not raise
        otel.set_attributes(
            span=mock_span, kwargs=kwargs, response_obj=[{"content": "hi"}]
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_set_attributes_with_none_provider(self):
        """custom_llm_provider=None should fall back to 'Unknown'."""
        otel = OpenTelemetry(config=OpenTelemetryConfig())
        mock_span = MagicMock()
        kwargs = self._make_kwargs(custom_llm_provider=None)
        response_obj = {
            "id": "test-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Verify gen_ai.system is exactly "Unknown", not None or empty
        found_system = False
        for call in mock_span.set_attribute.call_args_list:
            args = call[0] if call[0] else ()
            if len(args) >= 2 and "gen_ai.system" in str(args[0]):
                self.assertEqual(
                    args[1],
                    "Unknown",
                    "gen_ai.system should fall back to 'Unknown' when provider is None",
                )
                found_system = True
        self.assertTrue(found_system, "Expected gen_ai.system attribute to be set")


class TestOtelNonStringContent(unittest.TestCase):
    """Verify multimodal list[dict] content is serialized, not passed raw."""

    @patch.dict(os.environ, {}, clear=True)
    def test_set_attributes_multimodal_content(self):
        """message.content as list[dict] should be serialized to JSON string."""
        otel = OpenTelemetry(config=OpenTelemetryConfig())
        mock_span = MagicMock()

        multimodal_content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]

        kwargs = {
            "model": "gpt-4-vision",
            "messages": [{"role": "user", "content": multimodal_content}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }
        response_obj = {
            "id": "test-id",
            "model": "gpt-4-vision",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "It's a cat.", "role": "assistant"},
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Verify no list was passed as an attribute value, and that content
        # attributes contain valid serialized JSON where applicable
        for call in mock_span.set_attribute.call_args_list:
            args = call[0] if call[0] else ()
            if len(args) >= 2:
                self.assertNotIsInstance(
                    args[1],
                    list,
                    f"Attribute {args[0]} should not be a raw list",
                )
                # If it's a content-related attribute with our multimodal data,
                # verify it's valid JSON
                if "gen_ai.prompt" in str(args[0]) and isinstance(args[1], str):
                    try:
                        parsed = json.loads(args[1])
                        # Should contain our multimodal content structure
                        if isinstance(parsed, list) and len(parsed) > 0:
                            self.assertIn("type", parsed[0])
                    except (json.JSONDecodeError, TypeError):
                        pass  # not all content attrs are JSON


if __name__ == "__main__":
    unittest.main()
