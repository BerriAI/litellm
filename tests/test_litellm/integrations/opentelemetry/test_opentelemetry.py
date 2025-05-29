import json

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
# sys.path.insert(0, os.path.abspath("../../.."))
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

import os
import sys
import unittest
from datetime import datetime, timedelta
import time

# make sure we can import our package
# sys.path.insert(0, os.path.abspath("../../.."))

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, InMemoryLogExporter
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes, event_attributes as EventAttributes

from litellm.integrations.opentelemetry import OpenTelemetry


class TestOpenTelemetryGuardrais(unittest.TestCase):

    @patch("litellm.integrations.opentelemetry.datetime")
    def test_create_guardrail_span_with_valid_info(self, mock_datetime):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()
        mock_span = MagicMock()
        otel.tracer.start_span.return_value = mock_span

        # Create guardrail information
        guardrail_info = {
            "guardrail_name": "test_guardrail",
            "guardrail_mode": "input",
            "masked_entity_count": {"CREDIT_CARD": 2},
            "guardrail_response": "filtered_content",
            "start_time": 1609459200.0,
            "end_time": 1609459201.0,
        }

        # Create a kwargs dict with standard_logging_object containing guardrail information
        kwargs = {"standard_logging_object": {"guardrail_information": guardrail_info}}

        # Call the method
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Assertions
        otel.tracer.start_span.assert_called_once()

        # print all calls to mock_span.set_attribute
        print("Calls to mock_span.set_attribute:")
        for call in mock_span.set_attribute.call_args_list:
            print(call)

        # Check that the span has the correct attributes set
        mock_span.set_attribute.assert_any_call("guardrail_name", "test_guardrail")
        mock_span.set_attribute.assert_any_call("guardrail_mode", "input")
        mock_span.set_attribute.assert_any_call(
            "guardrail_response", "filtered_content"
        )
        mock_span.set_attribute.assert_any_call(
            "masked_entity_count", safe_dumps({"CREDIT_CARD": 2})
        )

        # Verify that the span was ended
        mock_span.end.assert_called_once()


    def test_create_guardrail_span_with_no_info(self):
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Test with no guardrail information
        kwargs = {"standard_logging_object": {}}
        otel._create_guardrail_span(kwargs=kwargs, context=None)

        # Verify that start_span was never called
        otel.tracer.start_span.assert_not_called()


# This test validates LiteLLM integration with OpenTelemetry.
#
# It tests the following telemetry flavors
# 1. spans only (backward-compatible mode)
# 2. spans and metrics, when LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS=true
# 3. spans, metrics and events, when LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS=true and LITELLM_OTEL_INTEGRATION_ENABLE_METRICS=true
#    note: in this case request and response messages will be logged as events, but not on the span attributes.
class TestOpenTelemetry(unittest.TestCase):
    POLL_INTERVAL = 0.05
    POLL_TIMEOUT = 2.0
    MODEL = 'arn:aws:bedrock:us-west-2:1234567890123:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'
    HERE = os.path.dirname(__file__)


    def wait_for_spans(self, exporter: InMemorySpanExporter, prefix: str):
        """Poll until we see at least one span with an attribute key starting with `prefix`."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            spans = exporter.get_finished_spans()
            matches = [
                s for s in spans
                if any(str(k).startswith(prefix) for k in s.attributes)
            ]
            if matches:
                return matches
            time.sleep(self.POLL_INTERVAL)
        return []

    def wait_for_metric(self, reader: InMemoryMetricReader, name: str):
        """Poll until we see a metric with the given name."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            data = reader.get_metrics_data()
            # guard against None or missing attribute
            if not data or not hasattr(data, "resource_metrics"):
                time.sleep(self.POLL_INTERVAL)
                continue

            for rm in data.resource_metrics:
                for sm in rm.scope_metrics:
                    for m in sm.metrics:
                        if m.name == name:
                            return m

            time.sleep(self.POLL_INTERVAL)
        return None

    def wait_for_event(self, exporter: InMemoryLogExporter, event_name: str):
        """Poll until we see at least one event with the given EventAttributes.EVENT_NAME."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            logs = exporter.get_finished_logs()
            matches = [
                l for l in logs
                if l.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
            ]
            if matches:
                return matches
            time.sleep(self.POLL_INTERVAL)
        return []

    def test_handle_success_generates_spans_metrics_and_events(self):
        # force both metrics & events on
        os.environ["LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS"] = "true"
        os.environ["LITELLM_OTEL_INTEGRATION_ENABLE_METRICS"] = "true"

        # ─── build in‐memory OTEL providers/exporters ─────────────────────────────
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

        log_exporter = InMemoryLogExporter()
        logger_provider = OTLoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        event_logger_provider = EventLoggerProvider(logger_provider)

        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        # ─── instantiate our OpenTelemetry logger with test providers ───────────
        otel = OpenTelemetry(
            tracer_provider=tracer_provider,
            event_logger_provider=event_logger_provider,
            meter_provider=meter_provider,
        )

        # OpenTelemetry attempts to set a global tracer provider, which can be set only once.
        # so we hack here to set a local tracer deriver from the provider we created.
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)

        with open(os.path.join(self.HERE, "data", "captured_kwargs.json")) as f:
            kwargs = json.load(f)
        with open(os.path.join(self.HERE, "data", "captured_response.json")) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

    #     # ─── assert spans ────────────────────────────────────────────────────────
        spans = self.wait_for_spans(span_exporter, "gen_ai.")
        self.assertTrue(spans, "Expected at least one gen_ai span")

        # verify our top‐level litellm_request span is present
        names = [s.name for s in spans]
        self.assertIn("litellm_request", names)

        # ─── assert metrics ──────────────────────────────────────────────────────
        duration_metric = self.wait_for_metric(metric_reader, "gen_ai.client.operation.duration")
        self.assertIsNotNone(duration_metric, "duration histogram was not recorded")

        # check that our model attribute made it onto at least one data point
        found_dp = any(
            dp.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == self.MODEL
            for dp in duration_metric.data.data_points
        )
        self.assertTrue(found_dp, "expected gen_ai.request.model attribute on a data point")

        # ─── assert events ───────────────────────────────────────────────────────
        user_events = self.wait_for_event(log_exporter, "gen_ai.user.message")
        choice_events = self.wait_for_event(log_exporter, "gen_ai.choice")
        self.assertTrue(user_events, "did not see a gen_ai.user.message event")
        self.assertTrue(choice_events, "did not see a gen_ai.choice event")

        # check event bodies
        user_body = user_events[0].log_record.body
        choice_body = choice_events[0].log_record.body

        self.assertEqual('What is the capital of France?', user_body.get("content"))
        self.assertEqual("stop", choice_body.get("finish_reason"))

    def test_handle_success_spans_only(self):
        # make sure neither events nor metrics is on
        os.environ.pop("LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS", None)
        os.environ.pop("LITELLM_OTEL_INTEGRATION_ENABLE_METRICS", None)

        # ─── build in‐memory OTEL providers/exporters ─────────────────────────────
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

        # no logs / no metrics
        log_exporter = InMemoryLogExporter()
        logger_provider = OTLoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        # ─── instantiate our OpenTelemetry logger with test providers ───────────
        otel = OpenTelemetry(
            tracer_provider=tracer_provider,
            event_logger_provider=EventLoggerProvider(logger_provider),
            meter_provider=meter_provider,
        )
        # bind our tracer to the test tracer provider (global registration is a no-op after the first time)
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)
        with open(os.path.join(self.HERE, "data", "captured_kwargs.json")) as f:
            kwargs = json.load(f)
        with open(os.path.join(self.HERE, "data", "captured_response.json")) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

        # ─── assert spans only ───────────────────────────────────────────────────
        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span")
        # must have the top‐level litellm_request span
        # self.assertIn(
        #     LITELLM_REQUEST_SPAN_NAME,
        #     [s.name for s in spans],
        #     "litellm_request span missing",
        # )
        # model attribute should be on that span
        found = any(
            s.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == self.MODEL
            for s in spans
        )
        self.assertTrue(found, "expected gen_ai.request.model on span attributes")

        # no metrics recorded
        self.assertIsNone(
            self.wait_for_metric(metric_reader, "gen_ai.client.operation.duration"),
            "Did not expect any metrics",
        )
        # no events emitted
        self.assertFalse(
            self.wait_for_event(log_exporter, "gen_ai.user.message"),
            "Did not expect any events",
        )

    def test_handle_success_spans_and_metrics(self):
        # only metrics on
        os.environ.pop("LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS", None)
        os.environ["LITELLM_OTEL_INTEGRATION_ENABLE_METRICS"] = "true"

        # ─── build in‐memory OTEL providers/exporters ─────────────────────────────
        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

        log_exporter = InMemoryLogExporter()
        logger_provider = OTLoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        # ─── instantiate our OpenTelemetry logger with test providers ───────────
        otel = OpenTelemetry(
            tracer_provider=tracer_provider,
            event_logger_provider=EventLoggerProvider(logger_provider),
            meter_provider=meter_provider,
        )
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)
        with open(os.path.join(self.HERE, "data", "captured_kwargs.json")) as f:
            kwargs = json.load(f)
        with open(os.path.join(self.HERE, "data", "captured_response.json")) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

        # ─── assert spans ────────────────────────────────────────────────────────
        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span")

        # ─── assert metrics ──────────────────────────────────────────────────────
        duration_metric = self.wait_for_metric(metric_reader, "gen_ai.client.operation.duration")
        self.assertIsNotNone(duration_metric, "duration histogram was not recorded")
        # model attribute should be present on a data point
        found_dp = any(
            dp.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == self.MODEL
            for dp in duration_metric.data.data_points
        )
        self.assertTrue(found_dp, "expected gen_ai.request.model attribute on a data point")

        # ─── no events when only metrics enabled ─────────────────────────────────
        self.assertFalse(
            self.wait_for_event(log_exporter, "gen_ai.user.message"),
            "Did not expect any events when only metrics are enabled",
        )