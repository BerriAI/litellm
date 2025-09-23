import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import time

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, InMemoryLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class TestOpenTelemetryGuardrails(unittest.TestCase):
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


class TestOpenTelemetry(unittest.TestCase):
    POLL_INTERVAL = 0.05
    POLL_TIMEOUT = 2.0
    MODEL = "arn:aws:bedrock:us-west-2:1234567890123:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    HERE = os.path.dirname(__file__)

    def wait_for_spans(self, exporter: InMemorySpanExporter, prefix: str):
        """Poll until we see at least one span with an attribute key starting with `prefix`."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            spans = exporter.get_finished_spans()
            matches = [
                s
                for s in spans
                if s.attributes and any(str(k).startswith(prefix) for k in s.attributes)
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

    def wait_for_log(self, reader: InMemoryLogExporter, name: str):
        """Poll until we see a log with the given name."""
        deadline = time.time() + self.POLL_TIMEOUT
        while time.time() < deadline:
            logs = reader.get_finished_logs()
            if not logs:
                time.sleep(self.POLL_INTERVAL)
                continue
            matches = [
                log
                for log in logs
                # if log.attributes and any(str(k).startswith(prefix) for k in log.attributes)
            ]
            if matches:
                return matches
            time.sleep(self.POLL_INTERVAL)
        return []

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

    def test_get_tracer_to_use_for_request_with_dynamic_headers(self):
        """Test that get_tracer_to_use_for_request returns a dynamic tracer when dynamic headers are present."""
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Mock the dynamic header extraction and tracer creation
        with patch.object(
            otel, "_get_dynamic_otel_headers_from_kwargs"
        ) as mock_get_headers, patch.object(
            otel, "_get_tracer_with_dynamic_headers"
        ) as mock_get_tracer:
            # Test case 1: With dynamic headers
            mock_get_headers.return_value = {
                "arize-space-id": "test-space",
                "api_key": "test-key",
            }
            mock_dynamic_tracer = MagicMock()
            mock_get_tracer.return_value = mock_dynamic_tracer

            kwargs = {
                "standard_callback_dynamic_params": {"arize_space_key": "test-space"}
            }
            result = otel.get_tracer_to_use_for_request(kwargs)

            # Assertions
            mock_get_headers.assert_called_once_with(kwargs)
            mock_get_tracer.assert_called_once_with(
                {"arize-space-id": "test-space", "api_key": "test-key"}
            )
            self.assertEqual(result, mock_dynamic_tracer)

    def test_get_tracer_to_use_for_request_without_dynamic_headers(self):
        """Test that get_tracer_to_use_for_request returns the default tracer when no dynamic headers are present."""
        # Setup
        otel = OpenTelemetry()
        otel.tracer = MagicMock()

        # Mock the dynamic header extraction to return None
        with patch.object(
            otel, "_get_dynamic_otel_headers_from_kwargs"
        ) as mock_get_headers:
            mock_get_headers.return_value = None

            kwargs = {}
            result = otel.get_tracer_to_use_for_request(kwargs)

            # Assertions
            mock_get_headers.assert_called_once_with(kwargs)
            self.assertEqual(result, otel.tracer)

    def test_get_dynamic_otel_headers_from_kwargs(self):
        """Test that _get_dynamic_otel_headers_from_kwargs correctly extracts dynamic headers from kwargs."""
        # Setup
        otel = OpenTelemetry()

        # Mock the construct_dynamic_otel_headers method
        with patch.object(otel, "construct_dynamic_otel_headers") as mock_construct:
            # Test case 1: With standard_callback_dynamic_params
            mock_construct.return_value = {
                "arize-space-id": "test-space",
                "api_key": "test-key",
            }

            standard_params = {
                "arize_space_key": "test-space",
                "arize_api_key": "test-key",
            }
            kwargs = {"standard_callback_dynamic_params": standard_params}

            result = otel._get_dynamic_otel_headers_from_kwargs(kwargs)

            # Assertions
            mock_construct.assert_called_once_with(
                standard_callback_dynamic_params=standard_params
            )
            self.assertEqual(
                result, {"arize-space-id": "test-space", "api_key": "test-key"}
            )

            # Test case 2: Without standard_callback_dynamic_params
            kwargs_empty = {}
            result_empty = otel._get_dynamic_otel_headers_from_kwargs(kwargs_empty)

            # Should return None when no dynamic params
            self.assertIsNone(result_empty)

            # Test case 3: With empty construct result
            mock_construct.return_value = {}
            result_empty_construct = otel._get_dynamic_otel_headers_from_kwargs(kwargs)

            # Should return None when construct returns empty dict
            self.assertIsNone(result_empty_construct)

    @patch("opentelemetry.sdk.trace.TracerProvider")
    @patch("opentelemetry.sdk.resources.Resource")
    def test_get_tracer_with_dynamic_headers(self, mock_resource, mock_tracer_provider):
        """Test that _get_tracer_with_dynamic_headers creates a temporary tracer with dynamic headers."""
        # Setup
        otel = OpenTelemetry()

        # Mock the span processor creation
        with patch.object(otel, "_get_span_processor") as mock_get_span_processor:
            mock_span_processor = MagicMock()
            mock_get_span_processor.return_value = mock_span_processor

            # Mock the tracer provider and its methods
            mock_provider_instance = MagicMock()
            mock_tracer_provider.return_value = mock_provider_instance
            mock_tracer = MagicMock()
            mock_provider_instance.get_tracer.return_value = mock_tracer

            # Mock the resource
            mock_resource_instance = MagicMock()
            mock_resource.return_value = mock_resource_instance

            # Test
            dynamic_headers = {"arize-space-id": "test-space", "api_key": "test-key"}
            result = otel._get_tracer_with_dynamic_headers(dynamic_headers)

            # Assertions
            mock_get_span_processor.assert_called_once_with(
                dynamic_headers=dynamic_headers
            )
            mock_provider_instance.add_span_processor.assert_called_once_with(
                mock_span_processor
            )
            mock_provider_instance.get_tracer.assert_called_once_with("litellm")
            self.assertEqual(result, mock_tracer)

    @patch.dict(os.environ, {}, clear=True)
    @patch("opentelemetry.sdk.resources.Resource.create")
    @patch("opentelemetry.sdk.resources.OTELResourceDetector")
    def test_get_litellm_resource_with_defaults(
        self, mock_detector_cls, mock_resource_create
    ):
        """Test _get_litellm_resource with default values when no environment variables are set."""
        from litellm.integrations.opentelemetry import _get_litellm_resource

        # Mock the Resource.create method
        mock_base_resource = MagicMock()
        mock_resource_create.return_value = mock_base_resource

        # Mock the OTELResourceDetector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_env_resource = MagicMock()
        mock_detector.detect.return_value = mock_env_resource

        # Mock the merged resource
        mock_merged_resource = MagicMock()
        mock_base_resource.merge.return_value = mock_merged_resource

        # Call the function
        result = _get_litellm_resource()

        # Verify Resource.create was called with correct default attributes
        expected_attributes = {
            "service.name": "litellm",
            "deployment.environment": "production",
            "model_id": "litellm",
        }
        mock_resource_create.assert_called_once_with(expected_attributes)
        mock_detector.detect.assert_called_once()
        mock_base_resource.merge.assert_called_once_with(mock_env_resource)
        self.assertEqual(result, mock_merged_resource)

    @patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "test-service",
            "OTEL_ENVIRONMENT_NAME": "staging",
            "OTEL_MODEL_ID": "test-model",
        },
        clear=True,
    )
    @patch("opentelemetry.sdk.resources.Resource.create")
    @patch("opentelemetry.sdk.resources.OTELResourceDetector")
    def test_get_litellm_resource_with_litellm_env_vars(
        self, mock_detector_cls, mock_resource_create
    ):
        """Test _get_litellm_resource with LiteLLM-specific environment variables."""
        from litellm.integrations.opentelemetry import _get_litellm_resource

        # Mock the Resource.create method
        mock_base_resource = MagicMock()
        mock_resource_create.return_value = mock_base_resource

        # Mock the OTELResourceDetector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_env_resource = MagicMock()
        mock_detector.detect.return_value = mock_env_resource

        # Mock the merged resource
        mock_merged_resource = MagicMock()
        mock_base_resource.merge.return_value = mock_merged_resource

        # Call the function
        result = _get_litellm_resource()

        # Verify Resource.create was called with environment variable values
        expected_attributes = {
            "service.name": "test-service",
            "deployment.environment": "staging",
            "model_id": "test-model",
        }
        mock_resource_create.assert_called_once_with(expected_attributes)
        mock_detector.detect.assert_called_once()
        mock_base_resource.merge.assert_called_once_with(mock_env_resource)
        self.assertEqual(result, mock_merged_resource)

    @patch.dict(
        os.environ,
        {
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=otel-service,deployment.environment=production,custom.attr=value",
            "OTEL_SERVICE_NAME": "should-be-overridden",
        },
        clear=True,
    )
    @patch("opentelemetry.sdk.resources.Resource.create")
    @patch("opentelemetry.sdk.resources.OTELResourceDetector")
    def test_get_litellm_resource_with_otel_resource_attributes(
        self, mock_detector_cls, mock_resource_create
    ):
        """Test _get_litellm_resource with OTEL_RESOURCE_ATTRIBUTES environment variable."""
        from litellm.integrations.opentelemetry import _get_litellm_resource

        # Mock the Resource.create method to simulate the actual behavior
        # In reality, Resource.create() would parse OTEL_RESOURCE_ATTRIBUTES and merge it
        mock_base_resource = MagicMock()
        mock_resource_create.return_value = mock_base_resource

        # Mock the OTELResourceDetector
        mock_detector = MagicMock()
        mock_detector_cls.return_value = mock_detector
        mock_env_resource = MagicMock()
        mock_detector.detect.return_value = mock_env_resource

        # Mock the merged resource
        mock_merged_resource = MagicMock()
        mock_base_resource.merge.return_value = mock_merged_resource

        # Call the function
        result = _get_litellm_resource()

        # Verify Resource.create was called with the base attributes
        # The actual OTEL_RESOURCE_ATTRIBUTES parsing is handled by OpenTelemetry SDK
        expected_attributes = {
            "service.name": "should-be-overridden",
            "deployment.environment": "production",
            "model_id": "should-be-overridden",
        }
        mock_resource_create.assert_called_once_with(expected_attributes)
        mock_detector.detect.assert_called_once()
        mock_base_resource.merge.assert_called_once_with(mock_env_resource)
        self.assertEqual(result, mock_merged_resource)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_litellm_resource_integration_with_real_resource(self):
        """Integration test to verify _get_litellm_resource works with actual OpenTelemetry Resource."""
        from litellm.integrations.opentelemetry import _get_litellm_resource

        # This test uses the real OpenTelemetry Resource.create() method
        result = _get_litellm_resource()

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # Verify the resource has the expected default attributes
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "litellm")
        self.assertEqual(attributes.get("deployment.environment"), "production")
        self.assertEqual(attributes.get("model_id"), "litellm")

    @patch.dict(
        os.environ,
        {
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=from-env,custom.attribute=test-value,deployment.environment=test-env"
        },
        clear=True,
    )
    def test_get_litellm_resource_real_otel_resource_attributes(self):
        """Integration test to verify OTEL_RESOURCE_ATTRIBUTES is properly handled."""
        from litellm.integrations.opentelemetry import _get_litellm_resource

        # This test uses the real OpenTelemetry Resource.create() method
        result = _get_litellm_resource()

        print("RESULT", result)

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # Verify that OTEL_RESOURCE_ATTRIBUTES values override the defaults
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "from-env")
        self.assertEqual(attributes.get("deployment.environment"), "test-env")
        self.assertEqual(attributes.get("custom.attribute"), "test-value")
        # model_id should still be set from the base attributes since it wasn't in OTEL_RESOURCE_ATTRIBUTES
        self.assertEqual(attributes.get("model_id"), "litellm")

    @patch.dict(
        os.environ,
        {
            "OTEL_SERVICE_NAME": "litellm-service",
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=otel-override,extra.attr=extra-value",
        },
        clear=True,
    )
    def test_get_litellm_resource_precedence(self):
        """Test that OTEL_SERVICE_NAME takes precedence over OTEL_RESOURCE_ATTRIBUTES according to OpenTelemetry spec."""
        from litellm.integrations.opentelemetry import _get_litellm_resource

        # This test verifies the OpenTelemetry standard behavior
        result = _get_litellm_resource()

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # According to OpenTelemetry spec, OTEL_SERVICE_NAME takes precedence over service.name in OTEL_RESOURCE_ATTRIBUTES
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "litellm-service")
        # But other attributes from OTEL_RESOURCE_ATTRIBUTES should still be present
        self.assertEqual(attributes.get("extra.attr"), "extra-value")

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

        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])

        # ─── instantiate our OpenTelemetry logger with test providers ───────────
        otel = OpenTelemetry(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        # OpenTelemetry attempts to set a global tracer provider, which can be set only once.
        # so we hack here to set a local tracer deriver from the provider we created.
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)

        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

        # ─── assert spans ────────────────────────────────────────────────────────
        spans = self.wait_for_spans(span_exporter, "gen_ai.")
        self.assertTrue(spans, "Expected at least one gen_ai span")

        # verify our top‐level litellm_request span is present
        names = [s.name for s in spans]
        self.assertIn("litellm_request", names)

        # ─── assert metrics ──────────────────────────────────────────────────────
        duration_metric = self.wait_for_metric(
            metric_reader, "gen_ai.client.operation.duration"
        )
        self.assertIsNotNone(duration_metric, "duration histogram was not recorded")

        # check that our model attribute made it onto at least one data point
        found_dp = False
        if (
            duration_metric
            and hasattr(duration_metric, "data")
            and hasattr(duration_metric.data, "data_points")
        ):
            found_dp = any(
                dp.attributes.get("gen_ai.request.model") == self.MODEL
                for dp in duration_metric.data.data_points
            )
        self.assertTrue(
            found_dp, "expected gen_ai.request.model attribute on a data point"
        )

        # ─── assert logs ───────────────────────────────────────────────────────
        logs = []
        logs = self.wait_for_log(log_exporter, "gen_ai.")
        self.assertTrue(logs, "Expected at least one gen_ai log")

        user_logs = [log for log in logs if log.log_record.attributes.get("event_name") == "gen_ai.content.prompt"]
        self.assertTrue(user_logs, "did not see a gen_ai.content.prompt log")
        # check log bodies
        user_prompt = user_logs[0].log_record.attributes.get("gen_ai.prompt")
        self.assertEqual("What is the capital of France?", user_prompt, "did not see a prompt message")

        choice_logs = [log for log in logs if log.log_record.attributes.get("event_name") == "gen_ai.content.completion"]
        self.assertTrue(choice_logs, "did not see a gen_ai.content.completion event")

        choice_response = choice_logs[0].log_record.body
        self.assertIsNotNone(choice_response, "did not see a response message")
        self.assertEqual("stop", choice_response.get("finish_reason"), "did not see expected finish reason")


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
            meter_provider=meter_provider,
            logger_provider=logger_provider,  # pass even if events disabled (safe)
        )
        # bind our tracer to the test tracer provider (global registration is a no-op after the first time)
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
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
            s.attributes
            and s.attributes.get("gen_ai.request.model") == self.MODEL
            for s in spans
        )
        self.assertTrue(found, "expected gen_ai.request.model on span attributes")

        # no metrics recorded
        self.assertIsNone(
            self.wait_for_metric(metric_reader, "gen_ai.client.operation.duration"),
            "Did not expect any metrics",
        )
        # no logs emitted
        logs = log_exporter.get_finished_logs()
        self.assertFalse(logs, "Did not expect any logs")

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
            meter_provider=meter_provider,
            logger_provider=logger_provider,  # needed if events were enabled
        )
        otel.tracer = tracer_provider.get_tracer(__name__)

        # ─── minimal input / output for a chat call ──────────────────────────────
        start = datetime.utcnow()
        end = start + timedelta(seconds=1)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
            response_obj = json.load(f)

        # ─── exercise the hook ───────────────────────────────────────────────────
        otel._handle_success(kwargs, response_obj, start, end)

        # ─── assert spans ────────────────────────────────────────────────────────
        spans = span_exporter.get_finished_spans()
        self.assertTrue(spans, "Expected at least one span")

        # ─── assert metrics ──────────────────────────────────────────────────────
        duration_metric = self.wait_for_metric(
            metric_reader, "gen_ai.client.operation.duration"
        )
        self.assertIsNotNone(duration_metric, "duration histogram was not recorded")
        # model attribute should be present on a data point
        found_dp = False
        if (
            duration_metric
            and hasattr(duration_metric, "data")
            and hasattr(duration_metric.data, "data_points")
        ):
            found_dp = any(
                dp.attributes.get("gen_ai.request.model") == self.MODEL
                for dp in duration_metric.data.data_points
            )
        self.assertTrue(
            found_dp, "expected gen_ai.request.model attribute on a data point"
        )

        # ─── no events when only metrics enabled ─────────────────────────────────
        logs = log_exporter.get_finished_logs()
        self.assertFalse(logs, "Did not expect any logs")

    def test_get_span_name_with_generation_name(self):
        """Test _get_span_name returns generation_name when present"""
        otel = OpenTelemetry()
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "generation_name": "custom_span"
                }
            }
        }
        result = otel._get_span_name(kwargs)
        self.assertEqual(result, "custom_span")

    def test_get_span_name_without_generation_name(self):
        """Test _get_span_name returns default when generation_name missing"""
        from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME

        otel = OpenTelemetry()
        kwargs = {"litellm_params": {"metadata": {}}}
        result = otel._get_span_name(kwargs)
        self.assertEqual(result, LITELLM_REQUEST_SPAN_NAME)

    @patch('litellm.turn_off_message_logging', False)
    def test_maybe_log_raw_request_creates_span(self):
        """Test _maybe_log_raw_request creates span when logging enabled"""
        from litellm.integrations.opentelemetry import RAW_REQUEST_SPAN_NAME

        otel = OpenTelemetry()
        otel.message_logging = True

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)
        otel.set_raw_request_attributes = MagicMock()
        otel._to_ns = MagicMock(return_value=1234567890)

        kwargs = {"litellm_params": {"metadata": {}}}
        otel._maybe_log_raw_request(kwargs, {}, datetime.now(), datetime.now(), MagicMock())

        mock_tracer.start_span.assert_called_once()
        self.assertEqual(mock_tracer.start_span.call_args[1]['name'], RAW_REQUEST_SPAN_NAME)

    @patch('litellm.turn_off_message_logging', True)
    def test_maybe_log_raw_request_skips_when_logging_disabled(self):
        """Test _maybe_log_raw_request skips when logging disabled"""
        otel = OpenTelemetry()
        mock_tracer = MagicMock()
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)

        kwargs = {"litellm_params": {"metadata": {}}}
        otel._maybe_log_raw_request(kwargs, {}, datetime.now(), datetime.now(), MagicMock())

        mock_tracer.start_span.assert_not_called()
