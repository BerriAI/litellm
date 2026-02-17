import json
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))
from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


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
        kwargs = {"standard_logging_object": {"guardrail_information": [ guardrail_info ]}}

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


class TestOpenTelemetryCostBreakdown(unittest.TestCase):
    def test_cost_breakdown_emitted_to_otel_span(self):
        """
        Test that cost breakdown from StandardLoggingPayload is emitted to OpenTelemetry span attributes.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        cost_breakdown = {
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
            "tool_usage_cost": 0.0001,
            "original_cost": 0.004,
            "discount_percent": 0.25,
            "discount_amount": 0.001,
        }

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
                "cost_breakdown": cost_breakdown,
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        mock_span.set_attribute.assert_any_call("gen_ai.cost.input_cost", 0.001)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.output_cost", 0.002)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.total_cost", 0.003)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.tool_usage_cost", 0.0001)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.original_cost", 0.004)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.discount_percent", 0.25)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.discount_amount", 0.001)

    def test_cost_breakdown_with_partial_fields(self):
        """
        Test that cost breakdown works correctly when only some fields are present.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        cost_breakdown = {
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
        }

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
                "cost_breakdown": cost_breakdown,
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        mock_span.set_attribute.assert_any_call("gen_ai.cost.input_cost", 0.001)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.output_cost", 0.002)
        mock_span.set_attribute.assert_any_call("gen_ai.cost.total_cost", 0.003)

        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]
        assert ("gen_ai.cost.tool_usage_cost", 0.0001) not in call_args_list
        assert ("gen_ai.cost.original_cost", 0.004) not in call_args_list


class TestOpenTelemetryProviderInitialization(unittest.TestCase):
    """Test suite for verifying provider initialization respects existing providers"""

    def test_init_tracing_respects_existing_tracer_provider(self):
        """
        Unit test: _init_tracing() should respect existing TracerProvider.

        When a TracerProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

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

    @patch.dict(os.environ, {"LITELLM_OTEL_INTEGRATION_ENABLE_METRICS": "true"}, clear=True)
    def test_init_metrics_respects_existing_meter_provider(self):
        """
        Unit test: _init_metrics() should respect existing MeterProvider.

        When a MeterProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider

        # Create and set an existing MeterProvider
        meter_provider = MeterProvider()
        metrics.set_meter_provider(meter_provider)
        existing_provider = metrics.get_meter_provider()

        # Act: Initialize OpenTelemetry integration (should detect existing provider)
        config = OpenTelemetryConfig.from_env()
        otel_integration = OpenTelemetry(config=config)

        # Assert: The existing provider should still be active
        current_provider = metrics.get_meter_provider()
        assert current_provider is existing_provider, (
            "Existing MeterProvider should be respected and not overridden"
        )

    @patch.dict(os.environ, {"LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS": "true"}, clear=True)
    def test_init_logs_respects_existing_logger_provider(self):
        """
        Unit test: _init_logs() should respect existing LoggerProvider.

        When a LoggerProvider already exists (e.g., set by Langfuse SDK),
        LiteLLM should use it instead of creating a new one.
        """
        from opentelemetry._logs import get_logger_provider, set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider

        # Create and set an existing LoggerProvider
        logger_provider = OTLoggerProvider()
        set_logger_provider(logger_provider)
        existing_provider = get_logger_provider()

        # Act: Initialize OpenTelemetry integration (should detect existing provider)
        config = OpenTelemetryConfig.from_env()
        otel_integration = OpenTelemetry(config=config)

        # Assert: The existing provider should still be active
        current_provider = get_logger_provider()
        assert current_provider is existing_provider, (
            "Existing LoggerProvider should be respected and not overridden"
        )


class TestOpenTelemetry(unittest.TestCase):
    POLL_INTERVAL = 0.05
    POLL_TIMEOUT = 2.0
    MODEL = "arn:aws:bedrock:us-west-2:1234567890123:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    HERE = os.path.dirname(__file__)

    @patch.dict(os.environ, {}, clear=True)
    def test_open_telemetry_config_manual_defaults(self):
        """Manual OpenTelemetryConfig creation should populate default identifiers."""
        config = OpenTelemetryConfig(exporter="console", endpoint="http://collector")
        self.assertEqual(config.service_name, "litellm")
        self.assertEqual(config.deployment_environment, "production")
        self.assertEqual(config.model_id, "litellm")

    @patch.dict(os.environ, {}, clear=True)
    def test_open_telemetry_config_custom_service_name(self):
        """Model ID should inherit provided service name when not explicitly set."""
        config = OpenTelemetryConfig(service_name="custom-service", exporter="console")
        self.assertEqual(config.service_name, "custom-service")
        self.assertEqual(config.deployment_environment, "production")
        self.assertEqual(config.model_id, "custom-service")

    @patch.dict(os.environ, {}, clear=True)
    def test_open_telemetry_config_auto_infer_otlp_http_when_endpoint_set(self):
        """When endpoint is set but exporter is default 'console', auto-infer 'otlp_http'.

        This fixes an issue where UI-configured OTEL settings would default to console
        output instead of sending traces to the configured endpoint.
        See: https://github.com/BerriAI/litellm/issues/XXXX
        """
        # When endpoint is specified without explicit exporter, should auto-infer otlp_http
        config = OpenTelemetryConfig(endpoint="https://otel-collector.example.com:443")
        self.assertEqual(config.exporter, "otlp_http")

        # When exporter is explicitly set to something other than console, should not override
        config_grpc = OpenTelemetryConfig(exporter="grpc", endpoint="https://otel-collector.example.com:443")
        self.assertEqual(config_grpc.exporter, "grpc")

        # When no endpoint is set, should keep console as default
        config_no_endpoint = OpenTelemetryConfig()
        self.assertEqual(config_no_endpoint.exporter, "console")

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
        kwargs = {"standard_logging_object": {"guardrail_information": [ guardrail_info ]}}

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

        config = OpenTelemetryConfig()
        result = OpenTelemetry._get_litellm_resource(config)

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

        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

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

        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

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
        config = OpenTelemetryConfig()
        result = OpenTelemetry._get_litellm_resource(config)

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
        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

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
        config = OpenTelemetryConfig.from_env()
        result = OpenTelemetry._get_litellm_resource(config)

        # Verify the result is a Resource instance
        from opentelemetry.sdk.resources import Resource

        self.assertIsInstance(result, Resource)

        # According to OpenTelemetry spec, OTEL_SERVICE_NAME takes precedence over service.name in OTEL_RESOURCE_ATTRIBUTES
        attributes = result.attributes
        self.assertEqual(attributes.get("service.name"), "litellm-service")
        # But other attributes from OTEL_RESOURCE_ATTRIBUTES should still be present
        self.assertEqual(attributes.get("extra.attr"), "extra-value")


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
            s.attributes and s.attributes.get("gen_ai.request.model") == self.MODEL
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

    @patch.dict(os.environ, {"LITELLM_OTEL_INTEGRATION_ENABLE_METRICS": "true"}, clear=True)
    def test_handle_success_spans_and_metrics(self):
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
        kwargs = {"litellm_params": {"metadata": {"generation_name": "custom_span"}}}
        result = otel._get_span_name(kwargs)
        self.assertEqual(result, "custom_span")

    def test_get_span_name_without_generation_name(self):
        """Test _get_span_name returns default when generation_name missing"""
        from litellm.integrations.opentelemetry import LITELLM_REQUEST_SPAN_NAME

        otel = OpenTelemetry()
        kwargs = {"litellm_params": {"metadata": {}}}
        result = otel._get_span_name(kwargs)
        self.assertEqual(result, LITELLM_REQUEST_SPAN_NAME)

    @patch("litellm.turn_off_message_logging", False)
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
        otel._maybe_log_raw_request(
            kwargs, {}, datetime.now(), datetime.now(), MagicMock()
        )

        mock_tracer.start_span.assert_called_once()
        self.assertEqual(
            mock_tracer.start_span.call_args[1]["name"], RAW_REQUEST_SPAN_NAME
        )

    @patch("litellm.turn_off_message_logging", True)
    def test_maybe_log_raw_request_skips_when_logging_disabled(self):
        """Test _maybe_log_raw_request skips when logging disabled"""
        otel = OpenTelemetry()
        mock_tracer = MagicMock()
        otel.get_tracer_to_use_for_request = MagicMock(return_value=mock_tracer)

        kwargs = {"litellm_params": {"metadata": {}}}
        otel._maybe_log_raw_request(
            kwargs, {}, datetime.now(), datetime.now(), MagicMock()
        )

        mock_tracer.start_span.assert_not_called()


class TestOpenTelemetryHeaderSplitting(unittest.TestCase):
    """Test suite for _get_headers_dictionary method"""

    def test_split_multiple_headers_comma_separated(self):
        """Test splitting multiple headers separated by commas"""
        otel = OpenTelemetry()
        headers = "api-key=key,other-config-value=value"
        result = otel._get_headers_dictionary(headers)
        self.assertEqual(result, {"api-key": "key", "other-config-value": "value"})

    def test_split_headers_with_equals_in_values(self):
        """Test splitting headers where values contain equals signs (split only on first '=')"""
        otel = OpenTelemetry()
        headers = "api-key=value1=part2,config=setting=enabled"
        result = otel._get_headers_dictionary(headers)
        self.assertEqual(
            result, {"api-key": "value1=part2", "config": "setting=enabled"}
        )


class TestOpenTelemetryEndpointNormalization(unittest.TestCase):
    """Test suite for the unified _normalize_otel_endpoint method"""

    def test_normalize_traces_endpoint_from_logs_path(self):
        """Test normalizing endpoint with /v1/logs to /v1/traces"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/logs", "traces"
        )
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_from_metrics_path(self):
        """Test normalizing endpoint with /v1/metrics to /v1/traces"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/metrics", "traces"
        )
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_from_base_url(self):
        """Test adding /v1/traces to base URL"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318", "traces")
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_from_v1_path(self):
        """Test adding traces to /v1 path"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318/v1", "traces")
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_traces_endpoint_already_correct(self):
        """Test endpoint already ending with /v1/traces remains unchanged"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/traces", "traces"
        )
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_metrics_endpoint_from_traces_path(self):
        """Test normalizing endpoint with /v1/traces to /v1/metrics"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/traces", "metrics"
        )
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_metrics_endpoint_from_logs_path(self):
        """Test normalizing endpoint with /v1/logs to /v1/metrics"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/logs", "metrics"
        )
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_metrics_endpoint_from_base_url(self):
        """Test adding /v1/metrics to base URL"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318", "metrics")
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_metrics_endpoint_already_correct(self):
        """Test endpoint already ending with /v1/metrics remains unchanged"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/metrics", "metrics"
        )
        self.assertEqual(result, "http://collector:4318/v1/metrics")

    def test_normalize_logs_endpoint_from_traces_path(self):
        """Test normalizing endpoint with /v1/traces to /v1/logs"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/traces", "logs"
        )
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_logs_endpoint_from_metrics_path(self):
        """Test normalizing endpoint with /v1/metrics to /v1/logs"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/v1/metrics", "logs"
        )
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_logs_endpoint_from_base_url(self):
        """Test adding /v1/logs to base URL"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318", "logs")
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_logs_endpoint_already_correct(self):
        """Test endpoint already ending with /v1/logs remains unchanged"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318/v1/logs", "logs")
        self.assertEqual(result, "http://collector:4318/v1/logs")

    def test_normalize_endpoint_with_trailing_slash(self):
        """Test that trailing slashes are properly handled"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("http://collector:4318/", "traces")
        self.assertEqual(result, "http://collector:4318/v1/traces")

    def test_normalize_endpoint_none(self):
        """Test that None endpoint returns None"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(None, "traces")
        self.assertIsNone(result)

    def test_normalize_endpoint_empty_string(self):
        """Test that empty string returns empty string"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint("", "traces")
        self.assertEqual(result, "")

    def test_normalize_endpoint_invalid_signal_type(self):
        """Test that invalid signal type returns endpoint unchanged with warning"""
        otel = OpenTelemetry()
        endpoint = "http://collector:4318/v1/traces"

        with patch("litellm._logging.verbose_logger.warning") as mock_warning:
            result = otel._normalize_otel_endpoint(endpoint, "invalid")

            # Should return endpoint unchanged
            self.assertEqual(result, endpoint)

            # Should log a warning
            mock_warning.assert_called_once()
            # Check the warning was called with the expected format string and parameters
            call_args = mock_warning.call_args[0]
            self.assertIn("Invalid signal_type", call_args[0])
            self.assertEqual(call_args[1], "invalid")  # signal_type parameter
            self.assertEqual(
                call_args[2], {"traces", "metrics", "logs"}
            )  # valid_signals parameter

    def test_normalize_endpoint_https(self):
        """Test normalization works with https URLs"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "https://collector.example.com:4318", "logs"
        )
        self.assertEqual(result, "https://collector.example.com:4318/v1/logs")

    def test_normalize_endpoint_with_path_prefix(self):
        """Test normalization works with URLs that have path prefixes"""
        otel = OpenTelemetry()
        result = otel._normalize_otel_endpoint(
            "http://collector:4318/otel/v1/traces", "logs"
        )
        # Should replace the final /traces with /logs
        self.assertEqual(result, "http://collector:4318/otel/v1/logs")

    def test_normalize_endpoint_consistency_across_signals(self):
        """Test that normalization is consistent for all signal types from the same base"""
        otel = OpenTelemetry()
        base = "http://collector:4318"

        traces_result = otel._normalize_otel_endpoint(base, "traces")
        metrics_result = otel._normalize_otel_endpoint(base, "metrics")
        logs_result = otel._normalize_otel_endpoint(base, "logs")

        # All should have the same base with different signal paths
        self.assertEqual(traces_result, "http://collector:4318/v1/traces")
        self.assertEqual(metrics_result, "http://collector:4318/v1/metrics")
        self.assertEqual(logs_result, "http://collector:4318/v1/logs")

    def test_normalize_endpoint_signal_switching(self):
        """Test switching between different signal types on the same endpoint"""
        otel = OpenTelemetry()

        # Start with traces
        endpoint = "http://collector:4318/v1/traces"

        # Switch to metrics
        metrics = otel._normalize_otel_endpoint(endpoint, "metrics")
        self.assertEqual(metrics, "http://collector:4318/v1/metrics")

        # Switch to logs
        logs = otel._normalize_otel_endpoint(metrics, "logs")
        self.assertEqual(logs, "http://collector:4318/v1/logs")

        # Switch back to traces
        traces = otel._normalize_otel_endpoint(logs, "traces")
        self.assertEqual(traces, "http://collector:4318/v1/traces")


class TestOpenTelemetryProtocolSelection(unittest.TestCase):
    """Test suite for verifying correct exporter selection based on protocol"""

    def test_get_span_processor_uses_http_exporter_for_otlp_http(self):
        """Test that otlp_http protocol uses OTLPSpanExporterHTTP"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(
            exporter="otlp_http", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the HTTP variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterHTTP)

    def test_get_span_processor_uses_grpc_exporter_for_otlp_grpc(self):
        """Test that otlp_grpc protocol uses OTLPSpanExporterGRPC"""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(
            exporter="otlp_grpc", endpoint="http://collector:4317"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterGRPC)

    def test_get_span_processor_uses_grpc_exporter_for_grpc_alias(self):
        """Test that 'grpc' protocol alias uses OTLPSpanExporterGRPC"""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(exporter="grpc", endpoint="http://collector:4317")
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterGRPC)

    def test_get_span_processor_uses_http_exporter_for_http_protobuf(self):
        """Test that http/protobuf protocol uses OTLPSpanExporterHTTP"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig(
            exporter="http/protobuf", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the HTTP variant
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterHTTP)

    def test_get_span_processor_uses_console_exporter_for_console(self):
        """Test that console protocol uses ConsoleSpanExporter"""
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        config = OpenTelemetryConfig(exporter="console")
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify it's a BatchSpanProcessor
        self.assertIsInstance(processor, BatchSpanProcessor)

        # Verify the exporter is the console variant
        self.assertIsInstance(processor.span_exporter, ConsoleSpanExporter)

    def test_get_log_exporter_uses_http_exporter_for_otlp_http(self):
        """Test that otlp_http protocol uses HTTP OTLPLogExporter"""
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        config = OpenTelemetryConfig(
            exporter="otlp_http", endpoint="http://collector:4318", enable_events=True
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the HTTP variant
        self.assertIsInstance(exporter, OTLPLogExporter)

        # Check that it's from the http module by checking the module name
        self.assertIn("http", exporter.__class__.__module__)

    def test_get_log_exporter_uses_grpc_exporter_for_otlp_grpc(self):
        """Test that otlp_grpc protocol uses gRPC OTLPLogExporter"""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        config = OpenTelemetryConfig(
            exporter="otlp_grpc", endpoint="http://collector:4317", enable_events=True
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(exporter, OTLPLogExporter)

        # Check that it's from the grpc module by checking the module name
        self.assertIn("grpc", exporter.__class__.__module__)

    def test_get_log_exporter_uses_grpc_exporter_for_grpc_alias(self):
        """Test that 'grpc' protocol alias uses gRPC OTLPLogExporter"""
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        config = OpenTelemetryConfig(
            exporter="grpc", endpoint="http://collector:4317", enable_events=True
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the gRPC variant
        self.assertIsInstance(exporter, OTLPLogExporter)

        # Check that it's from the grpc module by checking the module name
        self.assertIn("grpc", exporter.__class__.__module__)

    def test_get_log_exporter_uses_console_exporter_for_console(self):
        """Test that console protocol uses ConsoleLogExporter"""
        from opentelemetry.sdk._logs.export import ConsoleLogExporter

        config = OpenTelemetryConfig(exporter="console", enable_events=True)
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the exporter is the console variant
        self.assertIsInstance(exporter, ConsoleLogExporter)

    def test_get_log_exporter_defaults_to_console_for_unknown_protocol(self):
        """Test that unknown protocol defaults to ConsoleLogExporter with warning"""
        from opentelemetry.sdk._logs.export import ConsoleLogExporter

        config = OpenTelemetryConfig(exporter="unknown_protocol", enable_events=True)
        otel = OpenTelemetry(config=config)

        with patch("litellm._logging.verbose_logger.warning") as mock_warning:
            exporter = otel._get_log_exporter()

            # Verify the exporter defaults to console
            self.assertIsInstance(exporter, ConsoleLogExporter)

            # Verify a warning was logged
            mock_warning.assert_called_once()
            args = mock_warning.call_args[0]
            self.assertIn("Unknown log exporter", args[0])
            self.assertIn("unknown_protocol", args[1])

    @patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER": "otlp_http",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
        },
        clear=False,
    )
    def test_protocol_selection_from_environment_http(self):
        """Test that protocol selection works correctly from environment variables for HTTP"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterHTTP,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig.from_env()
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the HTTP exporter is used
        self.assertIsInstance(processor, BatchSpanProcessor)
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterHTTP)

    @patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER": "otlp_grpc",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
        },
        clear=False,
    )
    def test_protocol_selection_from_environment_grpc(self):
        """Test that protocol selection works correctly from environment variables for gRPC"""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = OpenTelemetryConfig.from_env()
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the gRPC exporter is used
        self.assertIsInstance(processor, BatchSpanProcessor)
        self.assertIsInstance(processor.span_exporter, OTLPSpanExporterGRPC)

    def test_http_exporter_endpoint_normalization_for_traces(self):
        """Test that HTTP trace exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_http", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the endpoint was normalized to include /v1/traces
        # Access the private _endpoint attribute if available
        if hasattr(processor.span_exporter, "_endpoint"):
            self.assertEqual(processor.span_exporter._endpoint, "http://collector:4318/v1/traces")  # type: ignore[attr-defined]

    def test_grpc_exporter_endpoint_normalization_for_traces(self):
        """Test that gRPC trace exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_grpc", endpoint="http://collector:4317"
        )
        otel = OpenTelemetry(config=config)

        processor = otel._get_span_processor()

        # Verify the endpoint was normalized to include /v1/traces
        # Note: gRPC exporters strip the http:// prefix, so we check for the normalized path
        if hasattr(processor.span_exporter, "_endpoint"):
            # gRPC exporter strips http:// prefix
            self.assertIn("collector:4317", processor.span_exporter._endpoint)  # type: ignore[attr-defined]
            # The endpoint should have been normalized with /v1/traces before being passed to gRPC exporter
            # We verify this by checking the normalization function was called correctly
            normalized = otel._normalize_otel_endpoint(
                "http://collector:4317", "traces"
            )
            self.assertEqual(normalized, "http://collector:4317/v1/traces")

    def test_http_log_exporter_endpoint_normalization_for_logs(self):
        """Test that HTTP log exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint="http://collector:4318/v1/traces",
            enable_events=True,
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the endpoint was normalized to /v1/logs (not /v1/traces)
        # Access the private _endpoint attribute if available
        if hasattr(exporter, "_endpoint"):
            self.assertEqual(exporter._endpoint, "http://collector:4318/v1/logs")  # type: ignore[attr-defined]

    def test_grpc_log_exporter_endpoint_normalization_for_logs(self):
        """Test that gRPC log exporter gets properly normalized endpoint"""
        config = OpenTelemetryConfig(
            exporter="otlp_grpc",
            endpoint="http://collector:4317/v1/traces",
            enable_events=True,
        )
        otel = OpenTelemetry(config=config)

        exporter = otel._get_log_exporter()

        # Verify the endpoint was normalized to /v1/logs (not /v1/traces)
        # Note: gRPC exporters strip the http:// prefix, so we check for the normalized path
        if hasattr(exporter, "_endpoint"):
            # gRPC exporter strips http:// prefix
            self.assertIn("collector:4317", exporter._endpoint)  # type: ignore[attr-defined]
            # The endpoint should have been normalized with /v1/logs before being passed to gRPC exporter
            # We verify this by checking the normalization function was called correctly
            normalized = otel._normalize_otel_endpoint(
                "http://collector:4317/v1/traces", "logs"
            )
            self.assertEqual(normalized, "http://collector:4317/v1/logs")

    def test_get_metric_reader_uses_http_exporter_for_http_protobuf(self):
        """Test that http/protobuf protocol uses OTLPMetricExporterHTTP"""
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        config = OpenTelemetryConfig(
            exporter="http/protobuf", endpoint="http://collector:4318"
        )
        otel = OpenTelemetry(config=config)

        reader = otel._get_metric_reader()

        self.assertIsInstance(reader, PeriodicExportingMetricReader)
        self.assertIsInstance(reader._exporter, OTLPMetricExporter)


class TestOpenTelemetryExternalSpan(unittest.TestCase):
    """
    Test suite for external span handling in OpenTelemetry integration.

    These tests verify that LiteLLM correctly handles spans created outside
    of LiteLLM (e.g., by Langfuse SDK, user application code, or global context)
    without closing them prematurely.

    Background:
    - External spans can come from: Langfuse SDK, user code, HTTP traceparent headers, global context
    - LiteLLM should NEVER close spans it did not create
    - Bug: LiteLLM was reusing and closing external spans in _start_primary_span
    """

    HERE = os.path.dirname(__file__)

    def setUp(self):
        """Set up common test fixtures"""
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )

        # Don't set global tracer provider - instead, get tracers directly from our provider
        # This avoids "Overriding of current TracerProvider is not allowed" warnings

        # Clear any existing spans
        self.span_exporter.clear()

    def _create_test_kwargs_and_response(self):
        """Load test data from JSON files"""
        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_kwargs.json")
        ) as f:
            kwargs = json.load(f)

        with open(
            os.path.join(self.HERE, "open_telemetry", "data", "captured_response.json")
        ) as f:
            response_obj = json.load(f)

        return kwargs, response_obj

    def _get_spans_by_name(self, name):
        """Get all spans with the given name"""
        spans = self.span_exporter.get_finished_spans()
        return [s for s in spans if s.name == name]

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_not_closed_with_use_otel_litellm_request_span_false(self):
        """
        Test that external spans are not closed when USE_OTEL_LITELLM_REQUEST_SPAN=false (default).

        Expected behavior:
        - External span remains open (is_recording = True)
        - raw_gen_ai_request spans are direct children of external span (shallow hierarchy)
        - No litellm_request span is created
        - Multiple completions work correctly
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id
            parent_span_id = parent_ctx.span_id

            self.assertTrue(
                parent_span.is_recording(),
                "External span should be recording before completion calls"
            )

            # First completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after first completion"
            )

            # Second completion call
            start_time2 = end_time
            end_time2 = start_time2 + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time2, end_time2)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after second completion"
            )

        # After exiting context, verify spans
        spans = self.span_exporter.get_finished_spans()

        # All spans should have the same trace_id
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                f"Span {span.name} should have same trace_id as parent"
            )

        # Should have external_parent_span
        parent_spans = self._get_spans_by_name("external_parent_span")
        self.assertEqual(len(parent_spans), 1, "Should have exactly one external_parent_span")

        # Verify LiteLLM set attributes on external parent span
        parent_span_finished = parent_spans[0]
        self.assertIsNotNone(
            parent_span_finished.attributes,
            "Parent span should have attributes set by LiteLLM"
        )
        self.assertIn(
            "gen_ai.request.model",
            parent_span_finished.attributes,
            "Parent span should have model attribute from LiteLLM"
        )

        # Should have raw_gen_ai_request spans (if message_logging is on)
        raw_spans = self._get_spans_by_name("raw_gen_ai_request")
        # Note: May be 0 if message_logging is off, or 2 if on

        # Should NOT have litellm_request spans (USE_OTEL_LITELLM_REQUEST_SPAN=false)
        litellm_spans = self._get_spans_by_name("litellm_request")
        self.assertEqual(
            len(litellm_spans),
            0,
            "Should NOT have litellm_request spans when USE_OTEL_LITELLM_REQUEST_SPAN=false"
        )

        # Verify raw_gen_ai_request spans are direct children of external span
        for raw_span in raw_spans:
            self.assertEqual(
                raw_span.parent.span_id if raw_span.parent else None,
                parent_span_id,
                f"raw_gen_ai_request should be direct child of external_parent_span"
            )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "true"}, clear=False)
    def test_external_span_not_closed_with_use_otel_litellm_request_span_true(self):
        """
        Test that external spans are not closed when USE_OTEL_LITELLM_REQUEST_SPAN=true.

        Expected behavior:
        - External span remains open (is_recording = True)
        - litellm_request spans are created as children of external span
        - raw_gen_ai_request spans are children of litellm_request spans
        - Correct hierarchy: external_parent → litellm_request → raw_gen_ai_request
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id
            parent_span_id = parent_ctx.span_id

            # First completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after first completion"
            )

            # Second completion call
            start_time2 = end_time
            end_time2 = start_time2 + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time2, end_time2)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording after second completion"
            )

        # After exiting context, verify spans
        spans = self.span_exporter.get_finished_spans()

        # All spans should have the same trace_id
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                f"Span {span.name} should have same trace_id as parent"
            )

        # Should have litellm_request spans (USE_OTEL_LITELLM_REQUEST_SPAN=true)
        litellm_spans = self._get_spans_by_name("litellm_request")
        self.assertEqual(
            len(litellm_spans),
            2,
            "Should have 2 litellm_request spans when USE_OTEL_LITELLM_REQUEST_SPAN=true"
        )

        # Verify litellm_request spans are children of external span
        for litellm_span in litellm_spans:
            self.assertEqual(
                litellm_span.parent.span_id if litellm_span.parent else None,
                parent_span_id,
                "litellm_request should be child of external_parent_span"
            )

        # Verify raw_gen_ai_request spans (if present) are children of litellm_request
        raw_spans = self._get_spans_by_name("raw_gen_ai_request")
        if raw_spans:
            litellm_span_ids = {s.context.span_id for s in litellm_spans}
            for raw_span in raw_spans:
                self.assertIn(
                    raw_span.parent.span_id if raw_span.parent else None,
                    litellm_span_ids,
                    "raw_gen_ai_request should be child of litellm_request"
                )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_with_multiple_completions(self):
        """
        Test that multiple completion calls work correctly within external span context.

        Expected behavior:
        - Both completion calls succeed
        - All spans belong to the same trace
        - External span remains open throughout
        - No errors or warnings about "ended span"
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id

            # Make multiple completion calls
            for i in range(3):
                start_time = datetime.utcnow()
                end_time = start_time + timedelta(seconds=1)

                # This should not raise any exceptions
                otel._handle_success(kwargs, response_obj, start_time, end_time)

                # Verify parent span is still recording after each call
                self.assertTrue(
                    parent_span.is_recording(),
                    f"External span should still be recording after completion #{i+1}"
                )

        # Verify all spans have the same trace_id
        spans = self.span_exporter.get_finished_spans()
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                f"All spans should belong to the same trace"
            )

        # Should have the external parent span
        parent_spans = self._get_spans_by_name("external_parent_span")
        self.assertEqual(len(parent_spans), 1, "Should have exactly one external_parent_span")

        # Verify LiteLLM set attributes on external parent span
        parent_span_finished = parent_spans[0]
        self.assertIn(
            "gen_ai.request.model",
            parent_span_finished.attributes,
            "Parent span should have model attribute from LiteLLM"
        )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_from_global_context(self):
        """
        Test external span detection from global context (Priority 3 in _get_span_context).

        This simulates the case where a span is set in the global context
        (e.g., by user code or Langfuse SDK) and LiteLLM detects it via
        trace.get_current_span().

        Expected behavior:
        - LiteLLM detects the span from global context
        - External span is not closed
        - Correct parent-child relationship
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span and set it as current using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_global_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id

            # Verify the span is in global context
            current_span = trace.get_current_span()
            self.assertEqual(current_span, parent_span, "Span should be in global context")

            # Make completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span from global context should not be closed"
            )

        # Verify trace structure
        spans = self.span_exporter.get_finished_spans()
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                "All spans should have the same trace_id"
            )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_hierarchy_preserved(self):
        """
        Test that span hierarchy is correctly preserved with external parent.

        Expected behavior:
        - Parent span IDs are correct
        - Trace structure matches expected hierarchy
        - Span names are correct
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)
        otel.message_logging = True  # Enable message logging to get raw_gen_ai_request spans

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_span_id = parent_span.get_span_context().span_id

            # Make completion call
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)
            otel._handle_success(kwargs, response_obj, start_time, end_time)

        # Verify hierarchy
        spans = self.span_exporter.get_finished_spans()

        # Get spans by name
        parent_spans = self._get_spans_by_name("external_parent_span")
        raw_spans = self._get_spans_by_name("raw_gen_ai_request")

        self.assertEqual(len(parent_spans), 1, "Should have one parent span")

        # Verify parent-child relationship
        if raw_spans:  # If message_logging is on
            for raw_span in raw_spans:
                self.assertEqual(
                    raw_span.parent.span_id if raw_span.parent else None,
                    parent_span_id,
                    "raw_gen_ai_request should be child of external_parent_span"
                )

    @patch.dict(os.environ, {"USE_OTEL_LITELLM_REQUEST_SPAN": "false"}, clear=False)
    def test_external_span_not_ended_on_failure(self):
        """
        Test that external spans are not closed even on failure.

        Expected behavior:
        - When _handle_failure is called with external span context
        - External span remains open (is_recording = True)
        - Error span is created correctly
        - External span status is NOT changed by LiteLLM
        """
        # Initialize OpenTelemetry
        otel = OpenTelemetry(tracer_provider=self.tracer_provider)

        # Load test data
        kwargs, response_obj = self._create_test_kwargs_and_response()

        # Create external parent span using our test TracerProvider
        tracer = self.tracer_provider.get_tracer(__name__)

        with tracer.start_as_current_span("external_parent_span") as parent_span:
            parent_ctx = parent_span.get_span_context()
            parent_trace_id = parent_ctx.trace_id

            # Simulate failure
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=1)

            # Create error response object
            error_response = {"error": "Test error"}

            # Call _handle_failure
            otel._handle_failure(kwargs, error_response, start_time, end_time)

            # Verify parent span is still recording
            self.assertTrue(
                parent_span.is_recording(),
                "External span should still be recording even after failure"
            )

        # Verify trace structure
        spans = self.span_exporter.get_finished_spans()

        # All spans should have the same trace_id
        for span in spans:
            self.assertEqual(
                span.context.trace_id,
                parent_trace_id,
                "All spans should have the same trace_id even on failure"
            )

        # Should have external_parent_span
        parent_spans = self._get_spans_by_name("external_parent_span")
        self.assertEqual(len(parent_spans), 1, "Should have exactly one external_parent_span")

        # Verify LiteLLM set attributes on external parent span even on failure
        parent_span_finished = parent_spans[0]
        self.assertIn(
            "gen_ai.request.model",
            parent_span_finished.attributes,
            "Parent span should have model attribute from LiteLLM even on failure"
        )


class TestOpenTelemetrySemanticConventions138(unittest.TestCase):
    """
    Test suite for OpenTelemetry 1.38 Semantic Conventions compliance.
    
    These tests verify that LiteLLM emits span attributes following the 
    OpenTelemetry GenAI semantic conventions v1.38, including:
    - gen_ai.input.messages (JSON string with parts array)
    - gen_ai.output.messages (JSON string with parts array)
    - gen_ai.usage.input_tokens / output_tokens (new naming)
    - gen_ai.response.finish_reasons (JSON array)
    
    See: https://github.com/BerriAI/litellm/issues/17794
    """

    def test_input_messages_uses_parts_structure(self):
        """
        Test that gen_ai.input.messages uses the OTEL 1.38 parts array structure.
        
        Expected format:
        [{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello world"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi there!"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Find the call that set gen_ai.input.messages
        input_messages_calls = [
            call for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.input.messages"
        ]
        self.assertEqual(len(input_messages_calls), 1, "Should have exactly one gen_ai.input.messages attribute")
        
        input_messages_value = input_messages_calls[0][0][1]
        parsed = json.loads(input_messages_value)
        
        # Verify structure
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "user")
        self.assertIn("parts", parsed[0])
        self.assertEqual(parsed[0]["parts"][0]["type"], "text")
        self.assertEqual(parsed[0]["parts"][0]["content"], "Hello world")

    def test_output_messages_uses_parts_structure(self):
        """
        Test that gen_ai.output.messages uses the OTEL 1.38 parts array structure.
        
        Expected format:
        [{"role": "assistant", "parts": [{"type": "text", "content": "Hi!"}], "finish_reason": "stop"}]
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello back!"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Find the call that set gen_ai.output.messages
        output_messages_calls = [
            call for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.output.messages"
        ]
        self.assertEqual(len(output_messages_calls), 1, "Should have exactly one gen_ai.output.messages attribute")
        
        output_messages_value = output_messages_calls[0][0][1]
        parsed = json.loads(output_messages_value)
        
        # Verify structure
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "assistant")
        self.assertIn("parts", parsed[0])
        self.assertEqual(parsed[0]["parts"][0]["type"], "text")
        self.assertEqual(parsed[0]["parts"][0]["content"], "Hello back!")
        self.assertEqual(parsed[0]["finish_reason"], "stop")

    def test_usage_tokens_use_new_naming_convention(self):
        """
        Test that token usage uses the OTEL 1.38 naming convention:
        - gen_ai.usage.input_tokens (not prompt_tokens)
        - gen_ai.usage.output_tokens (not completion_tokens)
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Verify new naming convention is used
        mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)

    def test_finish_reasons_is_json_array(self):
        """
        Test that gen_ai.response.finish_reasons is a proper JSON array.
        
        Expected: '["stop"]' (not "['stop']")
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        # Find the call that set gen_ai.response.finish_reasons
        finish_reasons_calls = [
            call for call in mock_span.set_attribute.call_args_list
            if call[0][0] == "gen_ai.response.finish_reasons"
        ]
        self.assertEqual(len(finish_reasons_calls), 1, "Should have exactly one gen_ai.response.finish_reasons attribute")
        
        finish_reasons_value = finish_reasons_calls[0][0][1]
        
        # Verify it's valid JSON (not Python repr)
        parsed = json.loads(finish_reasons_value)
        self.assertEqual(parsed, ["stop"])

    def test_operation_name_is_chat_for_completion(self):
        """
        Test that gen_ai.operation.name is 'chat' for completion calls.
        """
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
            "standard_logging_object": {
                "id": "test-id",
                "call_type": "completion",
                "metadata": {},
            },
        }

        response_obj = {
            "id": "test-response-id",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "chat")
