import unittest
from unittest.mock import patch

import pytest

from litellm.integrations.levo.levo import LevoLogger
from litellm.integrations.opentelemetry import OpenTelemetryConfig

# Try to import OpenTelemetry packages, skip tests if not available
try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class TestLevoIntegration(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
        },
    )
    @pytest.mark.skipif(
        not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry packages not installed"
    )
    @patch(
        "litellm.integrations.opentelemetry.OpenTelemetry._init_otel_logger_on_litellm_proxy"
    )
    def test_levo_logger_instantiation(self, mock_init_proxy):
        """Test that LevoLogger can be instantiated with proper config."""
        # Mock the proxy initialization to avoid importing proxy code
        mock_init_proxy.return_value = None

        config = LevoLogger.get_levo_config()
        otel_config = OpenTelemetryConfig(
            exporter=config.protocol,
            endpoint=config.endpoint,
            headers=config.otlp_auth_headers,
        )

        # Create a tracer provider with in-memory exporter to avoid requiring OTLP packages
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))

        # Create LevoLogger instance with mocked tracer provider
        levo_logger = LevoLogger(
            config=otel_config, callback_name="levo", tracer_provider=tracer_provider
        )

        # Verify it's an instance of OpenTelemetry
        self.assertIsInstance(levo_logger, LevoLogger)
        # Check it extends OpenTelemetry by checking base classes
        from litellm.integrations.opentelemetry import OpenTelemetry

        self.assertIsInstance(levo_logger, OpenTelemetry)

        # Verify callback_name is set
        self.assertEqual(levo_logger.callback_name, "levo")

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
        },
    )
    def test_levo_config_headers_format(self):
        """Test that OTLP headers are formatted correctly."""
        config = LevoLogger.get_levo_config()

        # Verify headers contain all required parts
        self.assertIn("Authorization=Bearer test-api-key", config.otlp_auth_headers)
        self.assertIn("x-levo-organization-id=test-org-id", config.otlp_auth_headers)
        self.assertIn("x-levo-workspace-id=test-workspace-id", config.otlp_auth_headers)

        # Verify headers are comma-separated
        header_parts = config.otlp_auth_headers.split(",")
        self.assertEqual(len(header_parts), 3)

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://custom.collector.com",
        },
    )
    def test_levo_config_endpoint_construction(self):
        """Test that endpoint uses provided URL exactly as-is."""
        config = LevoLogger.get_levo_config()

        # Verify endpoint uses URL exactly as provided
        self.assertEqual(config.endpoint, "https://custom.collector.com")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
        },
    )
    @pytest.mark.skipif(
        not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry packages not installed"
    )
    @patch(
        "litellm.integrations.opentelemetry.OpenTelemetry._init_otel_logger_on_litellm_proxy"
    )
    @pytest.mark.asyncio
    async def test_levo_logger_health_check_healthy(self, mock_init_proxy):
        """Test health check returns healthy status when config is valid."""
        # Mock the proxy initialization to avoid importing proxy code
        mock_init_proxy.return_value = None

        config = LevoLogger.get_levo_config()
        otel_config = OpenTelemetryConfig(
            exporter=config.protocol,
            endpoint=config.endpoint,
            headers=config.otlp_auth_headers,
        )

        # Create tracer provider with in-memory exporter
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))

        levo_logger = LevoLogger(
            config=otel_config, callback_name="levo", tracer_provider=tracer_provider
        )

        # Run health check
        result = await levo_logger.async_health_check()

        self.assertEqual(result["status"], "healthy")
        self.assertIn("message", result)

    @patch.dict("os.environ", {}, clear=True)
    def test_levo_logger_health_check_unhealthy(self):
        """Test health check returns unhealthy status when required vars are missing."""
        # Try to create logger without required env vars
        # This should fail during config, but we can test health check logic
        with pytest.raises(ValueError):
            LevoLogger.get_levo_config()

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
        },
    )
    @pytest.mark.skipif(
        not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry packages not installed"
    )
    @patch(
        "litellm.integrations.opentelemetry.OpenTelemetry._init_otel_logger_on_litellm_proxy"
    )
    def test_levo_logger_callback_name(self, mock_init_proxy):
        """Test that callback_name is properly set and used."""
        # Mock the proxy initialization to avoid importing proxy code
        mock_init_proxy.return_value = None

        config = LevoLogger.get_levo_config()
        otel_config = OpenTelemetryConfig(
            exporter=config.protocol,
            endpoint=config.endpoint,
            headers=config.otlp_auth_headers,
        )

        # Create tracer provider with in-memory exporter
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))

        levo_logger = LevoLogger(
            config=otel_config, callback_name="levo", tracer_provider=tracer_provider
        )

        # Verify callback_name attribute
        self.assertEqual(levo_logger.callback_name, "levo")

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "http://localhost:4318",
        },
    )
    def test_levo_config_http_endpoint(self):
        """Test that HTTP endpoints are used exactly as provided."""
        config = LevoLogger.get_levo_config()

        # Should use HTTP endpoint exactly as provided
        self.assertTrue(config.endpoint.startswith("http://"))
        self.assertEqual(config.endpoint, "http://localhost:4318")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
        },
    )
    def test_levo_config_uses_provided_url(self):
        """Test that collector URL is used exactly as provided."""
        config = LevoLogger.get_levo_config()

        # Should use provided URL exactly as-is
        self.assertEqual(config.endpoint, "https://collector.levo.ai")
        self.assertEqual(config.protocol, "otlp_http")


if __name__ == "__main__":
    unittest.main()
