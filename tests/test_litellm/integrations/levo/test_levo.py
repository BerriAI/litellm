import unittest
from unittest.mock import patch

import pytest

from litellm.integrations.levo.levo import LevoConfig, LevoLogger
from litellm.integrations.opentelemetry import OpenTelemetryConfig

# Try to import OpenTelemetry packages, skip tests if not available
try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class TestLevoConfig(unittest.TestCase):
    """Unit tests for LevoLogger configuration."""

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
        },
    )
    def test_get_levo_config_with_all_required_vars(self):
        """Test get_levo_config() with all required environment variables."""
        config = LevoLogger.get_levo_config()

        # Verify headers include all three values
        self.assertIn("Authorization=Bearer test-api-key", config.otlp_auth_headers)
        self.assertIn("x-levo-organization-id=test-org-id", config.otlp_auth_headers)
        self.assertIn("x-levo-workspace-id=test-workspace-id", config.otlp_auth_headers)

        # Verify endpoint uses provided collector URL exactly as-is
        self.assertEqual(config.endpoint, "https://collector.levo.ai")

        # Verify protocol is otlp_http
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "https://custom.collector.com",
        },
    )
    def test_get_levo_config_with_custom_collector_url(self):
        """Test get_levo_config() with custom collector URL."""
        config = LevoLogger.get_levo_config()

        # Verify endpoint uses custom URL exactly as provided
        self.assertEqual(config.endpoint, "https://custom.collector.com")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_levo_config_missing_api_key(self):
        """Test get_levo_config() raises ValueError when LEVOAI_API_KEY is missing."""
        with pytest.raises(ValueError, match="LEVOAI_API_KEY"):
            LevoLogger.get_levo_config()

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
        },
        clear=True,
    )
    def test_get_levo_config_missing_org_id(self):
        """Test get_levo_config() raises ValueError when LEVOAI_ORG_ID is missing."""
        with pytest.raises(ValueError, match="LEVOAI_ORG_ID"):
            LevoLogger.get_levo_config()

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
        },
        clear=True,
    )
    def test_get_levo_config_missing_workspace_id(self):
        """Test get_levo_config() raises ValueError when LEVOAI_WORKSPACE_ID is missing."""
        with pytest.raises(ValueError, match="LEVOAI_WORKSPACE_ID"):
            LevoLogger.get_levo_config()

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
        },
        clear=True,
    )
    def test_get_levo_config_missing_collector_url(self):
        """Test get_levo_config() raises ValueError when LEVOAI_COLLECTOR_URL is missing."""
        with pytest.raises(ValueError, match="LEVOAI_COLLECTOR_URL"):
            LevoLogger.get_levo_config()

    @patch.dict(
        "os.environ",
        {
            "LEVOAI_API_KEY": "test-api-key",
            "LEVOAI_ORG_ID": "test-org-id",
            "LEVOAI_WORKSPACE_ID": "test-workspace-id",
            "LEVOAI_COLLECTOR_URL": "http://localhost:4318",
        },
    )
    def test_get_levo_config_with_http_endpoint(self):
        """Test get_levo_config() with HTTP endpoint."""
        config = LevoLogger.get_levo_config()

        # Should use HTTP endpoint exactly as provided
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


class TestLevoIntegration(unittest.TestCase):
    """Integration tests for LevoLogger."""
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


@pytest.mark.parametrize(
    "env_vars, expected_headers_contains, expected_endpoint, expected_protocol",
    [
        pytest.param(
            {
                "LEVOAI_API_KEY": "test-key",
                "LEVOAI_ORG_ID": "test-org",
                "LEVOAI_WORKSPACE_ID": "test-workspace",
                "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
            },
            [
                "Authorization=Bearer test-key",
                "x-levo-organization-id=test-org",
                "x-levo-workspace-id=test-workspace",
            ],
            "https://collector.levo.ai",
            "otlp_http",
            id="collector URL with all required vars",
        ),
        pytest.param(
            {
                "LEVOAI_API_KEY": "key-123",
                "LEVOAI_ORG_ID": "org-456",
                "LEVOAI_WORKSPACE_ID": "workspace-789",
                "LEVOAI_COLLECTOR_URL": "https://custom.example.com",
            },
            [
                "Authorization=Bearer key-123",
                "x-levo-organization-id=org-456",
                "x-levo-workspace-id=workspace-789",
            ],
            "https://custom.example.com",
            "otlp_http",
            id="custom collector URL",
        ),
        pytest.param(
            {
                "LEVOAI_API_KEY": "key-123",
                "LEVOAI_ORG_ID": "org-456",
                "LEVOAI_WORKSPACE_ID": "workspace-789",
                "LEVOAI_COLLECTOR_URL": "http://localhost:9999",
            },
            ["Authorization=Bearer key-123"],
            "http://localhost:9999",
            "otlp_http",
            id="custom HTTP endpoint",
        ),
    ],
)
def test_get_levo_config_parametrized(
    monkeypatch,
    env_vars,
    expected_headers_contains,
    expected_endpoint,
    expected_protocol,
):
    """Parametrized tests for get_levo_config() with various configurations."""
    # Clear all Levo-related env vars first to ensure clean state
    for key in [
        "LEVOAI_API_KEY",
        "LEVOAI_ORG_ID",
        "LEVOAI_WORKSPACE_ID",
        "LEVOAI_COLLECTOR_URL",
        "LEVOAI_ENV_NAME",
    ]:
        monkeypatch.delenv(key, raising=False)

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = LevoLogger.get_levo_config()

    assert isinstance(config, LevoConfig)
    assert config.endpoint == expected_endpoint
    assert config.protocol == expected_protocol

    # Verify all expected header parts are present
    for header_part in expected_headers_contains:
        assert header_part in config.otlp_auth_headers


@pytest.mark.parametrize(
    "missing_var",
    [
        pytest.param("LEVOAI_API_KEY", id="missing API key"),
        pytest.param("LEVOAI_ORG_ID", id="missing org ID"),
        pytest.param("LEVOAI_WORKSPACE_ID", id="missing workspace ID"),
        pytest.param("LEVOAI_COLLECTOR_URL", id="missing collector URL"),
    ],
)
def test_get_levo_config_missing_required_vars(monkeypatch, missing_var):
    """Test that missing required environment variables raise ValueError."""
    # Clear all Levo-related env vars
    for key in [
        "LEVOAI_API_KEY",
        "LEVOAI_ORG_ID",
        "LEVOAI_WORKSPACE_ID",
        "LEVOAI_COLLECTOR_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Set all required vars except the missing one
    required_vars = {
        "LEVOAI_API_KEY": "test-key",
        "LEVOAI_ORG_ID": "test-org",
        "LEVOAI_WORKSPACE_ID": "test-workspace",
        "LEVOAI_COLLECTOR_URL": "https://collector.levo.ai",
    }
    required_vars.pop(missing_var)

    for key, value in required_vars.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match=missing_var):
        LevoLogger.get_levo_config()


if __name__ == "__main__":
    unittest.main()
