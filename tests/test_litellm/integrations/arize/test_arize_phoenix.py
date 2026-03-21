import unittest
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixConfig,
    ArizePhoenixLogger,
)
from litellm.integrations.arize._utils import ArizeOTELAttributes


class TestArizePhoenixConfig(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "PHOENIX_API_KEY": "test_api_key",
            "PHOENIX_COLLECTOR_HTTP_ENDPOINT": "http://test.endpoint",
        },
    )
    def test_get_arize_phoenix_config_http(self):
        # Call the function to get the configuration
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Verify the configuration - now uses standard Authorization Bearer format
        self.assertEqual(
            config.otlp_auth_headers, "Authorization=Bearer test_api_key"
        )
        self.assertEqual(config.endpoint, "http://test.endpoint/v1/traces")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "PHOENIX_API_KEY": "test_api_key",
            "PHOENIX_COLLECTOR_ENDPOINT": "grpc://test.endpoint",
        },
    )
    def test_get_arize_phoenix_config_grpc(self):
        # Call the function to get the configuration
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Verify the configuration - now uses standard Authorization Bearer format
        self.assertEqual(
            config.otlp_auth_headers, "Authorization=Bearer test_api_key"
        )
        self.assertEqual(config.endpoint, "grpc://test.endpoint")
        self.assertEqual(config.protocol, "otlp_grpc")

    @patch.dict(
        "os.environ",
        {
            "PHOENIX_API_KEY": "test_api_key",
            "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:6006",
        },
    )
    def test_get_arize_phoenix_config_http_local(self):
        # Test with local Phoenix instance
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Should automatically append /v1/traces to local endpoint
        self.assertEqual(
            config.otlp_auth_headers, "Authorization=Bearer test_api_key"
        )
        self.assertEqual(config.endpoint, "http://localhost:6006/v1/traces")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317",
        },
        clear=True
    )
    def test_get_arize_phoenix_config_grpc_no_api_key(self):
        # Test gRPC endpoint detection and no API key (for local development)
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # No API key should be fine for local development
        self.assertIsNone(config.otlp_auth_headers)
        self.assertEqual(config.endpoint, "http://localhost:4317")
        self.assertEqual(config.protocol, "otlp_grpc")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_arize_phoenix_config_defaults_to_local(self):
        # Test that it defaults to local Phoenix when no config is provided
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Should default to localhost
        self.assertEqual(config.endpoint, "http://localhost:6006/v1/traces")
        self.assertEqual(config.protocol, "otlp_http")
        # No auth headers when no API key is provided for local instance
        self.assertIsNone(config.otlp_auth_headers)



@pytest.mark.parametrize(
    "env_vars, expected_headers, expected_endpoint, expected_protocol",
    [
        pytest.param(
            {"PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "http://localhost:6006/v1/traces",
            "otlp_http",
            id="default to http protocol and self-hosted Phoenix endpoint",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "http://localhost:6006/v1/traces",
            "otlp_http",
            id="empty string/unset endpoint will default to http protocol and self-hosted Phoenix endpoint",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "http://localhost:4318", "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "http://localhost:4318/v1/traces",
            "otlp_http",
            id="prioritize http if both endpoints are set",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "https://localhost:6006/v1/traces",
            "otlp_http",
            id="custom https endpoint treated as http",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006"},
            None,
            "https://localhost:6006/v1/traces",
            "otlp_http",
            id="custom https endpoint with no auth treated as http",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "grpc://localhost:6006", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "grpc://localhost:6006",
            "otlp_grpc",
            id="explicit grpc endpoint with grpc:// prefix",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317"},
            None,
            "http://localhost:4317",
            "otlp_grpc",
            id="grpc endpoint with standard grpc port 4317",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "https://localhost:6006", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "https://localhost:6006/v1/traces",
            "otlp_http",
            id="custom http endpoint",
        ),
    ],
)
def test_get_arize_phoenix_config(monkeypatch, env_vars, expected_headers, expected_endpoint, expected_protocol):
    # Clear all Phoenix-related env vars first to ensure clean state
    for key in ["PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_COLLECTOR_HTTP_ENDPOINT"]:
        monkeypatch.delenv(key, raising=False)
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = ArizePhoenixLogger.get_arize_phoenix_config()

    assert isinstance(config, ArizePhoenixConfig)
    assert config.otlp_auth_headers == expected_headers
    assert config.endpoint == expected_endpoint
    assert config.protocol == expected_protocol

@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/v1/traces"},
            id="missing api_key with explicit Arize Phoenix Cloud endpoint"
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "https://app.phoenix.arize.com/v1/traces"},
            id="missing api_key with HTTP Arize Phoenix Cloud endpoint"
        ),
    ],
)
def test_get_arize_phoenix_config_expection_on_missing_api_key(monkeypatch, env_vars):
    # Clear all Phoenix-related env vars first to ensure clean state
    for key in ["PHOENIX_API_KEY", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_COLLECTOR_HTTP_ENDPOINT"]:
        monkeypatch.delenv(key, raising=False)
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match="PHOENIX_API_KEY must be set when using Phoenix Cloud"):
        ArizePhoenixLogger.get_arize_phoenix_config()



# ---------------------------------------------------------------------------
# Dynamic project naming from metadata
# ---------------------------------------------------------------------------


class TestGetDynamicProjectName:
    """Tests for _get_dynamic_project_name extraction logic."""

    def test_extracts_from_standard_logging_object_metadata(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "my-project"},
            }
        }
        assert ArizePhoenixLogger._get_dynamic_project_name(kwargs) == "my-project"

    def test_extracts_from_litellm_params_metadata(self):
        kwargs = {
            "litellm_params": {
                "metadata": {"phoenix_project_name": "sdk-project"},
            }
        }
        assert ArizePhoenixLogger._get_dynamic_project_name(kwargs) == "sdk-project"

    def test_returns_none_when_no_metadata(self):
        assert ArizePhoenixLogger._get_dynamic_project_name({}) is None

    def test_non_dict_standard_logging_object_does_not_raise(self):
        """isinstance(dict) guard prevents AttributeError on non-dict payloads."""
        kwargs = {"standard_logging_object": "not-a-dict"}
        assert ArizePhoenixLogger._get_dynamic_project_name(kwargs) is None


class TestDynamicProjectNameOnSpan:
    """set_arize_phoenix_attributes sets openinference.project.name on the span."""

    @patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": "env-fallback"}, clear=False)
    @patch("litellm.integrations.arize._utils.set_attributes")
    def test_dynamic_name_sets_span_attribute(self, _mock_set_attrs):
        span = MagicMock()
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "dynamic-proj"},
            }
        }
        ArizePhoenixLogger.set_arize_phoenix_attributes(span, kwargs, response_obj=None)

        span.set_attribute.assert_called_once_with("openinference.project.name", "dynamic-proj")

    @patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": "env-project"}, clear=False)
    @patch("litellm.integrations.arize._utils.set_attributes")
    def test_falls_back_to_env_var_when_no_dynamic_name(self, _mock_set_attrs):
        span = MagicMock()
        ArizePhoenixLogger.set_arize_phoenix_attributes(span, {}, response_obj=None)

        span.set_attribute.assert_called_once_with("openinference.project.name", "env-project")


if __name__ == "__main__":
    unittest.main()
