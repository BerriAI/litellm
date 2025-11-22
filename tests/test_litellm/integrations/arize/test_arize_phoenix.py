import unittest
from unittest.mock import patch

from litellm.integrations.arize.arize_phoenix import ArizePhoenixLogger


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


if __name__ == "__main__":
    unittest.main()
