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

        # Verify the configuration
        self.assertEqual(
            config.otlp_auth_headers, "Authorization=Bearer%20test_api_key"
        )
        self.assertEqual(config.endpoint, "http://test.endpoint")
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

        # Verify the configuration
        self.assertEqual(
            config.otlp_auth_headers, "Authorization=Bearer%20test_api_key"
        )
        self.assertEqual(config.endpoint, "grpc://test.endpoint")
        self.assertEqual(config.protocol, "otlp_grpc")


if __name__ == "__main__":
    unittest.main()
