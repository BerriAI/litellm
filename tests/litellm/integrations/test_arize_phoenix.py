import unittest
from unittest.mock import patch
import datetime
import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from litellm.integrations.arize.arize_phoenix import ArizePhoenixLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.integrations.opentelemetry import OpenTelemetry

class TestArizePhoenixLogger(unittest.TestCase):

    def _configure_logging(self, authenticated: bool):
        if authenticated:
            self.env_patcher = patch.dict('os.environ', {
                'PHOENIX_COLLECTOR_HTTP_ENDPOINT': 'http://phoenix:8000/v1/traces',
                'PHOENIX_API_KEY': 'test-api-key'
            })
        else:
            self.env_patcher = patch.dict('os.environ', {
                'PHOENIX_COLLECTOR_HTTP_ENDPOINT': 'http://phoenix:8000/v1/traces'
            })
        self.env_patcher.start()
        # Clear loggers cache
        litellm_logging._in_memory_loggers.clear()
        self.logging = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey"}],
            stream=False,
            call_type="completion",
            start_time=datetime.time(),
            litellm_call_id="12345",
            function_id="1245",
            dynamic_success_callbacks=['arize_phoenix']
        )

        self.assertIsNotNone(self.logging.dynamic_success_callbacks)
        self.assertIsInstance(self.logging.dynamic_success_callbacks, list)
        if self.logging.dynamic_success_callbacks:
            self.assertEqual(len(self.logging.dynamic_success_callbacks), 1)
            cb = self.logging.dynamic_success_callbacks[0]
            self.assertIsInstance(cb, OpenTelemetry)
            if isinstance(cb, OpenTelemetry):
                ot: OpenTelemetry = cb
                self.assertEqual(ot.callback_name, 'arize_phoenix')
                self.assertIsNotNone(ot.config)
                self.assertEqual(ot.config.endpoint, 'http://phoenix:8000/v1/traces')
                self.assertEqual(ot.config.exporter, 'otlp_http')

    def tearDown(self):
        self.env_patcher.stop()

    def test_init_with_authentication(self):
        """Test the initialization of ArizePhoenixLogger"""
        self._configure_logging(True)
        config = ArizePhoenixLogger.get_arize_phoenix_config()
        self.assertEqual(config.protocol, 'otlp_http')
        self.assertEqual(config.endpoint, 'http://phoenix:8000/v1/traces')
        self.assertEqual(config.otlp_auth_headers, 'Authorization=Bearer test-api-key')

    def test_init_without_authentication(self):
        """Test the initialization of ArizePhoenixLogger"""
        self._configure_logging(False)
        config = ArizePhoenixLogger.get_arize_phoenix_config()
        self.assertEqual(config.protocol, 'otlp_http')
        self.assertEqual(config.endpoint, 'http://phoenix:8000/v1/traces')
        self.assertIsNone(config.otlp_auth_headers)

    def test_logging_with_authentication(self):
        """Test the propagation of authentication headers"""
        self._configure_logging(True)
        self.assertIsNotNone(self.logging.dynamic_success_callbacks)
        self.assertIsInstance(self.logging.dynamic_success_callbacks, list)
        if self.logging.dynamic_success_callbacks:
            self.assertEqual(len(self.logging.dynamic_success_callbacks), 1)
            cb = self.logging.dynamic_success_callbacks[0]
            self.assertIsInstance(cb, OpenTelemetry)
            if isinstance(cb, OpenTelemetry):
                ot: OpenTelemetry = cb
                self.assertEqual(ot.callback_name, 'arize_phoenix')
                self.assertIsNotNone(ot.config)
                self.assertEqual(ot.config.endpoint, 'http://phoenix:8000/v1/traces')
                self.assertEqual(ot.config.exporter, 'otlp_http')
                self.assertEqual(ot.config.headers, 'Authorization=Bearer test-api-key')

    def test_logging_without_authentication(self):
        """Test the propagation of authentication headers"""
        self._configure_logging(False)
        self.assertIsNotNone(self.logging.dynamic_success_callbacks)
        self.assertIsInstance(self.logging.dynamic_success_callbacks, list)
        if self.logging.dynamic_success_callbacks:
            self.assertEqual(len(self.logging.dynamic_success_callbacks), 1)
            cb = self.logging.dynamic_success_callbacks[0]
            self.assertIsInstance(cb, OpenTelemetry)
            if isinstance(cb, OpenTelemetry):
                ot: OpenTelemetry = cb
                self.assertEqual(ot.callback_name, 'arize_phoenix')
                self.assertIsNotNone(ot.config)
                self.assertEqual(ot.config.endpoint, 'http://phoenix:8000/v1/traces')
                self.assertEqual(ot.config.exporter, 'otlp_http')
                self.assertIsNone(ot.config.headers)


if __name__ == '__main__':
    unittest.main()
