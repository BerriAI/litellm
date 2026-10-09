import os
import sys
import unittest
from unittest.mock import patch

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import litellm
from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger


class TestServiceLoggerOTEL(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset callbacks before each test
        litellm.service_callback = []
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-123"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-123"

    @patch("litellm.integrations.opentelemetry.OpenTelemetry._init_tracing")
    @patch("litellm.integrations.opentelemetry.OpenTelemetry._init_metrics")
    @patch("litellm.integrations.opentelemetry.OpenTelemetry._init_logs")
    async def test_langfuse_otel_ignores_service_logs(
        self, mock_logs, mock_metrics, mock_tracing
    ):
        """
        Test that LangfuseOtelLogger overrides the service logging hooks with 'pass'.
        """
        logger = LangfuseOtelLogger()

        # Verify hooks are overriden
        self.assertEqual(
            logger.async_service_success_hook.__qualname__,
            "LangfuseOtelLogger.async_service_success_hook",
        )
        self.assertEqual(
            logger.async_service_failure_hook.__qualname__,
            "LangfuseOtelLogger.async_service_failure_hook",
        )


if __name__ == "__main__":
    unittest.main()
