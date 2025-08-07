import os
import unittest
from unittest.mock import patch

from litellm.integrations.braintrust_logging import BraintrustLogger

class TestBraintrustLogger(unittest.TestCase):
    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-env-api-key"})
    @patch.dict(os.environ, {"BRAINTRUST_API_BASE": "https://test-env-api.com/v1"})
    def test_init_with_env_var(self):
        """Test BraintrustLogger initialization with environment variable."""
        logger = BraintrustLogger()
        self.assertEqual(logger.api_key, "test-env-api-key")
        self.assertEqual(logger.api_base, "https://test-env-api.com/v1")
        self.assertEqual(logger.headers["Authorization"], "Bearer test-env-api-key")
        self.assertEqual(logger.headers["Content-Type"], "application/json")

    def test_init_with_explicit_params(self):
        """Test BraintrustLogger initialization with explicit parameters."""
        logger = BraintrustLogger(api_key="explicit-key", api_base="https://custom-api.com/v1")
        self.assertEqual(logger.api_key, "explicit-key")
        self.assertEqual(logger.api_base, "https://custom-api.com/v1")
        self.assertEqual(logger.headers["Authorization"], "Bearer explicit-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_api_key(self):
        """Test BraintrustLogger initialization fails without API key."""
        with self.assertRaises(Exception) as context:
            BraintrustLogger()
        self.assertIn("Missing keys=['BRAINTRUST_API_KEY']", str(context.exception))

    def test_validate_environment_with_api_key(self):
        """Test validate_environment method with valid API key."""
        logger = BraintrustLogger(api_key="test-key")
        # Should not raise an exception
        logger.validate_environment(api_key="test-key")

    def test_validate_environment_missing_api_key(self):
        """Test validate_environment method with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception) as context:
                BraintrustLogger(api_key=None)
            self.assertIn("Missing keys=['BRAINTRUST_API_KEY']", str(context.exception))