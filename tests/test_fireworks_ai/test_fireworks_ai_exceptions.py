import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import pytest
import httpx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.fireworks_ai.common_utils import FireworksAIMixin
from litellm.exceptions import RateLimitError, BadRequestError


class TestFireworksAIExceptions(unittest.TestCase):
    def setUp(self):
        self.fireworks_mixin = FireworksAIMixin()

    def test_rate_limit_error(self):
        """Test that a 429 error is properly translated to a RateLimitError"""
        error_message = "server overloaded, please try again later"
        status_code = 429
        headers = {}

        exception = self.fireworks_mixin.get_error_class(
            error_message=error_message,
            status_code=status_code,
            headers=headers,
        )

        self.assertIsInstance(exception, RateLimitError)
        self.assertEqual(exception.llm_provider, "fireworks_ai")
        self.assertIn("Fireworks_aiException", exception.message)
        self.assertIn(error_message, exception.message)

    def test_bad_request_error(self):
        """Test that a 400 error is properly translated to a FireworksAIException"""
        error_message = "Invalid request"
        status_code = 400
        headers = {}

        exception = self.fireworks_mixin.get_error_class(
            error_message=error_message,
            status_code=status_code,
            headers=headers,
        )

        self.assertEqual(exception.status_code, 400)
        self.assertEqual(exception.message, error_message)