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


@pytest.mark.parametrize(
    "error_message,expected_exception_type",
    [
        ("server overloaded, please try again later", RateLimitError),
        ("some other error", litellm.exceptions.APIError),
    ],
)
def test_fireworks_ai_exception_mapping(error_message, expected_exception_type):
    """
    Test that the exception mapping correctly handles Fireworks AI errors
    """
    # Create a mock exception with the error message
    mock_exception = MagicMock()
    mock_exception.message = error_message
    
    # Create a mock for the exception mapping function
    with patch("litellm.litellm_core_utils.exception_mapping_utils.get_error_message", return_value=error_message):
        try:
            # Import the exception_type function
            from litellm.litellm_core_utils.exception_mapping_utils import exception_type
            
            # Create a mock original exception
            original_exception = MagicMock()
            original_exception.message = error_message
            original_exception.__str__ = MagicMock(return_value=error_message)
            
            # Call the exception mapping function
            result = exception_type(
                model="accounts/fireworks/models/llama4-maverick-instruct-basic",
                original_exception=original_exception,
                custom_llm_provider="fireworks_ai",
            )
            
            # This should not be reached as the function should raise an exception
            pytest.fail(f"Expected an exception but got: {result}")
                
        except Exception as e:
            # Check that the exception is of the expected type
            assert isinstance(e, expected_exception_type), f"Expected {expected_exception_type}, got {type(e)}"
            # For RateLimitError, check that the message contains the expected text
            if expected_exception_type == RateLimitError:
                assert "server overloaded" in str(e), f"Expected 'server overloaded' in error message, got: {str(e)}"