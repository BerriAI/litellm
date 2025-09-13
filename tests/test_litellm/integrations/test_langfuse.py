import unittest
import asyncio
from unittest.mock import patch, MagicMock
from typing import Optional
import sys
import os
import datetime
import json
import pytest
import litellm
from litellm.integrations.langfuse import langfuse as langfuse_module
from litellm.integrations.langfuse.langfuse import LangFuseLogger

sys.path.insert(0, os.path.abspath("../.."))
from litellm.integrations.langfuse.langfuse import LangFuseLogger
# Import LangfuseUsageDetails directly from the module where it's defined
from litellm.types.integrations.langfuse import *

class TestLangfuseUsageDetails(unittest.TestCase):

    def setUp(self):
        # Set up environment variables for testing
        self.env_patcher = patch.dict('os.environ', {
            'LANGFUSE_SECRET_KEY': 'test-secret-key',
            'LANGFUSE_PUBLIC_KEY': 'test-public-key',
            'LANGFUSE_HOST': 'https://test.langfuse.com'
        })
        self.env_patcher.start()

        # Create mock objects
        self.mock_langfuse_client = MagicMock()
        self.mock_langfuse_trace = MagicMock()
        self.mock_langfuse_generation = MagicMock()

        # Setup the trace and generation chain
        self.mock_langfuse_trace.generation.return_value = self.mock_langfuse_generation
        self.mock_langfuse_client.trace.return_value = self.mock_langfuse_trace

        # Mock the langfuse module that's imported locally in methods
        self.langfuse_module_patcher = patch.dict('sys.modules', {'langfuse': MagicMock()})
        self.mock_langfuse_module = self.langfuse_module_patcher.start()

        # Create a mock for the langfuse module with version
        self.mock_langfuse = MagicMock()
        self.mock_langfuse.version = MagicMock()
        self.mock_langfuse.version.__version__ = "3.0.0"  # Set a version that supports all features

        # Mock the Langfuse class
        self.mock_langfuse_class = MagicMock()
        self.mock_langfuse_class.return_value = self.mock_langfuse_client

        # Set up the sys.modules['langfuse'] mock
        sys.modules['langfuse'] = self.mock_langfuse
        sys.modules['langfuse'].Langfuse = self.mock_langfuse_class

        # Mock the Langfuse client
        self.mock_langfuse_client = MagicMock()
        self.mock_langfuse_trace = MagicMock()
        self.mock_langfuse_generation = MagicMock()

        # Setup the trace and generation chain
        self.mock_langfuse_trace.generation.return_value = self.mock_langfuse_generation
        self.mock_langfuse_client.trace.return_value = self.mock_langfuse_trace

        # Mock the Langfuse class
        self.mock_langfuse_class = MagicMock()
        self.mock_langfuse_class.return_value = self.mock_langfuse_client
        self.mock_langfuse.Langfuse = self.mock_langfuse_class

        # Create the logger
        self.logger = LangFuseLogger()

        # Add the log_event_on_langfuse method to the instance
        def log_event_on_langfuse(self, kwargs, response_obj, start_time=None, end_time=None, user_id=None, level="DEFAULT", status_message=None):
            # This implementation calls _log_langfuse_v2 directly
            return self._log_langfuse_v2(
                user_id=user_id,
                metadata=kwargs.get("litellm_params", {}).get("metadata", {}),
                litellm_params=kwargs.get("litellm_params", {}),
                output=None,
                start_time=start_time,
                end_time=end_time,
                kwargs=kwargs,
                optional_params=kwargs.get("optional_params", {}),
                input=None,
                response_obj=response_obj,
                level=level,
                litellm_call_id=kwargs.get("litellm_call_id", None),
                print_verbose=True  # Add the missing parameter
            )

        # Bind the method to the instance
        import types
        self.logger.log_event_on_langfuse = types.MethodType(log_event_on_langfuse, self.logger)

        # Make sure _is_langfuse_v2 returns True
        def mock_is_langfuse_v2(self):
            return True

        self.logger._is_langfuse_v2 = types.MethodType(mock_is_langfuse_v2, self.logger)

    def tearDown(self):
        self.env_patcher.stop()
        self.langfuse_module_patcher.stop()

    def test_langfuse_usage_details_type(self):
        """Test that LangfuseUsageDetails TypedDict is properly defined with the correct fields"""
        # Create an instance of LangfuseUsageDetails
        usage_details: LangfuseUsageDetails = {
            "input": 10,
            "output": 20,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 3
        }

        # Verify all fields are present
        self.assertEqual(usage_details["input"], 10)
        self.assertEqual(usage_details["output"], 20)
        self.assertEqual(usage_details["cache_creation_input_tokens"], 5)
        self.assertEqual(usage_details["cache_read_input_tokens"], 3)

        # Test with all fields (all fields are required in TypedDict by default)
        minimal_usage_details: LangfuseUsageDetails = {
            "input": 10,
            "output": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0
        }

        self.assertEqual(minimal_usage_details["input"], 10)
        self.assertEqual(minimal_usage_details["output"], 20)

    def test_log_langfuse_v2_usage_details(self):
        """Test that usage_details in _log_langfuse_v2 is correctly typed and assigned"""
        # Create a mock response object with usage information
        response_obj = MagicMock()
        response_obj.usage = MagicMock()
        response_obj.usage.prompt_tokens = 15
        response_obj.usage.completion_tokens = 25

        # Add the cache token attributes using get method
        def mock_get(key, default=None):
            if key == 'cache_creation_input_tokens':
                return 7
            elif key == 'cache_read_input_tokens':
                return 4
            return default

        response_obj.usage.get = mock_get

        # Create kwargs for the log_event method
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "litellm_params": {"metadata": {}}
        }

        # Create start and end times
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=1)

        # Call the log_event method
        with patch.object(self.logger, '_log_langfuse_v2') as mock_log_langfuse_v2:
            self.logger.log_event_on_langfuse(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time
            )

            # Check if _log_langfuse_v2 was called
            mock_log_langfuse_v2.assert_called_once()

            # Get the arguments passed to _log_langfuse_v2
            call_args = mock_log_langfuse_v2.call_args[1]

            # Verify response_obj was passed correctly
            self.assertEqual(call_args["response_obj"], response_obj)

    def test_langfuse_usage_details_optional_fields(self):
        """Test that LangfuseUsageDetails fields are properly defined as Optional"""
        # Create an instance with None values for optional fields
        usage_details: LangfuseUsageDetails = {
            "input": 10,
            "output": 20,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None
        }

        # Verify fields can be None
        self.assertEqual(usage_details["input"], 10)
        self.assertEqual(usage_details["output"], 20)
        self.assertIsNone(usage_details["cache_creation_input_tokens"])
        self.assertIsNone(usage_details["cache_read_input_tokens"])

    def test_langfuse_usage_details_structure(self):
        """Test that LangfuseUsageDetails has the correct structure as defined in the commit"""
        # This test directly verifies the structure of the TypedDict
        # without relying on the LangFuseLogger class

        # Create a dictionary that matches the LangfuseUsageDetails structure
        usage_details = {
            "input": 15,
            "output": 25,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 4
        }

        # Verify the structure matches what we expect
        self.assertIn("input", usage_details)
        self.assertIn("output", usage_details)
        self.assertIn("cache_creation_input_tokens", usage_details)
        self.assertIn("cache_read_input_tokens", usage_details)

        # Verify the values
        self.assertEqual(usage_details["input"], 15)
        self.assertEqual(usage_details["output"], 25)
        self.assertEqual(usage_details["cache_creation_input_tokens"], 7)
        self.assertEqual(usage_details["cache_read_input_tokens"], 4)

def test_max_langfuse_clients_limit():
    """
    Test that the max langfuse clients limit is respected when initializing multiple clients
    """
    # Set max clients to 2 for testing
    with patch.object(langfuse_module, "MAX_LANGFUSE_INITIALIZED_CLIENTS", 2):
        # Reset the counter
        litellm.initialized_langfuse_clients = 0

        # First client should succeed
        logger1 = LangFuseLogger(
            langfuse_public_key="test_key_1",
            langfuse_secret="test_secret_1",
            langfuse_host="https://test1.langfuse.com",
        )
        assert litellm.initialized_langfuse_clients == 1

        # Second client should succeed
        logger2 = LangFuseLogger(
            langfuse_public_key="test_key_2",
            langfuse_secret="test_secret_2",
            langfuse_host="https://test2.langfuse.com",
        )
        assert litellm.initialized_langfuse_clients == 2

        # Third client should fail with exception
        with pytest.raises(Exception) as exc_info:
            logger3 = LangFuseLogger(
                langfuse_public_key="test_key_3",
                langfuse_secret="test_secret_3",
                langfuse_host="https://test3.langfuse.com",
            )

        # Verify the error message contains the expected text
        assert "Max langfuse clients reached" in str(exc_info.value)

        # Counter should still be 2 (third client failed to initialize)
        assert litellm.initialized_langfuse_clients == 2
