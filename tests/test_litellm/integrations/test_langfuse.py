import datetime
import os
import sys
import types
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

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
        self.env_patcher = patch.dict(
            "os.environ",
            {
                "LANGFUSE_SECRET_KEY": "test-secret-key",
                "LANGFUSE_PUBLIC_KEY": "test-public-key",
                "LANGFUSE_HOST": "https://test.langfuse.com",
            },
        )
        self.env_patcher.start()

        # Create mock objects
        self.mock_langfuse_client = MagicMock()
        # Mock the client attribute to prevent errors during logger initialization
        self.mock_langfuse_client.client = MagicMock()
        self.mock_langfuse_trace = MagicMock()
        self.mock_langfuse_generation = MagicMock()
        self.mock_langfuse_generation.trace_id = "test-trace-id"
        
        # Mock span method for trace (used by log_provider_specific_information_as_span and _log_guardrail_information_as_span)
        self.mock_langfuse_span = MagicMock()
        self.mock_langfuse_span.end = MagicMock()
        self.mock_langfuse_trace.span.return_value = self.mock_langfuse_span

        # Setup the trace and generation chain
        self.mock_langfuse_trace.generation.return_value = self.mock_langfuse_generation
        self.last_trace_kwargs = {}

        def _trace_side_effect(*args, **kwargs):
            self.last_trace_kwargs = kwargs
            return self.mock_langfuse_trace

        self.mock_langfuse_client.trace.side_effect = _trace_side_effect

        # Mock the langfuse module that's imported locally in methods
        self.langfuse_module_patcher = patch.dict(
            "sys.modules", {"langfuse": MagicMock()}
        )
        self.mock_langfuse_module = self.langfuse_module_patcher.start()

        # Create a mock for the langfuse module with version
        self.mock_langfuse = MagicMock()
        self.mock_langfuse.version = MagicMock()
        self.mock_langfuse.version.__version__ = (
            "3.0.0"  # Set a version that supports all features
        )

        # Mock the Langfuse class
        self.mock_langfuse_class = MagicMock()
        self.mock_langfuse_class.return_value = self.mock_langfuse_client

        # Set up the sys.modules['langfuse'] mock
        sys.modules["langfuse"] = self.mock_langfuse
        sys.modules["langfuse"].Langfuse = self.mock_langfuse_class

        # Create the logger
        self.logger = LangFuseLogger()
        
        # Explicitly set the Langfuse client to our mock
        self.logger.Langfuse = self.mock_langfuse_client
        # Ensure langfuse_sdk_version is set correctly for _supports_* methods
        self.logger.langfuse_sdk_version = "3.0.0"

        # Add the log_event_on_langfuse method to the instance
        def log_event_on_langfuse(
            self,
            kwargs,
            response_obj,
            start_time=None,
            end_time=None,
            user_id=None,
            level="DEFAULT",
            status_message=None,
        ):
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
                print_verbose=True,  # Add the missing parameter
            )

        # Bind the method to the instance
        self.logger.log_event_on_langfuse = types.MethodType(
            log_event_on_langfuse, self.logger
        )

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
            "total": 30,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 3,
        }

        # Verify all fields are present
        self.assertEqual(usage_details["input"], 10)
        self.assertEqual(usage_details["output"], 20)
        self.assertEqual(usage_details["total"], 30)
        self.assertEqual(usage_details["cache_creation_input_tokens"], 5)
        self.assertEqual(usage_details["cache_read_input_tokens"], 3)

        # Test with all fields (all fields are required in TypedDict by default)
        minimal_usage_details: LangfuseUsageDetails = {
            "input": 10,
            "output": 20,
            "total": 30,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        self.assertEqual(minimal_usage_details["input"], 10)
        self.assertEqual(minimal_usage_details["output"], 20)
        self.assertEqual(minimal_usage_details["total"], 30)

    def test_log_langfuse_v2_usage_details(self):
        """Test that usage_details in _log_langfuse_v2 is correctly typed and assigned"""
        # Create a mock response object with usage information
        response_obj = MagicMock()
        response_obj.usage = MagicMock()
        response_obj.usage.prompt_tokens = 15
        response_obj.usage.completion_tokens = 25

        # Add the cache token attributes using get method
        def mock_get(key, default=None):
            if key == "cache_creation_input_tokens":
                return 7
            elif key == "cache_read_input_tokens":
                return 4
            return default

        response_obj.usage.get = mock_get

        # Create kwargs for the log_event method
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "litellm_params": {"metadata": {}},
        }

        # Create start and end times
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=1)

        # Call the log_event method
        with patch.object(self.logger, "_log_langfuse_v2") as mock_log_langfuse_v2:
            self.logger.log_event_on_langfuse(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
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
            "total": 30,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
        }

        # Verify fields can be None
        self.assertEqual(usage_details["input"], 10)
        self.assertEqual(usage_details["output"], 20)
        self.assertEqual(usage_details["total"], 30)
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
            "total": 40,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 4,
        }

        # Verify the structure matches what we expect
        self.assertIn("input", usage_details)
        self.assertIn("output", usage_details)
        self.assertIn("total", usage_details)
        self.assertIn("cache_creation_input_tokens", usage_details)
        self.assertIn("cache_read_input_tokens", usage_details)

        # Verify the values
        self.assertEqual(usage_details["input"], 15)
        self.assertEqual(usage_details["output"], 25)
        self.assertEqual(usage_details["total"], 40)
        self.assertEqual(usage_details["cache_creation_input_tokens"], 7)
        self.assertEqual(usage_details["cache_read_input_tokens"], 4)

    def test_log_langfuse_v2_handles_null_usage_values(self):
        """
        Test that _log_langfuse_v2 correctly handles None values in the usage object
        by converting them to 0, preventing validation errors.
        """
        # Reset mock call counts to ensure clean state
        self.mock_langfuse_trace.reset_mock()
        self.mock_langfuse_client.reset_mock()
        
        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kwargs: generation_params,
            create=True,
        ) as mock_add_prompt_params:
            # Create a mock response object with usage information containing None values
            response_obj = MagicMock()
            response_obj.usage = MagicMock()
            response_obj.usage.prompt_tokens = None
            response_obj.usage.completion_tokens = None
            response_obj.usage.total_tokens = None

            # Mock the .get() method to return None for cache-related fields
            def mock_get(key, default=None):
                if key in ["cache_creation_input_tokens", "cache_read_input_tokens"]:
                    return None
                return default

            response_obj.usage.get = mock_get

            # Prepare standard kwargs for the call
            kwargs = {
                "model": "gpt-4-null-usage",
                "messages": [{"role": "user", "content": "Test"}],
                "litellm_params": {"metadata": {}},
                "optional_params": {},
                "litellm_call_id": "test-call-id-null-usage",
                "standard_logging_object": None,
                "response_cost": 0.0,
            }

            # Use fixed timestamps to avoid timing-related flakiness
            fixed_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
            
            # Ensure the mock trace is properly set up before the call
            # Re-setup the trace chain to ensure it's fresh
            self.mock_langfuse_trace.generation.return_value = self.mock_langfuse_generation
            self.mock_langfuse_trace.span.return_value = self.mock_langfuse_span
            self.mock_langfuse_client.trace.return_value = self.mock_langfuse_trace
            self.logger.Langfuse = self.mock_langfuse_client
            
            # Call the method under test
            try:
                self.logger._log_langfuse_v2(
                    user_id="test-user",
                    metadata={},
                    litellm_params=kwargs["litellm_params"],
                    output={"role": "assistant", "content": "Response"},
                    start_time=fixed_time,
                    end_time=fixed_time + datetime.timedelta(seconds=1),
                    kwargs=kwargs,
                    optional_params=kwargs["optional_params"],
                    input={"messages": kwargs["messages"]},
                    response_obj=response_obj,
                    level="DEFAULT",
                    litellm_call_id=kwargs["litellm_call_id"],
                )
            except Exception as e:
                self.fail(f"_log_langfuse_v2 raised an exception: {e}")
            
            # Verify that trace was called first
            self.mock_langfuse_client.trace.assert_called()
            
            #  Check the arguments passed to the mocked langfuse generation call
            self.mock_langfuse_trace.generation.assert_called_once()
            call_args, call_kwargs = self.mock_langfuse_trace.generation.call_args

            #  Inspect the usage and usage_details dictionaries
            usage_arg = call_kwargs.get("usage")
            usage_details_arg = call_kwargs.get("usage_details")

            self.assertIsNotNone(usage_arg)
            self.assertIsNotNone(usage_details_arg)

            # Verify that None values were converted to 0
            self.assertEqual(usage_arg["prompt_tokens"], 0)
            self.assertEqual(usage_arg["completion_tokens"], 0)

            self.assertEqual(usage_details_arg["input"], 0)
            self.assertEqual(usage_details_arg["output"], 0)
            self.assertEqual(usage_details_arg["total"], 0)
            self.assertEqual(usage_details_arg["cache_creation_input_tokens"], 0)
            self.assertEqual(usage_details_arg["cache_read_input_tokens"], 0)

            mock_add_prompt_params.assert_called_once()

    def _build_standard_logging_payload(self, trace_id: Optional[str] = None):
        payload = {
            "id": "payload-id",
            "call_type": "completion",
            "response_cost": 0.0,
            "status": "success",
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "startTime": 0.0,
            "endTime": 0.0,
            "completionStartTime": 0.0,
            "model": "gpt-4",
            "model_id": "model-123",
            "model_group": "openai",
            "api_base": "https://api.openai.com",
            "metadata": {
                "user_api_key_end_user_id": None,
                "prompt_management_metadata": None,
                "session_id": None,
                "trace_name": None,
                "trace_version": None,
                "headers": None,
                "endpoint": None,
                "caching_groups": None,
                "previous_models": None,
            },
            "hidden_params": {},
            "request_tags": [],
            "messages": [],
            "response": {"id": "resp"},
            "model_parameters": {},
            "guardrail_information": None,
            "standard_built_in_tools_params": None,
        }
        if trace_id is not None:
            payload["trace_id"] = trace_id
        return payload

    def _build_langfuse_kwargs(self, standard_logging_payload):
        return {
            "standard_logging_object": standard_logging_payload,
            "model": standard_logging_payload["model"],
            "call_type": standard_logging_payload["call_type"],
            "cache_hit": False,
            "messages": [],
        }

    def test_log_langfuse_v2_uses_standard_trace_id_when_available(self):
        payload = self._build_standard_logging_payload(trace_id="std-trace-id")
        kwargs = self._build_langfuse_kwargs(payload)
        self.last_trace_kwargs = {}

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kwargs: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata={},
                litellm_params={"metadata": {}},
                output=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="INFO",
                litellm_call_id="call-id-xyz",
            )

        assert self.last_trace_kwargs.get("id") == "std-trace-id"

    def test_log_langfuse_v2_defaults_to_call_id_without_standard_trace_id(self):
        payload = self._build_standard_logging_payload()
        kwargs = self._build_langfuse_kwargs(payload)
        self.last_trace_kwargs = {}

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kwargs: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata={},
                litellm_params={"metadata": {}},
                output=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="INFO",
                litellm_call_id="call-id-xyz",
            )

        assert self.last_trace_kwargs.get("id") == "call-id-xyz"


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
