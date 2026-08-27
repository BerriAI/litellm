import datetime
import json
import sys
import types
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.integrations.langfuse import langfuse as langfuse_module
from litellm.integrations.langfuse.langfuse import LangFuseLogger


# Import LangfuseUsageDetails directly from the module where it's defined
from litellm.types.integrations.langfuse import *


class TestLangfuseUsageDetails(unittest.TestCase):
    def setUp(self):
        # Save global Langfuse client counter to restore after test
        self._original_langfuse_clients_count = litellm.initialized_langfuse_clients

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

        # Create a fresh logger instance for each test
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
        # Clean up logger instance to prevent state leakage
        if hasattr(self, "logger"):
            # Reset logger's Langfuse client to break any references
            self.logger.Langfuse = None
            # Delete logger instance to ensure complete cleanup
            del self.logger

        # Restore global Langfuse client counter to prevent cross-test pollution
        litellm.initialized_langfuse_clients = self._original_langfuse_clients_count

        self.env_patcher.stop()
        self.langfuse_module_patcher.stop()  # patch.dict automatically restores sys.modules

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
        # Reset the mock to ensure clean state; clear side_effect so return_value takes effect
        self.mock_langfuse_client.reset_mock(side_effect=True)
        self.mock_langfuse_trace.reset_mock(side_effect=True)
        self.mock_langfuse_generation.reset_mock(side_effect=True)

        # Re-setup the trace and generation chain with clean state
        self.mock_langfuse_generation.trace_id = "test-trace-id"
        mock_span = MagicMock()
        mock_span.end = MagicMock()
        self.mock_langfuse_trace.span.return_value = mock_span
        self.mock_langfuse_trace.generation.return_value = self.mock_langfuse_generation

        # Ensure trace returns our mock
        self.mock_langfuse_client.trace.return_value = self.mock_langfuse_trace
        self.logger.Langfuse = self.mock_langfuse_client

        with (
            patch(
                "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
                side_effect=lambda generation_params, **kwargs: generation_params,
                create=True,
            ) as mock_add_prompt_params,
            patch.object(self.logger, "_supports_prompt", return_value=True),
        ):
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
                "standard_logging_object": self._build_standard_logging_payload(),
                "response_cost": 0.0,
            }

            # Use fixed timestamps to avoid timing-related flakiness
            fixed_time = datetime.datetime(2024, 1, 1, 12, 0, 0)

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
            # only real StandardLoggingMetadata fields: session_id, trace_name,
            # headers and friends are request-metadata keys the allowlist drops,
            # so a payload carrying them cannot occur in production
            "metadata": {
                "user_api_key_end_user_id": None,
                "prompt_management_metadata": None,
                "user_api_key_hash": "hashed-key",
                "user_api_key_alias": "canary-alias",
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

    def test_log_langfuse_v2_uses_litellm_trace_id_fallback_over_call_id(self):
        """
        When standard_logging_object has no trace_id, but kwargs contains
        litellm_trace_id (the same ID the DB stores as Session ID), Langfuse
        should use litellm_trace_id — NOT litellm_call_id. This ensures the
        trace_id in Langfuse matches the Session ID shown in LiteLLM logs.
        """
        payload = self._build_standard_logging_payload()  # no trace_id
        kwargs = self._build_langfuse_kwargs(payload)
        kwargs["litellm_trace_id"] = "trace-id-from-kwargs"
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
                level="ERROR",
                litellm_call_id="call-id-xyz",
            )

        # litellm_trace_id should be preferred over litellm_call_id
        assert self.last_trace_kwargs.get("id") == "trace-id-from-kwargs"

    CANARY = "sk-lf-canary-SECRET-d4e5f6"

    def _canary_request_metadata(self):
        """Raw request metadata shaped like the proxy builds it, credentials included."""
        from litellm.proxy._types import UserAPIKeyAuth

        team_logging = [
            {
                "callback_name": "langfuse",
                "callback_vars": {"langfuse_secret_key": self.CANARY},
            }
        ]
        return {
            "user_api_key_auth": UserAPIKeyAuth(
                api_key="hashed-key",
                team_metadata={"logging": team_logging},
            ),
            "user_api_key_team_metadata": {"logging": team_logging},
            "user_api_key_metadata": {"secret_manager_settings": {"vault_token": self.CANARY}},
            "session_id": "canary-session",
            "trace_name": "canary-trace",
            "first_custom": "keep-first",
            "second_custom": "keep-second",
            "endpoint": "/v1/chat/completions",
            "headers": {"authorization": f"Bearer {self.CANARY}"},
        }

    def _emitted_payload_text(self):
        """Every blob this logger handed to the langfuse SDK, as one searchable string."""
        import json

        blobs = [self.last_trace_kwargs]
        if self.mock_langfuse_trace.generation.call_args is not None:
            blobs.append(self.mock_langfuse_trace.generation.call_args.kwargs)
        blobs.extend(call.kwargs for call in self.mock_langfuse_trace.span.call_args_list)
        return json.dumps(blobs, default=repr)

    def _drive_with_canary(self, extra_metadata=None, hidden_params=None):
        metadata = {**self._canary_request_metadata(), **(extra_metadata or {})}
        payload = self._build_standard_logging_payload(trace_id="canary-trace-id")
        if hidden_params is not None:
            payload["hidden_params"] = hidden_params
        kwargs = {**self._build_langfuse_kwargs(payload), "response_cost": 0.25}
        self.last_trace_kwargs = {}
        self.mock_langfuse_trace.generation.reset_mock()
        self.mock_langfuse_trace.span.reset_mock()

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kw: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata=metadata,
                litellm_params={"metadata": metadata},
                output=None,
                start_time=datetime.datetime(2024, 1, 1, 12, 0, 0),
                end_time=datetime.datetime(2024, 1, 1, 12, 0, 1),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="INFO",
                litellm_call_id="canary-call-id",
            )
        return self.mock_langfuse_trace.generation.call_args.kwargs["metadata"]

    def test_team_callback_credentials_never_reach_langfuse(self):
        """
        Regression for the credential leak: request metadata carries the whole
        UserAPIKeyAuth object, whose team_metadata holds the customer's own langfuse
        keys. The emitted blob is sourced from StandardLoggingPayload, so none of the
        three credential carriers can ride along.
        """
        generation_metadata = self._drive_with_canary()

        assert self.CANARY not in self._emitted_payload_text()
        for leaked_key in (
            "user_api_key_auth",
            "user_api_key_team_metadata",
            "user_api_key_metadata",
        ):
            assert leaked_key not in generation_metadata

    def test_debug_langfuse_dump_carries_no_credentials(self):
        """
        debug_langfuse dumps request metadata into the trace as a second emit site.
        It must be sourced from the allowlisted payload too.
        """
        self._drive_with_canary(extra_metadata={"debug_langfuse": True})

        dumped = self.last_trace_kwargs["metadata"]["metadata_passed_to_litellm"]
        assert "user_api_key_auth" not in dumped
        assert self.CANARY not in self._emitted_payload_text()

    def test_raw_request_metadata_reaches_the_emitted_blob_through_no_key(self):
        """
        The emitted blob is the allowlist plus litellm enrichments, nothing else.
        Nothing from raw request metadata is copied across, whatever its type, which
        is what makes the credential exclusion structural rather than a filter that
        has to be kept correct. Proxy callers keep their own metadata under the
        allowlisted requester_metadata key.
        """
        generation_metadata = self._drive_with_canary()

        for caller_key in ("first_custom", "second_custom", "session_id", "trace_name"):
            assert caller_key not in generation_metadata

    def test_provider_specific_span_receives_the_emitted_blob(self):
        """
        The provider span reads hidden_params, which is an enrichment on the emitted
        blob rather than a key of request metadata. Handing it the steering dict
        instead would silently stop emitting vertex grounding spans.
        """
        self._drive_with_canary(hidden_params={"vertex_ai_grounding_metadata": ["ground-a", "ground-b"]})

        span_inputs = [call.kwargs.get("input") for call in self.mock_langfuse_trace.span.call_args_list]
        assert span_inputs == ["ground-a", "ground-b"]
        assert self.CANARY not in self._emitted_payload_text()

    def test_caller_cannot_spoof_an_allowlisted_identity_field(self):
        """
        Request metadata never reaches the blob, so a caller naming user_api_key_alias
        cannot have their value emitted in place of the proxy-resolved one.
        """
        generation_metadata = self._drive_with_canary(
            extra_metadata={"user_api_key_alias": "spoofed-by-caller"}
        )

        assert generation_metadata["user_api_key_alias"] == "canary-alias"

    def test_caller_nested_metadata_cannot_erase_a_litellm_enrichment(self):
        """
        log_requester_metadata drops any top-level key whose name also appears inside
        requester_metadata. Sourcing the blob from the allowlist populates that nested
        dict for real, so a caller naming a key litellm_response_cost would otherwise
        blank out the cost litellm computed. Enrichments are layered after the dedupe.
        """
        payload = self._build_standard_logging_payload(trace_id="canary-trace-id")
        payload["metadata"]["requester_metadata"] = {"litellm_response_cost": "caller-value", "api_base": "caller"}
        kwargs = {**self._build_langfuse_kwargs(payload), "response_cost": 0.25}
        metadata = self._canary_request_metadata()
        self.mock_langfuse_trace.generation.reset_mock()

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kw: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata=metadata,
                litellm_params={"metadata": metadata, "api_base": "https://real-api-base"},
                output=None,
                start_time=datetime.datetime(2024, 1, 1, 12, 0, 0),
                end_time=datetime.datetime(2024, 1, 1, 12, 0, 1),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="INFO",
                litellm_call_id="canary-call-id",
            )

        generation_metadata = self.mock_langfuse_trace.generation.call_args.kwargs["metadata"]
        assert generation_metadata["litellm_response_cost"] == 0.25
        assert generation_metadata["api_base"] == "https://real-api-base"

    def test_denied_steering_keys_and_enrichments(self):
        """
        endpoint is a plain string, so without the deny-list it would ride the
        string re-injection straight into the emitted blob. The enrichments are
        litellm-computed and must survive the move off clean_metadata.
        """
        generation_metadata = self._drive_with_canary()

        assert "endpoint" not in generation_metadata
        assert "headers" not in generation_metadata
        assert generation_metadata["litellm_response_cost"] == 0.25
        assert "hidden_params" in generation_metadata

    def test_cache_hit_is_normalized_on_the_shared_kwargs(self):
        """
        kwargs here is the shared model_call_details dict. Callbacks that run after
        langfuse read cache_hit off it and copy it into their own payloads, so
        dropping the None to False normalization records None for datadog, logfire,
        generic_api and spend tracking.
        """
        metadata = self._canary_request_metadata()
        payload = self._build_standard_logging_payload(trace_id="canary-trace-id")
        kwargs = {**self._build_langfuse_kwargs(payload), "cache_hit": None}

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kw: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata=metadata,
                litellm_params={"metadata": metadata},
                output=None,
                start_time=datetime.datetime(2024, 1, 1, 12, 0, 0),
                end_time=datetime.datetime(2024, 1, 1, 12, 0, 1),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="INFO",
                litellm_call_id="canary-call-id",
            )

        assert kwargs["cache_hit"] is False

    def test_redact_user_api_key_info_still_strips_the_emitted_blob(self):
        """
        The flag used to act on the raw-derived blob. That blob is now sourced from
        StandardLoggingPayload, which is where the user_api_key_* fields live, so the
        redaction has to run on the assembled payload or the flag silently stops working.
        """
        with patch.object(litellm, "redact_user_api_key_info", True):
            generation_metadata = self._drive_with_canary()

        assert not [key for key in generation_metadata if key.startswith("user_api_key")]

    def test_steering_keys_still_read_from_raw_metadata(self):
        """
        Only the emitted payload moves to StandardLoggingPayload. The control fields
        keep reading raw metadata, which is what Braintrust's migration got wrong.
        """
        self._drive_with_canary()

        assert self.last_trace_kwargs.get("session_id") == "canary-session"
        assert self.last_trace_kwargs.get("name") == "canary-trace"

    def test_failure_trace_survives_a_missing_standard_logging_object(self):
        """
        get_standard_logging_object_payload is fail-open and returns None on any
        exception, which is exactly the failed-request case Langfuse most needs to
        show. The trace is still emitted with the litellm_trace_id fallback, and the
        blob degrades to caller strings plus enrichments rather than falling back to
        raw metadata, which would ship the UserAPIKeyAuth object.
        """
        metadata = self._canary_request_metadata()
        kwargs = {
            "standard_logging_object": None,
            "model": "gpt-4",
            "call_type": "completion",
            "cache_hit": False,
            "messages": [],
            "litellm_trace_id": "trace-id-failure",
        }
        self.last_trace_kwargs = {}
        self.mock_langfuse_trace.generation.reset_mock()

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kwargs: generation_params,
            create=True,
        ):
            trace_id, _ = self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata=metadata,
                litellm_params={"metadata": metadata},
                output=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="ERROR",
                litellm_call_id="call-id-different",
            )

        import json

        assert trace_id == "trace-id-failure"
        assert self.last_trace_kwargs.get("id") == "trace-id-failure"
        generation_metadata = self.mock_langfuse_trace.generation.call_args.kwargs["metadata"]
        assert "user_api_key_auth" not in generation_metadata
        assert self.CANARY not in self._emitted_payload_text()
        assert "first_custom" not in generation_metadata
        # hidden_params comes off the payload, so it is omitted rather than emitted
        # as an unserializable placeholder
        assert "hidden_params" not in generation_metadata
        json.dumps(generation_metadata)

    def test_log_langfuse_v2_session_id_passed_as_trace_session_id(self):
        """
        Test that metadata.session_id is correctly passed as trace_params["session_id"]
        for Langfuse session grouping, and does NOT override trace_id.
        Each LLM call should get its own unique trace_id while sharing the session_id.
        """
        payload = self._build_standard_logging_payload(trace_id="std-trace-123")
        kwargs = self._build_langfuse_kwargs(payload)
        self.last_trace_kwargs = {}

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kwargs: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata={"session_id": "my-session-abc"},
                litellm_params={"metadata": {"session_id": "my-session-abc"}},
                output=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="INFO",
                litellm_call_id="call-id-456",
            )

        # session_id should be set for Langfuse session grouping
        assert self.last_trace_kwargs.get("session_id") == "my-session-abc"
        # trace_id should remain the standard trace_id, NOT the session_id
        assert self.last_trace_kwargs.get("id") == "std-trace-123"

    def test_log_langfuse_v2_session_id_preserved_for_error_level(self):
        """
        Test that session_id is correctly passed in trace_params even when
        the log level is ERROR (failure case). This verifies the fix for
        failed requests losing session_id mapping in Langfuse.
        """
        payload = self._build_standard_logging_payload(trace_id="std-trace-err")
        kwargs = self._build_langfuse_kwargs(payload)
        self.last_trace_kwargs = {}

        with patch(
            "litellm.integrations.langfuse.langfuse._add_prompt_to_generation_params",
            side_effect=lambda generation_params, **kwargs: generation_params,
            create=True,
        ):
            self.logger._log_langfuse_v2(
                user_id="user-1",
                metadata={"session_id": "error-session-xyz"},
                litellm_params={"metadata": {"session_id": "error-session-xyz"}},
                output="BadRequestError: model not found",
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
                kwargs=kwargs,
                optional_params={},
                input={"messages": [{"role": "user", "content": "test"}]},
                response_obj=None,
                level="ERROR",
                litellm_call_id="call-id-err-789",
            )

        # session_id must be preserved even for ERROR level logs
        assert self.last_trace_kwargs.get("session_id") == "error-session-xyz"
        # trace_id should be the standard trace_id, not the session_id
        assert self.last_trace_kwargs.get("id") == "std-trace-err"
        # status_message should be set for error traces
        assert self.last_trace_kwargs.get("status_message") is not None

    def test_log_langfuse_v2_explicit_trace_id_takes_priority_over_session_id(self):
        """
        Test that when both trace_id and session_id are provided in metadata,
        trace_id takes priority as the trace identifier.
        """
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
                metadata={
                    "session_id": "session-999",
                    "trace_id": "explicit-trace-id-777",
                },
                litellm_params={
                    "metadata": {
                        "session_id": "session-999",
                        "trace_id": "explicit-trace-id-777",
                    }
                },
                output=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
                kwargs=kwargs,
                optional_params={},
                input=None,
                response_obj=None,
                level="DEFAULT",
                litellm_call_id="call-id-aaa",
            )

        # Explicit trace_id must take priority
        assert self.last_trace_kwargs.get("id") == "explicit-trace-id-777"
        # session_id must still be set for session grouping
        assert self.last_trace_kwargs.get("session_id") == "session-999"


def test_failure_handler_langfuse_kwargs_excludes_original_response():
    """
    Test that the actual Logging.failure_handler() passes kwargs without
    'original_response' to the Langfuse logger. Exercises the real code path
    rather than simulating the filtering logic.
    """
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging

    # Create a Logging instance
    logging_obj = Logging(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type="completion",
        start_time=datetime.datetime.utcnow(),
        litellm_call_id="test-call-id-failure",
        function_id="test-function-id",
    )

    # Set up model_call_details with original_response (simulates a coroutine)
    mock_coroutine = MagicMock()
    logging_obj.model_call_details["original_response"] = mock_coroutine
    logging_obj.model_call_details["litellm_params"] = {
        "metadata": {"session_id": "test-session-failure"},
        "litellm_session_id": None,
    }
    logging_obj.model_call_details["optional_params"] = {}

    # Capture what gets passed to log_event_on_langfuse
    captured_kwargs = {}
    mock_langfuse_logger = MagicMock()

    def capture_log_event(**log_kwargs):
        captured_kwargs.update(log_kwargs)
        return {"trace_id": "mock-trace-id", "generation_id": "mock-gen-id"}

    mock_langfuse_logger.log_event_on_langfuse.side_effect = capture_log_event

    # Set "langfuse" as a failure callback so the failure_handler processes it
    original_failure_callback = litellm.failure_callback
    litellm.failure_callback = ["langfuse"]

    try:
        # Mock LangFuseHandler to return our capturing mock logger
        with patch(
            "litellm.litellm_core_utils.litellm_logging.LangFuseHandler"
        ) as mock_handler_class:
            mock_handler_class.get_langfuse_logger_for_request.return_value = (
                mock_langfuse_logger
            )

            # Call the actual failure_handler
            test_exception = Exception("TestError: model not found")
            logging_obj.failure_handler(
                exception=test_exception,
                traceback_exception="Traceback: test",
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
            )

        # Verify log_event_on_langfuse was actually called
        assert (
            mock_langfuse_logger.log_event_on_langfuse.called
        ), "log_event_on_langfuse was not called"

        # Verify original_response is NOT in the kwargs passed to Langfuse
        langfuse_kwargs = captured_kwargs.get("kwargs", {})
        assert (
            "original_response" not in langfuse_kwargs
        ), "original_response should be excluded from kwargs passed to Langfuse"

        # Verify session_id metadata is preserved in the kwargs
        langfuse_metadata = langfuse_kwargs.get("litellm_params", {}).get(
            "metadata", {}
        )
        assert (
            langfuse_metadata.get("session_id") == "test-session-failure"
        ), "session_id should be preserved in kwargs passed to Langfuse"

        # Verify level is ERROR
        assert captured_kwargs.get("level") == "ERROR"
    finally:
        litellm.failure_callback = original_failure_callback


@pytest.mark.asyncio
async def test_async_log_failure_event_logs_to_langfuse():
    """
    Test that LangfusePromptManagement.async_log_failure_event() calls
    log_event_on_langfuse with level=ERROR even when standard_logging_object
    is present. This is the code path the proxy uses for failed LLM calls.
    """
    from litellm.integrations.langfuse.langfuse_prompt_management import (
        LangfusePromptManagement,
    )

    mock_langfuse_module = MagicMock()
    mock_langfuse_module.version.__version__ = "3.0.0"

    with (
        patch.dict(
            "os.environ",
            {
                "LANGFUSE_SECRET_KEY": "test-secret",
                "LANGFUSE_PUBLIC_KEY": "test-public",
                "LANGFUSE_HOST": "https://test.langfuse.com",
            },
        ),
        patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
    ):
        prompt_mgmt = LangfusePromptManagement()

        # Mock the langfuse logger returned by get_langfuse_logger_for_request
        mock_logger = MagicMock()
        mock_logger.log_event_on_langfuse.return_value = {
            "trace_id": "mock-trace",
            "generation_id": "mock-gen",
        }

        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseHandler"
        ) as mock_handler:
            mock_handler.get_langfuse_logger_for_request.return_value = mock_logger

            kwargs = {
                "litellm_params": {
                    "metadata": {"session_id": "test-session-fail"},
                },
                "litellm_call_id": "call-fail-123",
                "user": "test-user",
                "exception": Exception("API error: model not found"),
                "standard_logging_object": {
                    "error_str": "API error: model not found",
                    "trace_id": "std-trace-fail",
                    "metadata": {},
                },
            }

            await prompt_mgmt.async_log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
            )

            # Verify log_event_on_langfuse was called
            assert (
                mock_logger.log_event_on_langfuse.called
            ), "log_event_on_langfuse was not called for failure event"
            call_kwargs = mock_logger.log_event_on_langfuse.call_args[1]
            assert call_kwargs["level"] == "ERROR"
            assert call_kwargs["status_message"] == "API error: model not found"
            assert call_kwargs["response_obj"] is None


@pytest.mark.asyncio
async def test_async_log_failure_event_works_without_standard_logging_object():
    """
    Test that async_log_failure_event() still logs to Langfuse even when
    standard_logging_object is None (e.g. when get_standard_logging_object_payload
    threw an exception). This is the critical fix — before, it silently returned.
    """
    from litellm.integrations.langfuse.langfuse_prompt_management import (
        LangfusePromptManagement,
    )

    mock_langfuse_module = MagicMock()
    mock_langfuse_module.version.__version__ = "3.0.0"

    with (
        patch.dict(
            "os.environ",
            {
                "LANGFUSE_SECRET_KEY": "test-secret",
                "LANGFUSE_PUBLIC_KEY": "test-public",
                "LANGFUSE_HOST": "https://test.langfuse.com",
            },
        ),
        patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
    ):
        prompt_mgmt = LangfusePromptManagement()

        mock_logger = MagicMock()
        mock_logger.log_event_on_langfuse.return_value = {
            "trace_id": "mock-trace",
            "generation_id": "mock-gen",
        }

        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseHandler"
        ) as mock_handler:
            mock_handler.get_langfuse_logger_for_request.return_value = mock_logger

            kwargs = {
                "litellm_params": {
                    "metadata": {"session_id": "test-session-no-slo"},
                },
                "litellm_call_id": "call-no-slo-456",
                "user": "test-user",
                "exception": Exception("InternalServerError: something broke"),
                "standard_logging_object": None,  # This is the key — it's None
            }

            await prompt_mgmt.async_log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=datetime.datetime.utcnow(),
                end_time=datetime.datetime.utcnow(),
            )

            # CRITICAL: log_event_on_langfuse MUST still be called
            assert mock_logger.log_event_on_langfuse.called, (
                "log_event_on_langfuse was NOT called when standard_logging_object "
                "is None — failure trace would be silently dropped"
            )
            call_kwargs = mock_logger.log_event_on_langfuse.call_args[1]
            assert call_kwargs["level"] == "ERROR"
            # Falls back to exception from kwargs
            assert "InternalServerError" in call_kwargs["status_message"]


def test_max_langfuse_clients_limit():
    """
    Test that the max langfuse clients limit is respected when initializing multiple clients
    """
    # Mock langfuse package to avoid triggering real import.
    # The real langfuse import fails on Python 3.14 due to pydantic v1 incompatibility,
    # and sys.modules["langfuse"] may be absent after other tests in the suite clean up.
    mock_langfuse = MagicMock()
    mock_langfuse.version.__version__ = "3.0.0"
    # Set max clients to 2 for testing
    original_initialized_langfuse_clients = litellm.initialized_langfuse_clients
    with (
        patch.dict("sys.modules", {"langfuse": mock_langfuse}),
        patch.object(langfuse_module, "MAX_LANGFUSE_INITIALIZED_CLIENTS", 2),
    ):
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
        with pytest.raises(Exception, match='Max langfuse clients reached') as exc_info:
            logger3 = LangFuseLogger(
                langfuse_public_key="test_key_3",
                langfuse_secret="test_secret_3",
                langfuse_host="https://test3.langfuse.com",
            )

        # Verify the error message contains the expected text
        assert "Max langfuse clients reached" in str(exc_info.value)

        # Counter should still be 2 (third client failed to initialize)
        assert litellm.initialized_langfuse_clients == 2

    litellm.initialized_langfuse_clients = original_initialized_langfuse_clients


class _RecordingLangfuse:
    last_parameters: Optional[dict] = None

    def __init__(self, environment=None, **parameters):
        type(self).last_parameters = {"environment": environment, **parameters}
        self.client = MagicMock()


class _RecordingLangfuseWithoutEnvironment:
    last_parameters: Optional[dict] = None

    def __init__(self, **parameters):
        type(self).last_parameters = parameters
        self.client = MagicMock()


def _build_langfuse_logger(monkeypatch) -> LangFuseLogger:
    monkeypatch.setenv("LANGFUSE_MOCK", "false")
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    with patch("langfuse.Langfuse", _RecordingLangfuse):
        return LangFuseLogger(
            langfuse_public_key="pk-lit5228",
            langfuse_secret="sk-lit5228",
            langfuse_host="https://test.langfuse.com",
        )


def test_langfuse_environment_is_passed_to_sdk_client(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MOCK", "false")
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    with patch("langfuse.Langfuse", _RecordingLangfuse):
        logger = LangFuseLogger(
            langfuse_public_key="pk-env",
            langfuse_secret="sk-env",
            langfuse_host="https://test.langfuse.com",
            langfuse_environment="staging",
        )
    assert logger.langfuse_environment == "staging"
    assert _RecordingLangfuse.last_parameters["environment"] == "staging"


def test_langfuse_environment_falls_back_to_deployment_env_var(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MOCK", "false")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "deployment-wide")
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    with patch("langfuse.Langfuse", _RecordingLangfuse):
        logger = LangFuseLogger(
            langfuse_public_key="pk-env",
            langfuse_secret="sk-env",
            langfuse_host="https://test.langfuse.com",
        )
    assert logger.langfuse_environment == "deployment-wide"
    assert _RecordingLangfuse.last_parameters["environment"] == "deployment-wide"


def test_langfuse_environment_omitted_for_old_sdk_versions(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MOCK", "false")
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    with patch("langfuse.Langfuse", _RecordingLangfuseWithoutEnvironment):
        LangFuseLogger(
            langfuse_public_key="pk-env",
            langfuse_secret="sk-env",
            langfuse_host="https://test.langfuse.com",
            langfuse_environment="staging",
        )
    assert "environment" not in _RecordingLangfuseWithoutEnvironment.last_parameters


def test_dynamic_langfuse_environment_triggers_dynamic_logger():
    from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
    from litellm.types.utils import StandardCallbackDynamicParams

    params = StandardCallbackDynamicParams(langfuse_environment="team-a-env")

    assert LangFuseHandler._dynamic_langfuse_credentials_are_passed(params) is True

    config = LangFuseHandler.get_dynamic_langfuse_logging_config(
        standard_callback_dynamic_params=params
    )
    assert config["langfuse_environment"] == "team-a-env"


def test_langfuse_sdk_client_survives_httpx_cache_eviction(monkeypatch):
    import gc
    import weakref

    from litellm.caching.llm_caching_handler import LLMClientCache

    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    logger = _build_langfuse_logger(monkeypatch)
    sdk_client = _RecordingLangfuse.last_parameters["httpx_client"]

    cached_handler = _get_httpx_client()
    handler_ref = weakref.ref(cached_handler)

    assert sdk_client is logger.langfuse_client
    assert sdk_client is cached_handler.client

    litellm.in_memory_llm_clients_cache = LLMClientCache()
    del cached_handler
    gc.collect()

    assert litellm.in_memory_llm_clients_cache.get_cache("httpx_client") is None
    assert handler_ref() is not None, "logger must keep the handler that owns the client it handed the SDK"
    assert not sdk_client.is_closed


def test_langfuse_logger_reuses_the_shared_cached_client(monkeypatch):
    import gc

    from litellm.caching.llm_caching_handler import LLMClientCache

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    first = _build_langfuse_logger(monkeypatch)
    second = _build_langfuse_logger(monkeypatch)

    assert first.langfuse_client is second.langfuse_client

    del second
    gc.collect()

    assert not first.langfuse_client.is_closed


_LANGFUSE_REDACTED = "redacted-by-litellm"


def _steering_logger() -> LangFuseLogger:
    """``__new__`` skips the SDK and network setup in ``__init__``."""
    logger = LangFuseLogger.__new__(LangFuseLogger)
    logger.Langfuse = MagicMock()
    logger.langfuse_sdk_version = "2.60.0"
    return logger


def _emit(logger: LangFuseLogger, *, metadata=None, headers=None):
    """``log_event_on_langfuse`` is the entry point that folds ``langfuse_*`` headers into metadata."""
    now = datetime.datetime.now()
    response_obj = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "the-output"}}]
    )
    logger.log_event_on_langfuse(
        kwargs={
            "call_type": "completion",
            "litellm_params": {
                "metadata": dict(metadata or {}),
                "proxy_server_request": {"headers": dict(headers or {})},
            },
            "messages": [{"role": "user", "content": "the-input"}],
            "optional_params": {},
        },
        response_obj=response_obj,
        start_time=now,
        end_time=now,
    )
    return (
        logger.Langfuse.trace.call_args.kwargs,
        logger.Langfuse.trace.return_value.generation.call_args.kwargs,
    )


def test_mask_input_header_false_keeps_the_prompt():
    logger = _steering_logger()

    trace_params, generation_params = _emit(logger, headers={"langfuse_mask_input": "false"})

    assert trace_params["input"] == {"messages": [{"role": "user", "content": "the-input"}]}
    assert generation_params["input"] == {"messages": [{"role": "user", "content": "the-input"}]}


def test_mask_input_header_true_redacts_the_prompt():
    logger = _steering_logger()

    trace_params, generation_params = _emit(logger, headers={"langfuse_mask_input": "true"})

    assert trace_params["input"] == _LANGFUSE_REDACTED
    assert generation_params["input"] == _LANGFUSE_REDACTED


def test_mask_output_header_false_keeps_the_completion():
    logger = _steering_logger()

    trace_params, generation_params = _emit(logger, headers={"langfuse_mask_output": "false"})

    assert trace_params["output"] != _LANGFUSE_REDACTED
    assert generation_params["output"] != _LANGFUSE_REDACTED


def test_mask_output_header_true_redacts_the_completion():
    logger = _steering_logger()

    trace_params, generation_params = _emit(logger, headers={"langfuse_mask_output": "true"})

    assert trace_params["output"] == _LANGFUSE_REDACTED
    assert generation_params["output"] == _LANGFUSE_REDACTED


@pytest.mark.parametrize(
    "mask_input, expect_redacted",
    [
        (False, False),
        (True, True),
        # An unrecognised string keeps its truthiness, so existing behaviour is unchanged
        ("yes", True),
    ],
)
def test_mask_input_from_the_request_body_is_unchanged(mask_input, expect_redacted):
    logger = _steering_logger()

    trace_params, _ = _emit(logger, metadata={"mask_input": mask_input})

    assert (trace_params["input"] == _LANGFUSE_REDACTED) is expect_redacted


@pytest.mark.parametrize("flag", [True, "true"])
def test_update_trace_keys_header_applies_every_key_when_enabled(flag):
    logger = _steering_logger()

    with patch.object(litellm, "langfuse_enable_update_trace_keys", flag):
        trace_params, _ = _emit(
            logger,
            headers={
                "langfuse_existing_trace_id": "trace-1",
                "langfuse_update_trace_keys": "trace_release, trace_tail",
                "langfuse_trace_release": "v1.2.3",
                "langfuse_trace_tail": "last",
            },
        )

    assert trace_params["release"] == "v1.2.3"
    assert trace_params["tail"] == "last"


def test_update_trace_keys_is_off_by_default():
    """
    The caller picks the key name, so while the feature is on they can name
    user_api_key_auth and have the resolved auth object, including team callback
    credentials, serialized onto the trace. It stays inert until an operator opts in.
    """
    logger = _steering_logger()

    trace_params, _ = _emit(
        logger,
        metadata={
            "existing_trace_id": "trace-1",
            "update_trace_keys": ["user_api_key_auth", "trace_release"],
            "user_api_key_auth": {"team_metadata": {"logging": [{"callback_vars": {"secret": "sk-canary"}}]}},
            "trace_release": "v1.2.3",
        },
    )

    assert "user_api_key_auth" not in trace_params
    assert "release" not in trace_params
    assert "sk-canary" not in json.dumps(trace_params, default=repr)


def test_update_trace_keys_input_and_output_are_gated_too():
    logger = _steering_logger()

    off, _ = _emit(logger, metadata={"existing_trace_id": "trace-1", "update_trace_keys": ["input", "output"]})
    with patch.object(litellm, "langfuse_enable_update_trace_keys", True):
        on, _ = _emit(logger, metadata={"existing_trace_id": "trace-1", "update_trace_keys": ["input", "output"]})

    assert "input" not in off and "output" not in off
    assert "input" in on and "output" in on


def test_update_trace_keys_from_the_request_body_list_applies_when_enabled():
    logger = _steering_logger()

    with patch.object(litellm, "langfuse_enable_update_trace_keys", True):
        trace_params, _ = _emit(
            logger,
            metadata={
                "existing_trace_id": "trace-1",
                "update_trace_keys": ["trace_release"],
                "trace_release": "v1.2.3",
            },
        )

    assert trace_params["release"] == "v1.2.3"


def test_update_trace_keys_matches_whole_keys_not_substrings():
    logger = _steering_logger()

    trace_params, _ = _emit(
        logger,
        headers={"langfuse_existing_trace_id": "trace-1", "langfuse_update_trace_keys": "my_input"},
    )

    assert "input" not in trace_params


def test_langfuse_environment_is_coerced_and_validated(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MOCK", "false")
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    with patch("langfuse.Langfuse", _RecordingLangfuse):
        logger = LangFuseLogger(
            langfuse_public_key="pk-env",
            langfuse_secret="sk-env",
            langfuse_host="https://test.langfuse.com",
            langfuse_environment=123,  # non-string: must coerce, not crash
        )
    assert logger.langfuse_environment == "123"

    with pytest.raises(ValueError, match="langfuse_environment"):
        LangFuseLogger(
            langfuse_public_key="pk-env",
            langfuse_secret="sk-env",
            langfuse_host="https://test.langfuse.com",
            langfuse_environment="Production",
        )


def test_langfuse_empty_environment_falls_back_and_is_not_dynamic(monkeypatch):
    from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
    from litellm.types.utils import StandardCallbackDynamicParams

    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "production")

    # '' falls back to the deployment env var at init
    monkeypatch.setenv("LANGFUSE_MOCK", "false")
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    with patch("langfuse.Langfuse", _RecordingLangfuse):
        logger = LangFuseLogger(
            langfuse_public_key="pk-env",
            langfuse_secret="sk-env",
            langfuse_host="https://test.langfuse.com",
            langfuse_environment="",
        )
    assert logger.langfuse_environment == "production"

    # env-only params that add nothing do not select a dynamic logger
    for redundant in ["", "  ", "production"]:
        params = StandardCallbackDynamicParams(langfuse_environment=redundant)
        assert LangFuseHandler._dynamic_langfuse_credentials_are_passed(params) is False

    params = StandardCallbackDynamicParams(langfuse_environment="team-a-prod")
    assert LangFuseHandler._dynamic_langfuse_credentials_are_passed(params) is True
