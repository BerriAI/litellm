"""
Tests for standard_logging_payload_excluded_fields feature.

This feature allows users to exclude specific fields from StandardLoggingPayload
before any callback receives it. This is useful for:
- Reducing log sizes (excluding large fields like 'response' or 'messages')
- Privacy compliance (excluding sensitive fields)
- Cost management (less data stored/transmitted)

Example config:
    litellm_settings:
      success_callback: ["s3"]
      standard_logging_payload_excluded_fields: ["response", "messages"]
"""

import asyncio
import importlib
import os
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Dict, Final, Optional

import pytest
import pytest_asyncio

import litellm
from litellm.constants import LOGGING_WORKER_MAX_TIME_PER_COROUTINE
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def create_sample_standard_logging_payload() -> Dict:
    """Create a sample StandardLoggingPayload for testing."""
    return {
        "id": "test-id-123",
        "trace_id": "trace-123",
        "call_type": "completion",
        "stream": False,
        "response_cost": 0.001,
        "cost_breakdown": None,
        "response_cost_failure_debug_info": None,
        "status": "success",
        "status_fields": {},
        "custom_llm_provider": "openai",
        "total_tokens": 100,
        "prompt_tokens": 50,
        "completion_tokens": 50,
        "startTime": 1234567890.0,
        "endTime": 1234567891.0,
        "completionStartTime": 1234567890.5,
        "response_time": 1.0,
        "model_map_information": {},
        "model": "gpt-5.5",
        "model_id": "model-123",
        "model_group": None,
        "api_base": "https://api.openai.com/v1",
        "metadata": {},
        "cache_hit": False,
        "cache_key": None,
        "saved_cache_cost": 0.0,
        "request_tags": [],
        "end_user": None,
        "requester_ip_address": None,
        "user_agent": None,
        "messages": [{"role": "user", "content": "Hello, this is sensitive data!"}],
        "response": {"choices": [{"message": {"content": "This is a sensitive response!"}}]},
        "error_str": None,
        "error_information": None,
        "model_parameters": {},
        "hidden_params": {},
        "guardrail_information": None,
        "standard_built_in_tools_params": None,
    }


def create_model_call_details(
    standard_logging_payload: Optional[Dict] = None,
) -> Dict:
    """Create model_call_details dict with standard_logging_object."""
    if standard_logging_payload is None:
        standard_logging_payload = create_sample_standard_logging_payload()
    return {
        "standard_logging_object": standard_logging_payload,
        "other_key": "other_value",
    }


def test_get_callback_env_vars():
    env_vars = CustomLogger.get_callback_env_vars("langfuse")
    assert env_vars == [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]

    alias_env_vars = CustomLogger.get_callback_env_vars("langfuse_otel")
    assert alias_env_vars == env_vars

    missing_env_vars = CustomLogger.get_callback_env_vars("does_not_exist")
    assert missing_env_vars == []

    none_env_vars = CustomLogger.get_callback_env_vars(None)
    assert none_env_vars == []


def test_responses_redaction_covers_instructions_reasoning_and_tool_arguments() -> None:
    response: Final = {
        "instructions": "INSTRUCTION_CANARY",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "ANSWER_CANARY"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "REASONING_CANARY"}]},
            {"type": "function_call", "name": "example_tool", "arguments": '{"value":"ARGUMENT_CANARY"}'},
        ],
    }
    details: Final = {"standard_logging_object": {"response": response}}
    logger: Final = CustomLogger(turn_off_message_logging=True)

    stored: Final = logger.redact_standard_logging_payload_from_model_call_details(details)["standard_logging_object"]

    assert stored["response"] == {
        "instructions": "redacted-by-litellm",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "redacted-by-litellm"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "redacted-by-litellm"}]},
            {"type": "function_call", "name": "example_tool", "arguments": "redacted-by-litellm"},
        ],
    }
    assert details["standard_logging_object"]["response"] is response
    assert response["instructions"] == "INSTRUCTION_CANARY"
    assert response["output"][1]["summary"][0]["text"] == "REASONING_CANARY"
    assert response["output"][2]["arguments"] == '{"value":"ARGUMENT_CANARY"}'


class TestStandardLoggingPayloadExcludedFields:
    """Test suite for standard_logging_payload_excluded_fields feature."""

    def setup_method(self):
        """Reset litellm settings before each test."""
        litellm.standard_logging_payload_excluded_fields = None

    def teardown_method(self):
        """Clean up after each test."""
        litellm.standard_logging_payload_excluded_fields = None

    def test_no_excluded_fields_no_change(self):
        """Test that payload is unchanged when no fields are excluded."""
        logger = CustomLogger()
        model_call_details = create_model_call_details()
        original_keys = set(model_call_details["standard_logging_object"].keys())

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        result_keys = set(result["standard_logging_object"].keys())
        assert result_keys == original_keys

    def test_exclude_single_field(self):
        """Test excluding a single field (response)."""
        litellm.standard_logging_payload_excluded_fields = ["response"]

        logger = CustomLogger()
        model_call_details = create_model_call_details()

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        assert "response" not in result["standard_logging_object"]
        assert "messages" in result["standard_logging_object"]
        assert "model" in result["standard_logging_object"]

    def test_exclude_multiple_fields(self):
        """Test excluding multiple fields (response, messages)."""
        litellm.standard_logging_payload_excluded_fields = ["response", "messages"]

        logger = CustomLogger()
        model_call_details = create_model_call_details()

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        assert "response" not in result["standard_logging_object"]
        assert "messages" not in result["standard_logging_object"]
        assert "model" in result["standard_logging_object"]
        assert "model_parameters" in result["standard_logging_object"]

    def test_exclude_metadata_field(self):
        """Test excluding the metadata field."""
        litellm.standard_logging_payload_excluded_fields = ["metadata"]

        logger = CustomLogger()
        payload = create_sample_standard_logging_payload()
        payload["metadata"] = {"sensitive_key": "sensitive_value"}
        model_call_details = create_model_call_details(payload)

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        assert "metadata" not in result["standard_logging_object"]

    def test_exclude_hidden_params(self):
        """Test excluding hidden_params field."""
        litellm.standard_logging_payload_excluded_fields = ["hidden_params"]

        logger = CustomLogger()
        payload = create_sample_standard_logging_payload()
        payload["hidden_params"] = {"api_key": "sk-secret-key"}
        model_call_details = create_model_call_details(payload)

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        assert "hidden_params" not in result["standard_logging_object"]

    def test_exclude_nonexistent_field_no_error(self):
        """Test that excluding a non-existent field doesn't cause an error."""
        litellm.standard_logging_payload_excluded_fields = [
            "nonexistent_field",
            "response",
        ]

        logger = CustomLogger()
        model_call_details = create_model_call_details()

        # Should not raise an exception
        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        assert "response" not in result["standard_logging_object"]
        assert "messages" in result["standard_logging_object"]

    def test_original_payload_not_modified(self):
        """Test that the original model_call_details is not modified."""
        litellm.standard_logging_payload_excluded_fields = ["response", "messages"]

        logger = CustomLogger()
        model_call_details = create_model_call_details()
        original_payload = deepcopy(model_call_details)

        logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        # Original should still have the fields
        assert "response" in model_call_details["standard_logging_object"]
        assert "messages" in model_call_details["standard_logging_object"]
        assert model_call_details == original_payload

    def test_combined_with_turn_off_message_logging(self):
        """Test that excluded_fields works together with turn_off_message_logging."""
        litellm.standard_logging_payload_excluded_fields = ["metadata", "hidden_params"]

        logger = CustomLogger(turn_off_message_logging=True)
        model_call_details = create_model_call_details()

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        # excluded_fields should remove these
        assert "metadata" not in result["standard_logging_object"]
        assert "hidden_params" not in result["standard_logging_object"]

        # turn_off_message_logging should redact these
        redacted_str = "redacted-by-litellm"
        assert result["standard_logging_object"]["messages"][0]["content"] == redacted_str
        assert result["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == redacted_str

    def test_excluded_fields_takes_precedence_over_redaction(self):
        """Test that if a field is both excluded and would be redacted, it's excluded."""
        litellm.standard_logging_payload_excluded_fields = ["response"]

        logger = CustomLogger(turn_off_message_logging=True)
        model_call_details = create_model_call_details()

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        # response should be excluded (not redacted)
        assert "response" not in result["standard_logging_object"]

        # messages should still be redacted
        redacted_str = "redacted-by-litellm"
        assert result["standard_logging_object"]["messages"][0]["content"] == redacted_str

    def test_exclude_all_sensitive_fields(self):
        """Test excluding all potentially sensitive fields."""
        litellm.standard_logging_payload_excluded_fields = [
            "messages",
            "response",
            "metadata",
            "hidden_params",
            "model_parameters",
            "error_str",
            "error_information",
        ]

        logger = CustomLogger()
        model_call_details = create_model_call_details()

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        standard_obj = result["standard_logging_object"]

        # All sensitive fields should be removed
        assert "messages" not in standard_obj
        assert "response" not in standard_obj
        assert "metadata" not in standard_obj
        assert "hidden_params" not in standard_obj
        assert "model_parameters" not in standard_obj
        assert "error_str" not in standard_obj
        assert "error_information" not in standard_obj

        # Non-sensitive fields should remain
        assert "id" in standard_obj
        assert "model" in standard_obj
        assert "response_cost" in standard_obj
        assert "total_tokens" in standard_obj

    def test_empty_excluded_fields_list(self):
        """Test that an empty list doesn't affect the payload."""
        litellm.standard_logging_payload_excluded_fields = []

        logger = CustomLogger()
        model_call_details = create_model_call_details()
        original_keys = set(model_call_details["standard_logging_object"].keys())

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        result_keys = set(result["standard_logging_object"].keys())
        assert result_keys == original_keys

    def test_none_standard_logging_object(self):
        """Test handling when standard_logging_object is None."""
        litellm.standard_logging_payload_excluded_fields = ["response"]

        logger = CustomLogger()
        model_call_details = {"other_key": "other_value"}

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        # Should return unchanged when no standard_logging_object
        assert result == model_call_details


class TestExcludedFieldsIntegration:
    """Integration tests for excluded fields with actual callbacks."""

    def setup_method(self):
        """Reset litellm settings before each test."""
        litellm.standard_logging_payload_excluded_fields = None
        litellm.callbacks = []

    def teardown_method(self):
        """Clean up after each test."""
        litellm.standard_logging_payload_excluded_fields = None
        litellm.callbacks = []

    def test_custom_callback_receives_filtered_payload(self):
        """Test that a custom callback receives the filtered payload."""
        captured_payloads = []

        class TestCallback(CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                captured_payloads.append(kwargs.get("standard_logging_object", {}))

        litellm.standard_logging_payload_excluded_fields = ["response", "messages"]

        callback = TestCallback()
        model_call_details = create_model_call_details()

        # Simulate what litellm_logging.py does
        filtered_details = callback.redact_standard_logging_payload_from_model_call_details(model_call_details)

        callback.log_success_event(
            kwargs=filtered_details,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert len(captured_payloads) == 1
        assert "response" not in captured_payloads[0]
        assert "messages" not in captured_payloads[0]
        assert "model" in captured_payloads[0]


class TestExcludedFieldsConfigLoading:
    """Test that the config is properly loaded from litellm_settings."""

    def setup_method(self):
        """Reset litellm settings before each test."""
        litellm.standard_logging_payload_excluded_fields = None

    def teardown_method(self):
        """Clean up after each test."""
        litellm.standard_logging_payload_excluded_fields = None

    def test_config_attribute_exists(self):
        """Test that the config attribute exists on litellm module."""
        assert hasattr(litellm, "standard_logging_payload_excluded_fields")

    def test_config_default_is_none(self):
        """Test that the default value is None."""
        # Reset to ensure we're testing the default
        litellm.standard_logging_payload_excluded_fields = None
        assert litellm.standard_logging_payload_excluded_fields is None

    def test_config_can_be_set_to_list(self):
        """Test that the config can be set to a list."""
        litellm.standard_logging_payload_excluded_fields = ["response", "messages"]
        assert litellm.standard_logging_payload_excluded_fields == [
            "response",
            "messages",
        ]

    def test_config_setattr_simulates_proxy_loading(self):
        """Test that setattr works as the proxy would use it."""
        # Simulating how proxy_server.py sets litellm_settings
        config_value = ["response", "messages", "metadata"]
        setattr(litellm, "standard_logging_payload_excluded_fields", config_value)

        assert litellm.standard_logging_payload_excluded_fields == config_value

        # Test it actually works in the logger
        logger = CustomLogger()
        model_call_details = create_model_call_details()

        result = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)

        assert "response" not in result["standard_logging_object"]
        assert "messages" not in result["standard_logging_object"]
        assert "metadata" not in result["standard_logging_object"]


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest_asyncio.fixture(loop_scope="function", autouse=True)
async def drain_logging_worker(isolate_litellm_state: None) -> AsyncIterator[None]:
    yield
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS)


LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS: Final = LOGGING_WORKER_MAX_TIME_PER_COROUTINE + 5.0


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging
    from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])


_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "s3_audit_callback_params",
    "datadog_params",
    "vector_store_registry",
)

_DEFAULTS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield
