import asyncio
import datetime
import json
import os
import sys
from datetime import timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

import litellm
from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _get_vector_store_request_for_spend_logs_payload,
    _sanitize_request_body_for_spend_logs_payload,
    get_logging_payload,
)


def test_sanitize_request_body_for_spend_logs_payload_basic():
    request_body = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    assert _sanitize_request_body_for_spend_logs_payload(request_body) == request_body


def test_sanitize_request_body_for_spend_logs_payload_long_string():
    from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB

    # Create a string longer than MAX_STRING_LENGTH_PROMPT_IN_DB (2048)
    long_string = "a" * 3000  # Create a string longer than MAX_STRING_LENGTH_PROMPT_IN_DB
    request_body = {"text": long_string, "normal_text": "short text"}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    
    # Calculate expected lengths: 35% start + 65% end + truncation message
    start_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.35)
    end_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.65)
    total_keep = start_chars + end_chars
    if total_keep > MAX_STRING_LENGTH_PROMPT_IN_DB:
        end_chars = MAX_STRING_LENGTH_PROMPT_IN_DB - start_chars
    
    skipped_chars = len(long_string) - (start_chars + end_chars)
    expected_truncation_message = f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {skipped_chars} chars) ..."
    expected_length = start_chars + len(expected_truncation_message) + end_chars
    
    assert len(sanitized["text"]) == expected_length
    assert sanitized["text"].startswith("a" * start_chars)
    assert sanitized["text"].endswith("a" * end_chars)
    assert expected_truncation_message in sanitized["text"]
    assert sanitized["normal_text"] == "short text"


def test_sanitize_request_body_for_spend_logs_payload_nested_dict():
    from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB

    # Create a string longer than MAX_STRING_LENGTH_PROMPT_IN_DB
    long_string = "a" * (MAX_STRING_LENGTH_PROMPT_IN_DB + 500)
    request_body = {"outer": {"inner": {"text": long_string, "normal": "short"}}}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    
    # Calculate expected lengths based on actual MAX_STRING_LENGTH_PROMPT_IN_DB
    start_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.35)
    end_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.65)
    total_keep = start_chars + end_chars
    if total_keep > MAX_STRING_LENGTH_PROMPT_IN_DB:
        end_chars = MAX_STRING_LENGTH_PROMPT_IN_DB - start_chars
    
    skipped_chars = len(long_string) - total_keep
    expected_truncation_message = f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {skipped_chars} chars) ..."
    expected_length = start_chars + len(expected_truncation_message) + end_chars
    
    assert len(sanitized["outer"]["inner"]["text"]) == expected_length
    assert sanitized["outer"]["inner"]["normal"] == "short"


def test_sanitize_request_body_for_spend_logs_payload_nested_list():
    from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB

    # Create a string longer than MAX_STRING_LENGTH_PROMPT_IN_DB
    long_string = "a" * (MAX_STRING_LENGTH_PROMPT_IN_DB + 500)
    request_body = {
        "items": [{"text": long_string}, {"text": "short"}, [{"text": long_string}]]
    }
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    
    # Calculate expected lengths based on actual MAX_STRING_LENGTH_PROMPT_IN_DB
    start_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.35)
    end_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.65)
    total_keep = start_chars + end_chars
    if total_keep > MAX_STRING_LENGTH_PROMPT_IN_DB:
        end_chars = MAX_STRING_LENGTH_PROMPT_IN_DB - start_chars
    
    skipped_chars = len(long_string) - total_keep
    expected_truncation_message = f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {skipped_chars} chars) ..."
    expected_length = start_chars + len(expected_truncation_message) + end_chars
    
    assert len(sanitized["items"][0]["text"]) == expected_length
    assert sanitized["items"][1]["text"] == "short"
    assert len(sanitized["items"][2][0]["text"]) == expected_length


def test_sanitize_request_body_for_spend_logs_payload_non_string_values():
    request_body = {"number": 42, "boolean": True, "none": None, "float": 3.14}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert sanitized == request_body


def test_sanitize_request_body_for_spend_logs_payload_empty():
    request_body: dict[str, Any] = {}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert sanitized == request_body


def test_sanitize_request_body_for_spend_logs_payload_mixed_types():
    from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB

    # Create a string longer than MAX_STRING_LENGTH_PROMPT_IN_DB
    long_string = "a" * (MAX_STRING_LENGTH_PROMPT_IN_DB + 500)
    request_body = {
        "text": long_string,
        "number": 42,
        "nested": {"list": ["short", long_string], "dict": {"key": long_string}},
    }
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    
    # Calculate expected lengths based on actual MAX_STRING_LENGTH_PROMPT_IN_DB
    start_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.35)
    end_chars = int(MAX_STRING_LENGTH_PROMPT_IN_DB * 0.65)
    total_keep = start_chars + end_chars
    if total_keep > MAX_STRING_LENGTH_PROMPT_IN_DB:
        end_chars = MAX_STRING_LENGTH_PROMPT_IN_DB - start_chars
    
    skipped_chars = len(long_string) - total_keep
    expected_truncation_message = f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {skipped_chars} chars) ..."
    expected_length = start_chars + len(expected_truncation_message) + end_chars
    
    assert len(sanitized["text"]) == expected_length
    assert sanitized["number"] == 42
    assert sanitized["nested"]["list"][0] == "short"
    assert len(sanitized["nested"]["list"][1]) == expected_length
    assert len(sanitized["nested"]["dict"]["key"]) == expected_length


def test_sanitize_request_body_for_spend_logs_payload_circular_reference():
    # Create a circular reference
    a: dict[str, Any] = {}
    b: dict[str, Any] = {"a": a}
    a["b"] = b

    # Test that it handles circular reference without infinite recursion
    sanitized = _sanitize_request_body_for_spend_logs_payload(a)
    assert sanitized == {
        "b": {"a": {}}
    }  # Should return empty dict for circular reference


@patch(
    "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs"
)
def test_get_vector_store_request_for_spend_logs_payload_store_prompts_true(
    mock_should_store,
):
    # When _should_store_prompts_and_responses_in_spend_logs returns True
    mock_should_store.return_value = True

    # Sample vector store request metadata
    vector_store_request = [
        {
            "vector_store_search_response": {
                "data": [
                    {"content": [{"text": "sensitive information", "type": "text"}]}
                ]
            }
        }
    ]

    # When store_prompts is True, the original data should be returned unchanged
    result = _get_vector_store_request_for_spend_logs_payload(vector_store_request)
    assert result == vector_store_request
    assert (
        result[0]["vector_store_search_response"]["data"][0]["content"][0]["text"]
        == "sensitive information"
    )


@patch(
    "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs"
)
def test_get_vector_store_request_for_spend_logs_payload_store_prompts_false(
    mock_should_store,
):
    # When _should_store_prompts_and_responses_in_spend_logs returns False
    mock_should_store.return_value = False

    # Sample vector store request metadata
    vector_store_request = [
        {
            "vector_store_search_response": {
                "data": [
                    {"content": [{"text": "sensitive information", "type": "text"}]}
                ]
            }
        }
    ]

    # When store_prompts is False, text should be redacted
    result = _get_vector_store_request_for_spend_logs_payload(vector_store_request)
    assert result is not None
    assert (
        result[0]["vector_store_search_response"]["data"][0]["content"][0]["text"]
        == REDACTED_BY_LITELM_STRING
    )
    # Ensure other fields are unchanged
    assert (
        result[0]["vector_store_search_response"]["data"][0]["content"][0]["type"]
        == "text"
    )


@patch(
    "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs"
)
def test_get_vector_store_request_for_spend_logs_payload_null_input(mock_should_store):
    # When input is None
    mock_should_store.return_value = False
    result = _get_vector_store_request_for_spend_logs_payload(None)
    assert result is None


def test_safe_dumps_handles_circular_references():
    """Test that safe_dumps can handle circular references without raising exceptions"""
    
    # Create a circular reference
    obj1 = {"name": "obj1"}
    obj2 = {"name": "obj2", "ref": obj1}
    obj1["ref"] = obj2  # This creates a circular reference
    
    # This should not raise an exception
    result = safe_dumps(obj1)
    
    # Should be a valid JSON string
    assert isinstance(result, str)
    
    # Should contain placeholder for circular reference
    assert "CircularReference Detected" in result
    
    # Should be parseable as JSON
    parsed = json.loads(result)
    assert parsed["name"] == "obj1"
    assert parsed["ref"]["name"] == "obj2"


def test_safe_dumps_normal_objects():
    """Test that safe_dumps works correctly with normal objects"""
    
    normal_obj = {
        "string": "test",
        "number": 42,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
    
    result = safe_dumps(normal_obj)
    
    # Should be a valid JSON string that can be parsed
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == normal_obj


def test_safe_dumps_complex_metadata_like_object():
    """Test with a complex metadata-like object similar to what caused the issue"""
    
    # Simulate a complex metadata object
    metadata = {
        "user_api_key": "test-key",
        "model": "gpt-4",
        "usage": {"total_tokens": 100},
        "mcp_tool_call_metadata": {
            "name": "test_tool", 
            "arguments": {"param": "value"}
        }
    }
    
    # Add a potential circular reference
    usage_detail = {"parent_metadata": metadata}
    metadata["usage"]["detail"] = usage_detail
    
    # This should not raise an exception
    result = safe_dumps(metadata)
    
    # Should be a valid JSON string
    assert isinstance(result, str)
    
    # Should be parseable as JSON
    parsed = json.loads(result)
    assert parsed["user_api_key"] == "test-key"
    assert parsed["model"] == "gpt-4"


@patch("litellm.proxy.proxy_server.master_key", None)
@patch("litellm.proxy.proxy_server.general_settings", {})
def test_get_logging_payload_api_key_preserved_when_standard_logging_payload_is_none():
    """
    Critical - Product incident was caused by this bug.
    
    Test that api_key is NOT set to empty string when standard_logging_payload is None.
    
    This is a regression test for a bug where:
    - On failed requests (bad request errors), standard_logging_payload is None
    - The else block was incorrectly setting api_key = ""
    - This caused empty api_key in DailyUserSpend table despite SpendLogs having the correct key
    
    Expected behavior:
    - api_key from metadata should be extracted and hashed
    - Even when standard_logging_payload is None, the api_key should be preserved
    - The returned payload should have the hashed api_key, not empty string
    """
    # Setup: Simulate a failed request scenario
    test_api_key = "sk-WLi4iRn4JmbVlTaYw12IOA"
    
    # Create kwargs similar to what's passed during a bad request error
    kwargs = {
        "model": "openai/gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}],
        "call_type": "acompletion",
        "litellm_params": {
            "metadata": {
                "user_api_key": test_api_key,  # This is the key that should be preserved
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
            }
        },
        # Note: No 'standard_logging_object' in kwargs - simulating failure case
    }
    
    # Create a mock error response (bad request)
    response_obj = Exception("BadRequestError: Invalid parameter 'usersss'")
    
    # Create timestamps
    start_time = datetime.datetime.now(timezone.utc)
    end_time = datetime.datetime.now(timezone.utc)
    
    # Call get_logging_payload
    payload = get_logging_payload(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time
    )
    
    # CRITICAL ASSERTION: api_key should NOT be empty string
    assert payload["api_key"] != "", \
        "BUG: api_key is empty! When standard_logging_payload is None, " \
        "the api_key from metadata should be preserved and hashed."
    
    # The api_key should be hashed (not the raw key)
    assert payload["api_key"] != test_api_key, \
        "api_key should be hashed, not the raw key"
    
    # The api_key should be a valid hash (64 character hex string for SHA256)
    assert len(payload["api_key"]) == 64, \
        f"Expected 64 character hash, got {len(payload['api_key'])} characters"
    
    # Verify other fields are set correctly
    assert payload["model"] == "openai/gpt-4.1"
    assert payload["user"] == "test_user"
    
    print(f"✅ Test passed! api_key preserved: {payload['api_key']}")

