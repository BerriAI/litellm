import asyncio
import datetime
import json
import os
import sys
from datetime import timezone
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _get_response_for_spend_logs_payload,
    _get_vector_store_request_for_spend_logs_payload,
    _sanitize_request_body_for_spend_logs_payload,
    get_logging_payload,
)
from litellm.types.utils import StandardLoggingPayload


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


@patch(
    "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs"
)
def test_get_response_for_spend_logs_payload_truncates_large_base64(mock_should_store):
    from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB

    mock_should_store.return_value = True
    large_text = "A" * (MAX_STRING_LENGTH_PROMPT_IN_DB + 500)
    payload = cast(
        StandardLoggingPayload,
        {
        "response": {
            "data": [
                {
                    "b64_json": large_text,
                    "other_field": "value",
                }
            ]
        }
        },
    )

    response_json = _get_response_for_spend_logs_payload(payload)
    parsed = json.loads(response_json)
    truncated_value = parsed["data"][0]["b64_json"]
    assert len(truncated_value) < len(large_text)
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in truncated_value
    assert parsed["data"][0]["other_field"] == "value"


@patch(
    "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs"
)
def test_get_response_for_spend_logs_payload_truncates_large_embedding(mock_should_store):
    from litellm.constants import MAX_STRING_LENGTH_PROMPT_IN_DB

    mock_should_store.return_value = True
    embedding_values = [
        round(i * 0.0001, 6) for i in range(MAX_STRING_LENGTH_PROMPT_IN_DB + 500)
    ]
    large_embedding = json.dumps(embedding_values)
    payload = cast(
        StandardLoggingPayload,
        {
            "response": {
                "data": [
                    {
                        "embedding": large_embedding,
                        "other_field": "value",
                    }
                ]
            }
        },
    )

    response_json = _get_response_for_spend_logs_payload(payload)
    parsed = json.loads(response_json)
    truncated_value = parsed["data"][0]["embedding"]
    
    assert isinstance(truncated_value, str)
    assert len(truncated_value) < len(large_embedding)
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in truncated_value
    assert parsed["data"][0]["other_field"] == "value"


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


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.master_key", "sk-master-key")
@patch("litellm.proxy.proxy_server.general_settings", {})
async def test_api_key_preserved_through_failure_hook_to_database():
    """
    CRITICAL E2E TEST: Validates the COMPLETE code path from failure hook to database.
    
    This is THE comprehensive test that protects against the production incident.
    It tests the EXACT flow that caused the bug:
    
    1. async_post_call_failure_hook is called with api_key in UserAPIKeyAuth
    2. Failure hook calls update_database with the token parameter
    3. update_database calls get_logging_payload to create payload
    4. BUG WAS HERE: get_logging_payload set api_key = "" when standard_logging_payload was None
    5. Empty api_key was written to DailyUserSpend table
    
    This test validates the ENTIRE flow to ensure the bug cannot regress.
    If this test fails in CI/CD, the build MUST fail.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.proxy.utils import hash_token

    # Setup
    test_api_key = "sk-test-critical-e2e-key"
    hashed_key = hash_token(test_api_key)
    
    # Track what payload gets created
    captured_payloads = []
    
    async def mock_update_database(
        token, response_cost, user_id, end_user_id, team_id,
        kwargs, completion_response, start_time, end_time, org_id
    ):
        """Mock update_database and capture the payload it creates"""
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            get_logging_payload,
        )

        # Call get_logging_payload EXACTLY as update_database does
        payload = get_logging_payload(
            kwargs=kwargs,
            response_obj=completion_response,
            start_time=start_time,
            end_time=end_time
        )
        
        captured_payloads.append({
            "token": token,
            "payload": payload,
        })
    
    # Mock dependencies
    mock_db_writer = MagicMock()
    mock_db_writer.update_database = AsyncMock(side_effect=mock_update_database)
    
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.db_spend_update_writer = mock_db_writer
    
    # Create UserAPIKeyAuth (what the failure hook receives)
    user_api_key_dict = UserAPIKeyAuth(
        api_key=hashed_key,
        user_id="test_user",
        team_id="test_team",
        max_budget=None,
        spend=0.0,
        key_alias="test-key",
        budget_reset_at=None,
        user_email=None,
        org_id="test_org",
        team_alias=None,
        end_user_id=None,
        request_route="/chat/completions",
        metadata={}
    )
    
    # Request data with bad parameter (triggers failure)
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "invalid_param": "causes_400_error",  # BAD PARAMETER
        "litellm_params": {
            "metadata": {
                "user_api_key": hashed_key,
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
            }
        }
    }
    
    exception = Exception("BadRequestError: Invalid parameter 'invalid_param'")
    
    # Execute the ACTUAL failure hook code path
    logger = _ProxyDBLogger()
    
    with patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj):
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=exception,
            user_api_key_dict=user_api_key_dict,
            traceback_str=None
        )
        
        await asyncio.sleep(0.1)  # Wait for async operations
    
    # =========================================================================
    # CRITICAL ASSERTIONS - If ANY fail, the production bug has regressed!
    # =========================================================================
    
    assert len(captured_payloads) == 1, "update_database should be called once"
    
    data = captured_payloads[0]
    payload = data["payload"]
    payload_api_key = payload.get("api_key")
    
    # THE CRITICAL ASSERTION - This would fail with the original bug!
    assert payload_api_key != "", \
        "🚨 CRITICAL BUG: payload['api_key'] is empty! " \
        "This is the EXACT production incident bug. " \
        "get_logging_payload() is setting api_key = '' when " \
        "standard_logging_payload is None (failure case)."
    
    assert payload_api_key is not None, \
        "🚨 CRITICAL: payload['api_key'] is None!"
    
    assert payload_api_key == hashed_key, \
        f"🚨 CRITICAL: Expected api_key={hashed_key}, got {payload_api_key}"
    
    # Verify token parameter matches
    assert data["token"] == hashed_key, \
        f"Token parameter should be {hashed_key}"
    
    # Verify other fields
    assert payload.get("model") == "gpt-3.5-turbo"
    assert payload.get("user") == "test_user"
    
    print("\n" + "="*80)
    print("✅ CRITICAL E2E TEST PASSED")
    print("="*80)
    print(f"Token: {data['token']}")
    print(f"Payload api_key: {payload_api_key}")
    print(f"Match: {data['token'] == payload_api_key}")
    print("="*80)
    print("Production incident bug is FIXED and protected:")
    print("- Failed requests preserve api_key through entire flow")
    print("- Both SpendLogs AND DailyUserSpend will have correct api_key")
    print("="*80 + "\n")


@patch("litellm.proxy.proxy_server.master_key", None)
@patch("litellm.proxy.proxy_server.general_settings", {})
def test_get_logging_payload_includes_agent_id_from_kwargs():
    """
    Test that get_logging_payload extracts agent_id from kwargs and includes it in the payload.
    """
    test_agent_id = "agent-uuid-12345"

    kwargs = {
        "model": "a2a_agent/test-agent",
        "custom_llm_provider": "a2a_agent",
        "agent_id": test_agent_id,
        "litellm_params": {
            "metadata": {
                "user_api_key": "sk-test-key",
            }
        },
    }

    response_obj = {
        "id": "test-response-123",
        "jsonrpc": "2.0",
        "result": {"status": "completed"},
    }

    start_time = datetime.datetime.now(timezone.utc)
    end_time = datetime.datetime.now(timezone.utc)

    payload = get_logging_payload(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=start_time,
        end_time=end_time,
    )

    assert payload["agent_id"] == test_agent_id, f"Expected agent_id '{test_agent_id}', got '{payload.get('agent_id')}'"

