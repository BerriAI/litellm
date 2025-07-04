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
from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _get_vector_store_request_for_spend_logs_payload,
    _sanitize_request_body_for_spend_logs_payload,
)


def test_sanitize_request_body_for_spend_logs_payload_basic():
    request_body = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    assert _sanitize_request_body_for_spend_logs_payload(request_body) == request_body


def test_sanitize_request_body_for_spend_logs_payload_long_string():
    long_string = "a" * 2000  # Create a string longer than MAX_STRING_LENGTH
    request_body = {"text": long_string, "normal_text": "short text"}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert len(sanitized["text"]) == 1000 + len("... (truncated 1000 chars)")
    assert sanitized["normal_text"] == "short text"


def test_sanitize_request_body_for_spend_logs_payload_nested_dict():
    request_body = {"outer": {"inner": {"text": "a" * 2000, "normal": "short"}}}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert len(sanitized["outer"]["inner"]["text"]) == 1000 + len(
        "... (truncated 1000 chars)"
    )
    assert sanitized["outer"]["inner"]["normal"] == "short"


def test_sanitize_request_body_for_spend_logs_payload_nested_list():
    request_body = {
        "items": [{"text": "a" * 2000}, {"text": "short"}, [{"text": "a" * 2000}]]
    }
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert len(sanitized["items"][0]["text"]) == 1000 + len(
        "... (truncated 1000 chars)"
    )
    assert sanitized["items"][1]["text"] == "short"
    assert len(sanitized["items"][2][0]["text"]) == 1000 + len(
        "... (truncated 1000 chars)"
    )


def test_sanitize_request_body_for_spend_logs_payload_non_string_values():
    request_body = {"number": 42, "boolean": True, "none": None, "float": 3.14}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert sanitized == request_body


def test_sanitize_request_body_for_spend_logs_payload_empty():
    request_body: dict[str, Any] = {}
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert sanitized == request_body


def test_sanitize_request_body_for_spend_logs_payload_mixed_types():
    request_body = {
        "text": "a" * 2000,
        "number": 42,
        "nested": {"list": ["short", "a" * 2000], "dict": {"key": "a" * 2000}},
    }
    sanitized = _sanitize_request_body_for_spend_logs_payload(request_body)
    assert len(sanitized["text"]) == 1000 + len("... (truncated 1000 chars)")
    assert sanitized["number"] == 42
    assert sanitized["nested"]["list"][0] == "short"
    assert len(sanitized["nested"]["list"][1]) == 1000 + len(
        "... (truncated 1000 chars)"
    )
    assert len(sanitized["nested"]["dict"]["key"]) == 1000 + len(
        "... (truncated 1000 chars)"
    )


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
