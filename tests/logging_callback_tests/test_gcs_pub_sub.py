import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import litellm
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_pubsub.pub_sub import *
from datetime import datetime, timedelta
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingModelInformation,
    StandardLoggingMetadata,
    StandardLoggingHiddenParams,
)

verbose_logger.setLevel(logging.DEBUG)

ignored_keys = [
    "request_id",
    "session_id",
    "startTime",
    "endTime",
    "completionStartTime",
    "endTime",
    "metadata.model_map_information",
    "metadata.usage_object",
    "metadata.cold_storage_object_key",
    "metadata.litellm_overhead_time_ms",
    "metadata.cost_breakdown",
]


def _compare_nested_dicts(
    actual: dict, expected: dict, path: str = "", ignore_keys: list[str] = []
) -> list[str]:
    """Compare nested dictionaries and return a list of differences in a human-friendly format."""
    differences = []

    # Check if current path should be ignored
    if path in ignore_keys:
        return differences

    # Check for keys in actual but not in expected
    for key in actual.keys():
        current_path = f"{path}.{key}" if path else key
        if current_path not in ignore_keys and key not in expected:
            differences.append(f"Extra key in actual: {current_path}")

    for key, expected_value in expected.items():
        current_path = f"{path}.{key}" if path else key
        if current_path in ignore_keys:
            continue
        if key not in actual:
            differences.append(f"Missing key: {current_path}")
            continue

        actual_value = actual[key]

        # Try to parse JSON strings
        if isinstance(expected_value, str):
            try:
                expected_value = json.loads(expected_value)
            except json.JSONDecodeError:
                pass
        if isinstance(actual_value, str):
            try:
                actual_value = json.loads(actual_value)
            except json.JSONDecodeError:
                pass

        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            differences.extend(
                _compare_nested_dicts(
                    actual_value, expected_value, current_path, ignore_keys
                )
            )
        elif isinstance(expected_value, dict) or isinstance(actual_value, dict):
            differences.append(
                f"Type mismatch at {current_path}: expected dict, got {type(actual_value).__name__}"
            )
        else:
            # For non-dict values, only report if they're different
            if actual_value != expected_value:
                # Format the values to be more readable
                actual_str = str(actual_value)
                expected_str = str(expected_value)
                if len(actual_str) > 50 or len(expected_str) > 50:
                    actual_str = f"{actual_str[:50]}..."
                    expected_str = f"{expected_str[:50]}..."
                differences.append(
                    f"Value mismatch at {current_path}:\n  expected: {expected_str}\n  got:      {actual_str}"
                )
    return differences


def assert_gcs_pubsub_request_matches_expected(
    actual_request_body: dict,
    expected_file_name: str,
):
    """
    Helper function to compare actual GCS PubSub request body with expected JSON file.

    Args:
        actual_request_body (dict): The actual request body received from the API call
        expected_file_name (str): Name of the JSON file containing expected request body
    """
    # Get the current directory and read the expected request body
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(pwd, "gcs_pub_sub_body", expected_file_name)

    with open(expected_body_path, "r") as f:
        expected_request_body = json.load(f)

    # Replace dynamic values in actual request body
    differences = _compare_nested_dicts(
        actual_request_body, expected_request_body, ignore_keys=ignored_keys
    )
    if differences:
        assert False, f"Dictionary mismatch: {differences}"

def assert_gcs_pubsub_request_matches_expected_standard_logging_payload(
    actual_request_body: dict,
    expected_file_name: str,
):
    """
    Helper function to compare actual GCS PubSub request body with expected JSON file.

    Args:
        actual_request_body (dict): The actual request body received from the API call
        expected_file_name (str): Name of the JSON file containing expected request body
    """
    # Get the current directory and read the expected request body
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(pwd, "gcs_pub_sub_body", expected_file_name)

    with open(expected_body_path, "r") as f:
        expected_request_body = json.load(f)

    # Replace dynamic values in actual request body
    FIELDS_TO_VALIDATE = [
        "custom_llm_provider",
        "hidden_params",
        "messages",
        "response",
        "model",
        "status",
        "stream",
    ]

    actual_request_body["response"]["id"] = expected_request_body["response"]["id"]
    actual_request_body["response"]["created"] = expected_request_body["response"][
        "created"
    ]

    for field in FIELDS_TO_VALIDATE:
        assert field in actual_request_body

    FIELDS_EXISTENCE_CHECKS = [
        "response_cost",
        "response_time",
        "completion_tokens",
        "prompt_tokens",
        "total_tokens"
    ]

    for field in FIELDS_EXISTENCE_CHECKS:
        assert field in actual_request_body


@pytest.mark.asyncio
async def test_async_gcs_pub_sub():
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 202
    mock_post.return_value.text = "Accepted"

    # Initialize the GcsPubSubLogger and set the mock
    gcs_pub_sub_logger = GcsPubSubLogger(flush_interval=1)
    gcs_pub_sub_logger.async_httpx_client.post = mock_post

    mock_construct_request_headers = AsyncMock()
    mock_construct_request_headers.return_value = {"Authorization": "Bearer mock_token"}
    gcs_pub_sub_logger.construct_request_headers = mock_construct_request_headers
    litellm.callbacks = [gcs_pub_sub_logger]

    # Make the completion call
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="hi",
    )

    await asyncio.sleep(3)  # Wait for async flush

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    print("sent to url", actual_url)
    assert (
        actual_url
        == "https://pubsub.googleapis.com/v1/projects/reliableKeys/topics/litellmDB:publish"
    )
    actual_request = mock_post.call_args[1]["json"]

    # Extract and decode the base64 encoded message
    encoded_message = actual_request["messages"][0]["data"]
    import base64

    decoded_message = base64.b64decode(encoded_message).decode("utf-8")

    # Parse the JSON string into a dictionary
    actual_request = json.loads(decoded_message)
    print("##########\n")
    print(json.dumps(actual_request, indent=4))
    print("##########\n")
    # Verify the request body matches expected format
    assert_gcs_pubsub_request_matches_expected_standard_logging_payload(
        actual_request, "standard_logging_payload.json"
    )


@pytest.mark.asyncio
async def test_async_gcs_pub_sub_v1():
    # Create a mock for the async_httpx_client's post method
    litellm.gcs_pub_sub_use_v1 = True
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 202
    mock_post.return_value.text = "Accepted"

    # Initialize the GcsPubSubLogger and set the mock
    gcs_pub_sub_logger = GcsPubSubLogger(flush_interval=1)
    gcs_pub_sub_logger.async_httpx_client.post = mock_post

    mock_construct_request_headers = AsyncMock()
    mock_construct_request_headers.return_value = {"Authorization": "Bearer mock_token"}
    gcs_pub_sub_logger.construct_request_headers = mock_construct_request_headers
    litellm.callbacks = [gcs_pub_sub_logger]

    # Make the completion call
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="hi",
    )

    await asyncio.sleep(3)  # Wait for async flush

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    print("sent to url", actual_url)
    assert (
        actual_url
        == "https://pubsub.googleapis.com/v1/projects/reliableKeys/topics/litellmDB:publish"
    )
    actual_request = mock_post.call_args[1]["json"]

    # Extract and decode the base64 encoded message
    encoded_message = actual_request["messages"][0]["data"]
    import base64

    decoded_message = base64.b64decode(encoded_message).decode("utf-8")

    # Parse the JSON string into a dictionary
    actual_request = json.loads(decoded_message)
    print("##########\n")
    print(json.dumps(actual_request, indent=4))
    print("##########\n")
    # Verify the request body matches expected format
    assert_gcs_pubsub_request_matches_expected(
        actual_request, "spend_logs_payload.json"
    )
