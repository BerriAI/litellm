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
from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger


@pytest.mark.asyncio
async def test_generic_api_callback():
    """
    Test the GenericAPILogger callback with a standard logging payload.
    This test mocks the HTTP client and validates that the logger properly
    formats and sends the expected payload.
    """

    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "OK"

    # Set up an endpoint for testing
    test_endpoint = "https://example.com/api/logs"
    test_headers = {"Authorization": "Bearer test_token"}
    os.environ["GENERIC_LOGGER_ENDPOINT"] = test_endpoint

    # Initialize the GenericAPILogger and set the mock
    generic_logger = GenericAPILogger(
        endpoint=test_endpoint, headers=test_headers, flush_interval=1
    )
    generic_logger.async_httpx_client.post = mock_post
    litellm.callbacks = [generic_logger]

    # Make the completion call
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="hi",
        user="test_user",
    )

    # Wait for async flush
    await asyncio.sleep(3)

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    print("##########\n")
    print(
        "logs were flushed to URL",
        actual_url,
        "with the following headers",
        mock_post.call_args[1]["headers"],
    )
    assert (
        actual_url == test_endpoint
    ), f"Expected URL {test_endpoint}, got {actual_url}"

    # Validate headers
    assert (
        mock_post.call_args[1]["headers"]["Content-Type"] == "application/json"
    ), "Content-Type should be application/json"

    # For the GenericAPILogger, it sends the payload directly as JSON in the data field
    json_data = mock_post.call_args[1]["data"]
    # Parse the JSON string

    print("##########\n")
    print("json_data", json_data)
    actual_request = json.loads(json_data)

    # The payload is a list of StandardLoggingPayload objects in the log queue
    assert isinstance(actual_request, list), "Request body should be a list"
    assert len(actual_request) > 0, "Request body list should not be empty"

    # Validate the first payload item
    payload_item: StandardLoggingPayload = StandardLoggingPayload(**actual_request[0])
    print("##########\n")
    print(json.dumps(payload_item, indent=4))
    print("##########\n")

    # Basic assertions for standard logging payload
    assert payload_item["response_cost"] > 0, "Response cost should be greater than 0"
    assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"
    assert (
        payload_item["model_parameters"]["user"] == "test_user"
    ), "User should be test_user"
    assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"
    assert payload_item["messages"] == [
        {"role": "user", "content": "Hello, world!"}
    ], "Messages should be the same"
    assert (
        payload_item["response"]["choices"][0]["message"]["content"] == "hi"
    ), "Response should be hi"


@pytest.mark.asyncio
async def test_generic_api_callback_multiple_logs():
    """
    Test the GenericAPILogger callback with multiple chat completions
    """
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "OK"

    # Set up an endpoint for testing
    test_endpoint = "https://example.com/api/logs"
    test_headers = {"Authorization": "Bearer test_token"}
    os.environ["GENERIC_LOGGER_ENDPOINT"] = test_endpoint

    # Initialize the GenericAPILogger and set the mock
    generic_logger = GenericAPILogger(
        endpoint=test_endpoint, headers=test_headers, flush_interval=5
    )
    generic_logger.async_httpx_client.post = mock_post
    litellm.callbacks = [generic_logger]

    # Make the completion call
    for _ in range(10):
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="hi",
            user="test_user",
        )

    # Wait for async flush
    await asyncio.sleep(6)

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    print("##########\n")
    print(
        "logs were flushed to URL",
        actual_url,
        "with the following headers",
        mock_post.call_args[1]["headers"],
    )
    assert (
        actual_url == test_endpoint
    ), f"Expected URL {test_endpoint}, got {actual_url}"

    # For the GenericAPILogger, it sends the payload directly as JSON in the data field
    json_data = mock_post.call_args[1]["data"]
    # Parse the JSON string

    print("##########\n")
    print("json_data", json_data)
    actual_request = json.loads(json_data)

    # The payload is a list of StandardLoggingPayload objects in the log queue
    assert isinstance(actual_request, list), "Request body should be a list"
    assert len(actual_request) > 0, "Request body list should not be empty"
    assert (
        len(actual_request) == 10
    ), "Request body list should be 10 items, since we made 10 calls"

    # Validate all payload items
    for payload_item in actual_request:
        payload_item: StandardLoggingPayload = StandardLoggingPayload(**payload_item)
        print("##########\n")
        print(json.dumps(payload_item, indent=4))
        print("##########\n")

        assert (
            payload_item["response_cost"] > 0
        ), "Response cost should be greater than 0"
        assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"
        assert (
            payload_item["model_parameters"]["user"] == "test_user"
        ), "User should be test_user"
        assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"
        assert payload_item["messages"] == [
            {"role": "user", "content": "Hello, world!"}
        ], "Messages should be the same"
        assert (
            payload_item["response"]["choices"][0]["message"]["content"] == "hi"
        ), "Response should be hi"


@pytest.mark.asyncio
async def test_generic_api_callback_ndjson_format():
    """
    Test the GenericAPILogger callback with ndjson log format.
    Validates that logs are sent as newline-delimited JSON.
    """
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "OK"

    # Set up an endpoint for testing
    test_endpoint = "https://example.com/api/logs"
    test_headers = {"Authorization": "Bearer test_token"}
    os.environ["GENERIC_LOGGER_ENDPOINT"] = test_endpoint

    # Initialize the GenericAPILogger with ndjson format
    generic_logger = GenericAPILogger(
        endpoint=test_endpoint,
        headers=test_headers,
        flush_interval=1,
        log_format="ndjson"  # Set NDJSON format
    )
    generic_logger.async_httpx_client.post = mock_post
    litellm.callbacks = [generic_logger]

    # Make multiple completion calls to generate multiple logs
    for i in range(3):
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Hello, world! {i}"}],
            mock_response="hi",
            user="test_user",
        )

    # Wait for async flush
    await asyncio.sleep(3)

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Get the actual request body from the mock
    actual_url = mock_post.call_args[1]["url"]
    assert actual_url == test_endpoint, f"Expected URL {test_endpoint}, got {actual_url}"

    # Get the data sent
    ndjson_data = mock_post.call_args[1]["data"]
    print("##########\n")
    print("ndjson_data:", ndjson_data)
    print("##########\n")

    # Validate it's NDJSON format (newline-delimited)
    assert isinstance(ndjson_data, str), "Data should be a string for NDJSON"

    # Split by newlines and parse each line
    lines = ndjson_data.strip().split("\n")
    assert len(lines) == 3, f"Expected 3 lines of NDJSON, got {len(lines)}"

    # Validate each line is valid JSON
    for i, line in enumerate(lines):
        payload_item = json.loads(line)
        payload_item = StandardLoggingPayload(**payload_item)

        # Basic assertions
        assert payload_item["response_cost"] > 0, "Response cost should be greater than 0"
        assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"
        assert payload_item["model_parameters"]["user"] == "test_user", "User should be test_user"


@pytest.mark.asyncio
async def test_generic_api_callback_single_format():
    """
    Test the GenericAPILogger callback with single log format.
    Validates that each log is sent as an individual request in parallel.
    """
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "OK"

    # Set up an endpoint for testing
    test_endpoint = "https://example.com/api/logs"
    test_headers = {"Authorization": "Bearer test_token"}
    os.environ["GENERIC_LOGGER_ENDPOINT"] = test_endpoint

    # Initialize the GenericAPILogger with single format
    generic_logger = GenericAPILogger(
        endpoint=test_endpoint,
        headers=test_headers,
        flush_interval=1,  # Quick flush to trigger batch send
        log_format="single"  # Set single format
    )
    generic_logger.async_httpx_client.post = mock_post
    litellm.callbacks = [generic_logger]

    # Make 3 completion calls
    for i in range(3):
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Hello, world! {i}"}],
            mock_response="hi",
            user="test_user",
        )

    # Wait for async flush
    await asyncio.sleep(3)

    # Assert httpx post was called 3 times (once per log in batch)
    assert mock_post.call_count == 3, f"Expected 3 calls, got {mock_post.call_count}"

    # Validate each call sent a single log object (not an array)
    for call_idx in range(3):
        call_args = mock_post.call_args_list[call_idx]
        json_data = call_args[1]["data"]

        print(f"########## Call {call_idx} ##########")
        print("json_data:", json_data)

        # Parse and validate - should be a single object, not an array
        actual_request = json.loads(json_data)
        assert isinstance(actual_request, dict), f"Call {call_idx}: Expected dict, got {type(actual_request)}"

        # Validate it's a valid StandardLoggingPayload
        payload_item = StandardLoggingPayload(**actual_request)
        assert payload_item["response_cost"] > 0, "Response cost should be greater than 0"
        assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"


@pytest.mark.asyncio
async def test_generic_api_callback_json_array_format_explicit():
    """
    Test the GenericAPILogger callback with explicit json_array format.
    Validates backward compatibility when explicitly set to json_array.
    """
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "OK"

    # Set up an endpoint for testing
    test_endpoint = "https://example.com/api/logs"
    test_headers = {"Authorization": "Bearer test_token"}
    os.environ["GENERIC_LOGGER_ENDPOINT"] = test_endpoint

    # Initialize the GenericAPILogger with explicit json_array format
    generic_logger = GenericAPILogger(
        endpoint=test_endpoint,
        headers=test_headers,
        flush_interval=1,
        log_format="json_array"  # Explicitly set json_array
    )
    generic_logger.async_httpx_client.post = mock_post
    litellm.callbacks = [generic_logger]

    # Make multiple completion calls
    for i in range(5):
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Hello, world! {i}"}],
            mock_response="hi",
            user="test_user",
        )

    # Wait for async flush
    await asyncio.sleep(3)

    # Assert httpx post was called once (batched)
    mock_post.assert_called_once()

    # Get the data and validate it's a JSON array
    json_data = mock_post.call_args[1]["data"]
    actual_request = json.loads(json_data)

    assert isinstance(actual_request, list), "Request body should be a list (JSON array)"
    assert len(actual_request) == 5, f"Expected 5 items, got {len(actual_request)}"

    # Validate each item
    for payload_item in actual_request:
        payload_item = StandardLoggingPayload(**payload_item)
        assert payload_item["response_cost"] > 0, "Response cost should be greater than 0"
        assert payload_item["model"] == "gpt-4o", "Model should be gpt-4o"


@pytest.mark.asyncio
async def test_generic_api_callback_sumologic_uses_ndjson():
    """
    Test that the sumologic callback uses ndjson format by default
    when loaded from generic_api_compatible_callbacks.json
    """
    # Create a mock for the async_httpx_client's post method
    mock_post = AsyncMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "OK"

    # Set environment variable for sumologic
    os.environ["SUMOLOGIC_WEBHOOK_URL"] = "https://collectors.sumologic.com/receiver/v1/http/test123"

    # Initialize using callback_name (loads from JSON config)
    generic_logger = GenericAPILogger(
        callback_name="sumologic",
        flush_interval=1
    )
    generic_logger.async_httpx_client.post = mock_post
    litellm.callbacks = [generic_logger]

    # Verify the logger has ndjson format
    assert generic_logger.log_format == "ndjson", "Sumologic should use ndjson format"

    # Make completion calls
    for i in range(2):
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Test {i}"}],
            mock_response="response",
            user="test_user",
        )

    # Wait for async flush
    await asyncio.sleep(3)

    # Assert httpx post was called
    mock_post.assert_called_once()

    # Verify NDJSON format
    ndjson_data = mock_post.call_args[1]["data"]
    assert isinstance(ndjson_data, str), "Data should be a string for NDJSON"

    lines = ndjson_data.strip().split("\n")
    assert len(lines) == 2, f"Expected 2 lines of NDJSON, got {len(lines)}"

    # Each line should be valid JSON
    for line in lines:
        json.loads(line)  # Will raise if invalid JSON


@pytest.mark.asyncio
async def test_generic_api_callback_invalid_log_format():
    """
    Test that invalid log_format values raise a ValueError
    """
    test_endpoint = "https://example.com/api/logs"
    os.environ["GENERIC_LOGGER_ENDPOINT"] = test_endpoint

    with pytest.raises(ValueError, match="Invalid log_format"):
        GenericAPILogger(
            endpoint=test_endpoint,
            log_format="invalid_format"  # type: ignore  # Intentionally invalid for testing
        )
