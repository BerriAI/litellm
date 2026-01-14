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
