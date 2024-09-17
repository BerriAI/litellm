import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.datadog.types import DatadogPayload

verbose_logger.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_datadog_logging_http_request():
    """
    - Test that the HTTP request is made to Datadog
    - sent to the /api/v2/logs endpoint
    - the payload is batched
    - each element in the payload is a DatadogPayload
    - each element in a DatadogPayload.message contains all the valid fields
    """
    try:
        from litellm.integrations.datadog.datadog import DataDogLogger

        os.environ["DD_SITE"] = "https://fake.datadoghq.com"
        os.environ["DD_API_KEY"] = "anything"
        dd_logger = DataDogLogger()

        litellm.callbacks = [dd_logger]

        litellm.set_verbose = True

        # Create a mock for the async_client's post method
        mock_post = AsyncMock()
        mock_post.return_value.status_code = 202
        mock_post.return_value.text = "Accepted"
        dd_logger.async_client.post = mock_post

        # Make the completion call
        for _ in range(5):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "what llm are u"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="Accepted",
            )
            print(response)

        # Wait for 5 seconds
        await asyncio.sleep(6)

        # Assert that the mock was called
        assert mock_post.called, "HTTP request was not made"

        # Get the arguments of the last call
        args, kwargs = mock_post.call_args

        print("CAll args and kwargs", args, kwargs)

        # Print the request body

        # You can add more specific assertions here if needed
        # For example, checking if the URL is correct
        assert kwargs["url"].endswith("/api/v2/logs"), "Incorrect DataDog endpoint"

        body = kwargs["data"]

        # use gzip to unzip the body
        with gzip.open(io.BytesIO(body), "rb") as f:
            body = f.read().decode("utf-8")
        print(body)

        # body is string parse it to dict
        body = json.loads(body)
        print(body)

        assert len(body) == 5  # 5 logs should be sent to DataDog

        # Assert that the first element in body has the expected fields and shape
        assert isinstance(body[0], dict), "First element in body should be a dictionary"

        # Get the expected fields and their types from DatadogPayload
        expected_fields = DatadogPayload.__annotations__
        # Assert that all elements in body have the fields of DatadogPayload with correct types
        for log in body:
            assert isinstance(log, dict), "Each log should be a dictionary"
            for field, expected_type in expected_fields.items():
                assert field in log, f"Field '{field}' is missing from the log"
                assert isinstance(
                    log[field], expected_type
                ), f"Field '{field}' has incorrect type. Expected {expected_type}, got {type(log[field])}"

        # Additional assertion to ensure no extra fields are present
        for log in body:
            assert set(log.keys()) == set(
                expected_fields.keys()
            ), f"Log contains unexpected fields: {set(log.keys()) - set(expected_fields.keys())}"

        # Parse the 'message' field as JSON and check its structure
        message = json.loads(body[0]["message"])

        expected_message_fields = [
            "id",
            "call_type",
            "cache_hit",
            "start_time",
            "end_time",
            "response_time",
            "model",
            "user",
            "model_parameters",
            "spend",
            "messages",
            "response",
            "usage",
            "metadata",
        ]

        for field in expected_message_fields:
            assert field in message, f"Field '{field}' is missing from the message"

        # Check specific fields
        assert message["call_type"] == "acompletion"
        assert message["model"] == "gpt-3.5-turbo"
        assert isinstance(message["model_parameters"], dict)
        assert "temperature" in message["model_parameters"]
        assert "max_tokens" in message["model_parameters"]
        assert isinstance(message["response"], dict)
        assert isinstance(message["usage"], dict)
        assert isinstance(message["metadata"], dict)

    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")


@pytest.mark.asyncio
@pytest.mark.skip(reason="local-only test, to test if everything works fine.")
async def test_datadog_logging():
    try:
        litellm.success_callback = ["datadog"]
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)

        await asyncio.sleep(5)
    except Exception as e:
        print(e)
