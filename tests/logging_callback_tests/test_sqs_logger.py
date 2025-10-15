import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import unquote

import litellm
import pytest

from litellm.integrations.sqs import SQSLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.mark.asyncio
async def test_async_sqs_logger_flush():
    expected_queue_url = "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
    expected_region = "us-east-1"
    
    sqs_logger = SQSLogger(
        sqs_queue_url=expected_queue_url,
        sqs_region_name=expected_region,
        sqs_flush_interval=1,
    )
    
    # Mock the httpx client
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    sqs_logger.async_httpx_client.post = AsyncMock(return_value=mock_response)
    
    litellm.callbacks = [sqs_logger]

    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="hi",
    )

    await asyncio.sleep(2)

    # Verify that httpx post was called
    sqs_logger.async_httpx_client.post.assert_called()
    
    # Get the call arguments
    call_args = sqs_logger.async_httpx_client.post.call_args
    
    # Verify the URL is correct
    called_url = call_args[0][0]  # First positional argument
    assert called_url == expected_queue_url, f"Expected URL {expected_queue_url}, got {called_url}"
    
    # Verify the payload contains StandardLoggingPayload data
    called_data = call_args.kwargs['data']
    
    # Extract the MessageBody from the URL-encoded data
    # Format: "Action=SendMessage&Version=2012-11-05&MessageBody=<url_encoded_json>"
    assert "Action=SendMessage" in called_data
    assert "Version=2012-11-05" in called_data
    assert "MessageBody=" in called_data
    
    # Extract and decode the message body
    message_body_start = called_data.find("MessageBody=") + len("MessageBody=")
    message_body_encoded = called_data[message_body_start:]
    message_body_json = unquote(message_body_encoded)
    
    # Parse the JSON to verify it's a StandardLoggingPayload
    payload_data = json.loads(message_body_json)
    
    # Verify it has the expected StandardLoggingPayload structure
    assert "model" in payload_data
    assert "messages" in payload_data
    assert "response" in payload_data
    assert payload_data["model"] == "gpt-4o"
    assert len(payload_data["messages"]) == 1
    assert payload_data["messages"][0]["role"] == "user"
    assert payload_data["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_async_sqs_logger_error_flush():
    expected_queue_url = "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
    expected_region = "us-east-1"

    sqs_logger = SQSLogger(
        sqs_queue_url=expected_queue_url,
        sqs_region_name=expected_region,
        sqs_flush_interval=1,
    )

    # Mock the httpx client
    mock_response = MagicMock()
    mock_response.raise_for_status = Exception("Something went wrong")
    sqs_logger.async_httpx_client.post = AsyncMock(return_value=mock_response)

    litellm.callbacks = [sqs_logger]

    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="Error occurred"
    )

    await asyncio.sleep(2)

    # Verify that httpx post was called
    sqs_logger.async_httpx_client.post.assert_called()

    # Get the call arguments
    call_args = sqs_logger.async_httpx_client.post.call_args

    # Verify the URL is correct
    called_url = call_args[0][0]  # First positional argument
    assert called_url == expected_queue_url, f"Expected URL {expected_queue_url}, got {called_url}"

    # Verify the payload contains StandardLoggingPayload data
    called_data = call_args.kwargs['data']

    # Extract the MessageBody from the URL-encoded data
    # Format: "Action=SendMessage&Version=2012-11-05&MessageBody=<url_encoded_json>"
    assert "Action=SendMessage" in called_data
    assert "Version=2012-11-05" in called_data
    assert "MessageBody=" in called_data

    # Extract and decode the message body
    message_body_start = called_data.find("MessageBody=") + len("MessageBody=")
    message_body_encoded = called_data[message_body_start:]
    message_body_json = unquote(message_body_encoded)

    # Parse the JSON to verify it's a StandardLoggingPayload
    payload_data = json.loads(message_body_json)

    # Verify it has the expected StandardLoggingPayload structure
    assert "model" in payload_data
    assert "messages" in payload_data
    assert "response" in payload_data
    assert payload_data["model"] == "gpt-4o"
    assert len(payload_data["messages"]) == 1
    assert payload_data["messages"][0]["role"] == "user"
    assert payload_data["messages"][0]["content"] == "hello"
