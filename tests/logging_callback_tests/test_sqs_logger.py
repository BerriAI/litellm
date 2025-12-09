import asyncio
import base64
import json
import os
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import unquote

import litellm
import pytest

from litellm.integrations.sqs import SQSLogger
from litellm.types.utils import StandardLoggingPayload

from litellm.litellm_core_utils.app_crypto import AppCrypto


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



# =============================================================================
# 📥 Logging Queue Tests
# =============================================================================

@pytest.mark.asyncio
async def test_async_log_success_event_adds_to_queue(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    fake_payload = {"some": "data"}
    await logger.async_log_success_event(
        {"standard_logging_object": fake_payload}, None, None, None
    )
    assert fake_payload in logger.log_queue


@pytest.mark.asyncio
async def test_async_log_failure_event_adds_to_queue(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    fake_payload = {"fail": True}
    await logger.async_log_failure_event(
        {"standard_logging_object": fake_payload}, None, None, None
    )
    assert fake_payload in logger.log_queue



# =============================================================================
# 🧾 async_send_batch Tests
# =============================================================================

@pytest.mark.asyncio
async def test_async_send_batch_triggers_tasks(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")
    logger.async_send_message = AsyncMock()

    logger.log_queue = [{"log": 1}, {"log": 2}]
    await logger.async_send_batch()

    assert logger.async_send_message.await_count == 0  # uses create_task internally



# =============================================================================
# 🔐 AppCrypto Tests
# =============================================================================

def test_appcrypto_encrypt_decrypt_roundtrip():
    key = os.urandom(32)
    crypto = AppCrypto(key)
    data = {"event": "test", "value": 42}
    aad = b"context"
    enc = crypto.encrypt_json(data, aad=aad)
    dec = crypto.decrypt_json(enc, aad=aad)
    assert dec == data


def test_appcrypto_invalid_key_length():
    with pytest.raises(ValueError, match="32 bytes"):
        AppCrypto(b"short")


# =============================================================================
# 🪣 SQSLogger Initialization Tests
# =============================================================================

def test_sqs_logger_init_without_encryption(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    # Patch asyncio.create_task to avoid RuntimeError
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")
    assert logger.sqs_queue_url == "https://example.com"
    assert logger.app_crypto is None


def test_sqs_logger_init_with_encryption(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    key_b64 = base64.b64encode(os.urandom(32)).decode()

    logger = SQSLogger(
        sqs_queue_url="https://example.com",
        sqs_region_name="us-west-2",
        sqs_aws_use_application_level_encryption=True,
        sqs_app_encryption_key_b64=key_b64,
        sqs_app_encryption_aad="tenant=bill",
    )
    assert logger.app_crypto is not None
    assert logger.sqs_app_encryption_aad == "tenant=bill"


def test_sqs_logger_init_with_encryption_missing_key(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    with pytest.raises(ValueError, match="required when encryption is enabled"):
        SQSLogger(
            sqs_queue_url="https://example.com",
            sqs_region_name="us-west-2",
            sqs_aws_use_application_level_encryption=True,
        )


# =============================================================================
# 📥 Logging Queue Tests
# =============================================================================

@pytest.mark.asyncio
async def test_async_log_success_event_adds_to_queue(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    fake_payload = {"some": "data"}
    await logger.async_log_success_event(
        {"standard_logging_object": fake_payload}, None, None, None
    )
    assert fake_payload in logger.log_queue


@pytest.mark.asyncio
async def test_async_log_failure_event_adds_to_queue(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    fake_payload = {"fail": True}
    await logger.async_log_failure_event(
        {"standard_logging_object": fake_payload}, None, None, None
    )
    assert fake_payload in logger.log_queue


# =============================================================================
# 🧾 async_send_batch Tests
# =============================================================================

@pytest.mark.asyncio
async def test_async_send_batch_triggers_tasks(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    logger.async_send_message = AsyncMock()
    logger.log_queue = [{"log": 1}, {"log": 2}]

    await logger.async_send_batch()
    # It uses asyncio.create_task() so direct await count = 0 is expected
    asyncio.create_task.assert_called()



@pytest.mark.asyncio
async def test_strip_base64_removes_file_and_nontext_entries():
    logger = SQSLogger(sqs_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "image", "file": {"file_data": "data:image/png;base64,AAAA"}},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,BBBB"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response"},
                    {"type": "audio", "file": {"file_data": "data:audio/wav;base64,CCCC"}},
                ],
            },
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    # 1️⃣ All file/image/audio entries removed
    assert len(stripped["messages"][0]["content"]) == 1
    assert stripped["messages"][0]["content"][0]["text"] == "Hello world"

    assert len(stripped["messages"][1]["content"]) == 1
    assert stripped["messages"][1]["content"][0]["text"] == "Response"

    # 2️⃣ No residual 'file' keys left
    for msg in stripped["messages"]:
        for content in msg["content"]:
            assert "file" not in content
            assert content.get("type") == "text"


@pytest.mark.asyncio
async def test_strip_base64_keeps_non_file_content():
    logger = SQSLogger(sqs_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Just text"},
                    {"type": "text", "text": "Another message"},
                ],
            }
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    # Should not modify normal text messages
    assert stripped["messages"][0]["content"] == payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_strip_base64_handles_empty_or_missing_messages():
    logger = SQSLogger(sqs_strip_base64_files=True)

    payload_no_messages = {}
    stripped1 = await logger._strip_base64_from_messages(payload_no_messages)
    assert stripped1 == payload_no_messages

    payload_empty = {"messages": []}
    stripped2 = await logger._strip_base64_from_messages(payload_empty)
    assert stripped2 == payload_empty


@pytest.mark.asyncio
async def test_strip_base64_mixed_nested_objects():
    """
    Handles weird/nested content structures gracefully.
    """
    logger = SQSLogger(sqs_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Keep me"},
                    {"type": "custom", "metadata": "ignore but non-text"},
                    {"foo": "bar"},
                    {"file": {"file_data": "data:application/pdf;base64,XXX"}},
                ],
                "extra": {"trace_id": "123"},
            }
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    # 'custom' (non-text) and 'file' entries removed
    content = stripped["messages"][0]["content"]
    assert len(content) == 2
    assert {"type": "text", "text": "Keep me"} in content
    assert {"foo": "bar"} in content
    # Other metadata stays
    assert stripped["messages"][0]["extra"]["trace_id"] == "123"


@pytest.mark.asyncio
async def test_strip_base64_recursive_redaction():
    logger = SQSLogger(sqs_strip_base64_files=True)
    payload = {
        "messages": [
            {
                "content": [
                    {"type": "text", "text": "normal text"},
                    {"type": "text", "text": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"},
                    {"type": "text", "text": "Nested: {'data': 'data:application/pdf;base64,AAA...'}"},
                    {"file": {"file_data": "data:application/pdf;base64,AAAA"}},
                    {"metadata": {"preview": "data:audio/mp3;base64,AAAAA=="}},
                ]
            }
        ]
    }

    result = await logger._strip_base64_from_messages(payload)
    content = result["messages"][0]["content"]

    # Dropped file-type entry
    assert not any("file" in c for c in content)
    # Base64 redacted globally
    for c in content:
        if isinstance(c, dict):
            s = json.dumps(c).lower()
            # allow "[base64_redacted]" but nothing else
            assert "base64," not in s, f"Found real base64 blob in: {s}"


@pytest.mark.asyncio
async def test_async_health_check_healthy(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")
    logger.async_send_message = AsyncMock(return_value=None)

    result = await logger.async_health_check()
    assert result["status"] == "healthy"
    assert result.get("error_message") is None


@pytest.mark.asyncio
async def test_async_health_check_unhealthy(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")
    logger.async_send_message = AsyncMock(side_effect=Exception("boom"))

    result = await logger.async_health_check()
    assert result["status"] == "unhealthy"
    assert "boom" in (result.get("error_message") or "")
