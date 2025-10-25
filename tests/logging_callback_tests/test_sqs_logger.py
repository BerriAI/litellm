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


@pytest.fixture
async def logger():
    async def _make():
        return SQSLogger(sqs_strip_base64_files=True)
    return await _make()

# === helper ===
def make_payload(content):
    """Minimal StandardLoggingPayload-like dict for testing"""
    return {"messages": [{"role": "user", "content": content}]}


# === TEST CASES ===

@pytest.mark.asyncio
async def test_pdf_base64_redaction(logger):
    pdf_data = (
        "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MK..."
    )
    payload = make_payload([{"file": {"file_data": pdf_data}}])

    stripped = await logger._strip_base64_from_messages(payload)

    file_data = stripped["messages"][0]["content"][0]["file"]["file_data"]
    assert "[base64 PDF content redacted]" in file_data
    # confirm no raw base64 remains
    assert "JVBERi0x" not in file_data


@pytest.mark.asyncio
async def test_image_base64_redaction(logger):
    img_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
    payload = make_payload([{"file": {"file_data": img_data}}])

    stripped = await logger._strip_base64_from_messages(payload)

    redacted = stripped["messages"][0]["content"][0]["file"]["file_data"]
    assert redacted == "[base64 image content redacted]"


@pytest.mark.asyncio
async def test_audio_base64_redaction(logger):
    audio_data = "data:audio/wav;base64,UklGRigAAABXQVZFZm10..."
    payload = make_payload([{"file": {"file_data": audio_data}}])

    stripped = await logger._strip_base64_from_messages(payload)
    val = stripped["messages"][0]["content"][0]["file"]["file_data"]
    assert val == "[base64 audio content redacted]"


@pytest.mark.asyncio
async def test_unknown_mime_redaction(logger):
    data = "data:application/octet-stream;base64,AAAAAABBBBCCCC"
    payload = make_payload([{"file": {"file_data": data}}])

    stripped = await logger._strip_base64_from_messages(payload)
    val = stripped["messages"][0]["content"][0]["file"]["file_data"]
    assert val == "[base64 file content redacted]"


@pytest.mark.asyncio
async def test_non_base64_untouched(logger):
    non_base64 = "some-plain-text-value"
    payload = make_payload([{"file": {"file_data": non_base64}}])

    stripped = await logger._strip_base64_from_messages(payload)
    assert stripped["messages"][0]["content"][0]["file"]["file_data"] == non_base64


@pytest.mark.asyncio
async def test_nested_structure(logger):
    """Deeply nested dicts/lists should all be stripped."""
    pdf_data = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MK..."
    img_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
    nested_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"file": {"file_data": pdf_data}},
                    {
                        "extra": [
                            {"file": {"file_data": img_data}},
                            {"text": "keep me"},
                        ]
                    },
                ],
            }
        ]
    }

    stripped = await logger._strip_base64_from_messages(nested_payload)
    msg = stripped["messages"][0]["content"]

    pdf_part = msg[0]["file"]["file_data"]
    image_part = msg[1]["extra"][0]["file"]["file_data"]
    text_part = msg[1]["extra"][1]["text"]

    assert pdf_part == "[base64 PDF content redacted]"
    assert image_part == "[base64 image content redacted]"
    assert text_part == "keep me"


@pytest.mark.asyncio
async def test_response_section_also_redacted(logger):
    pdf_data = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MK..."
    payload = {
        "messages": [{"role": "user", "content": [{"file": {"file_data": pdf_data}}]}],
        "response": [{"file": {"file_data": pdf_data}}],
    }

    stripped = await logger._strip_base64_from_messages(payload)
    assert stripped["response"][0]["file"]["file_data"] == "[base64 PDF content redacted]"