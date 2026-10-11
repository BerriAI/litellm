import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.sqs import SQSLogger


@pytest.mark.asyncio
async def test_async_send_batch_does_not_await_send_directly(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    spawned = []
    real_create_task = asyncio.create_task

    def spy_create_task(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        spawned.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", spy_create_task)

    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")
    logger.async_send_message = AsyncMock()
    logger.log_queue = [{"log": 1}, {"log": 2}]

    try:
        await logger.async_send_batch()
        assert logger.async_send_message.await_count == 0
    finally:
        for task in spawned:
            task.cancel()
        await asyncio.gather(*spawned, return_exceptions=True)


def test_sqs_logger_init_without_encryption(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
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


@pytest.mark.asyncio
async def test_async_log_success_event_adds_to_queue(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    fake_payload = {"some": "data"}
    await logger.async_log_success_event({"standard_logging_object": fake_payload}, None, None, None)
    assert fake_payload in logger.log_queue


@pytest.mark.asyncio
async def test_async_log_failure_event_adds_to_queue(monkeypatch):
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger = SQSLogger(sqs_queue_url="https://example.com", sqs_region_name="us-west-2")

    fake_payload = {"fail": True}
    await logger.async_log_failure_event({"standard_logging_object": fake_payload}, None, None, None)
    assert fake_payload in logger.log_queue


@pytest.mark.asyncio
async def test_strip_base64_removes_file_and_nontext_entries():
    logger = SQSLogger(sqs_strip_base64_files=True)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {
                        "type": "image",
                        "file": {"file_data": "data:image/png;base64,AAAA"},
                    },
                    {
                        "type": "file",
                        "file": {"file_data": "data:application/pdf;base64,BBBB"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response"},
                    {
                        "type": "audio",
                        "file": {"file_data": "data:audio/wav;base64,CCCC"},
                    },
                ],
            },
        ]
    }

    stripped = await logger._strip_base64_from_messages(payload)

    assert len(stripped["messages"][0]["content"]) == 1
    assert stripped["messages"][0]["content"][0]["text"] == "Hello world"

    assert len(stripped["messages"][1]["content"]) == 1
    assert stripped["messages"][1]["content"][0]["text"] == "Response"

    for msg in stripped["messages"]:
        for content in msg["content"]:
            assert "file" not in content
            assert content.get("type") == "text"


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

    content = stripped["messages"][0]["content"]
    assert len(content) == 2
    assert {"type": "text", "text": "Keep me"} in content
    assert {"foo": "bar"} in content
    assert stripped["messages"][0]["extra"]["trace_id"] == "123"


@pytest.mark.asyncio
async def test_strip_base64_recursive_redaction():
    logger = SQSLogger(sqs_strip_base64_files=True)
    payload = {
        "messages": [
            {
                "content": [
                    {"type": "text", "text": "normal text"},
                    {
                        "type": "text",
                        "text": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg",
                    },
                    {
                        "type": "text",
                        "text": "Nested: {'data': 'data:application/pdf;base64,AAA...'}",
                    },
                    {"file": {"file_data": "data:application/pdf;base64,AAAA"}},
                    {"metadata": {"preview": "data:audio/mp3;base64,AAAAA=="}},
                ]
            }
        ]
    }

    result = await logger._strip_base64_from_messages(payload)
    content = result["messages"][0]["content"]

    assert not any("file" in c for c in content)
    for c in content:
        if isinstance(c, dict):
            s = json.dumps(c).lower()
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
