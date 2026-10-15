import asyncio
import base64
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Final, cast
from urllib.parse import parse_qs
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.integrations.sqs import SQSLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def _fixed_signing_time() -> datetime:
    return datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_async_send_batch_dispatches_queued_payload(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr("botocore.auth.get_current_datetime", _fixed_signing_time)
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    queue_url: Final = "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
    sent: Final = asyncio.Event()
    payload: Final = cast(
        StandardLoggingPayload,
        {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "content": "queued request"}],
        },
    )

    async def respond(_request: httpx.Request) -> httpx.Response:
        sent.set()
        return httpx.Response(200)

    route: Final = respx_mock.post(queue_url).mock(side_effect=respond)
    logger: Final = SQSLogger(
        sqs_queue_url=queue_url,
        sqs_region_name="us-east-1",
        sqs_aws_access_key_id="unit-access-key",
        sqs_aws_secret_access_key="unit-secret-key",
    )
    logger.log_queue = [payload]

    await logger.async_send_batch()
    await sent.wait()

    assert route.call_count == 1
    assert json.loads(parse_qs(route.calls.last.request.content.decode())["MessageBody"][0]) == payload


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
async def test_async_send_message_posts_the_queue_url_and_encoded_payload(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr("botocore.auth.get_current_datetime", _fixed_signing_time)
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    queue_url: Final = "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
    payload: Final = cast(
        StandardLoggingPayload,
        {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "content": "log this request"}],
            "response": {"choices": [{"message": {"content": "logged"}}]},
        },
    )
    route: Final = respx_mock.post(queue_url).mock(return_value=httpx.Response(200))
    logger: Final = SQSLogger(
        sqs_queue_url=queue_url,
        sqs_region_name="us-east-1",
        sqs_aws_access_key_id="unit-access-key",
        sqs_aws_secret_access_key="unit-secret-key",
    )

    await logger.async_send_message(payload)

    body: Final = parse_qs(route.calls.last.request.content.decode())
    assert route.call_count == 1
    assert body["Action"] == ["SendMessage"]
    assert body["Version"] == ["2012-11-05"]
    assert json.loads(body["MessageBody"][0]) == payload


@pytest.mark.asyncio
async def test_async_send_message_handles_a_rejected_queue_request(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr("botocore.auth.get_current_datetime", _fixed_signing_time)
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    queue_url: Final = "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
    payload: Final = cast(
        StandardLoggingPayload,
        {
            "model": "openai/gpt-5.5",
            "messages": [{"role": "user", "content": "log this request"}],
            "response": {"choices": [{"message": {"content": "logged"}}]},
        },
    )
    route: Final = respx_mock.post(queue_url).mock(return_value=httpx.Response(403, text="rejected"))
    logger: Final = SQSLogger(
        sqs_queue_url=queue_url,
        sqs_region_name="us-east-1",
        sqs_aws_access_key_id="unit-access-key",
        sqs_aws_secret_access_key="unit-secret-key",
    )

    await logger.async_send_message(payload)

    body: Final = parse_qs(route.calls.last.request.content.decode())
    assert route.call_count == 1
    assert json.loads(body["MessageBody"][0]) == payload


@pytest.mark.asyncio
async def test_strip_base64_returns_text_only_messages_unchanged() -> None:
    logger: Final = SQSLogger(sqs_strip_base64_files=True)
    payload: Final = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "text", "text": "Keep this sentence"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
        ]
    }

    assert await logger._strip_base64_from_messages(payload) == payload


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
