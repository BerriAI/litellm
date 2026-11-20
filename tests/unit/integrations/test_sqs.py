import asyncio
import base64
import json
import logging
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Final, cast
from unittest.mock import MagicMock
from urllib.parse import parse_qs

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.sqs import SQSLogger
from litellm.litellm_core_utils.app_crypto import AppCrypto
from litellm.litellm_core_utils.litellm_logging import create_dummy_standard_logging_payload
from litellm.types.utils import StandardLoggingPayload

_QUEUE_URL: Final = "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue"
_ENCRYPTED_MESSAGE: Final = TypeAdapter(dict[str, dict[str, str] | bool])
_TOOLS: Final = [
    {
        "type": "function",
        "function": {"name": f"tool_{i}", "description": "d" * 400, "parameters": {"type": "object"}},
    }
    for i in range(20)
]
_MODEL_INFO: Final = {"key": "gpt-5.5", "supported_openai_params": [f"param_{i}" for i in range(200)]}
_MODEL_MAP: Final = {"model_map_key": "gpt-5.5", "model_map_value": _MODEL_INFO}
_MODEL_MAP_WITHOUT_VALUE: Final = {"model_map_key": "gpt-5.5", "model_map_value": None}
_PARAMETERS: Final = {"temperature": 0.2, "tools": _TOOLS}
_PARAMETERS_WITHOUT_TOOLS: Final = {"temperature": 0.2}
_CHAT_RESPONSE: Final = {"id": "chatcmpl-1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
_RESPONSES_API_RESPONSE_WITHOUT_TOOLS: Final = {"id": "resp_1", "status": "completed", "output": [{"type": "message"}]}
_RESPONSES_API_RESPONSE: Final = {**_RESPONSES_API_RESPONSE_WITHOUT_TOOLS, "tools": _TOOLS}
_TEXT_RESPONSE: Final = "a transcription returned as plain text"


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


def _logged_payload(
    response: object, model_parameters: object, model_map_information: object
) -> StandardLoggingPayload:
    return cast(
        StandardLoggingPayload,
        {
            **json.loads(json.dumps(create_dummy_standard_logging_payload(), default=float)),
            "litellm_call_id": "call-with-tools",
            "response": response,
            "model_parameters": model_parameters,
            "model_map_information": model_map_information,
        },
    )


def _chat_payload() -> StandardLoggingPayload:
    return _logged_payload(_CHAT_RESPONSE, _PARAMETERS, _MODEL_MAP)


def _chat_payload_without_model_map_value() -> StandardLoggingPayload:
    return _logged_payload(_CHAT_RESPONSE, _PARAMETERS, _MODEL_MAP_WITHOUT_VALUE)


def _chat_payload_without_model_map_value_and_tools() -> StandardLoggingPayload:
    return _logged_payload(_CHAT_RESPONSE, _PARAMETERS_WITHOUT_TOOLS, _MODEL_MAP_WITHOUT_VALUE)


def _size(payload: StandardLoggingPayload) -> int:
    return len(json.dumps(payload).encode("utf-8"))


def _sqs_logger(
    monkeypatch: pytest.MonkeyPatch,
    sqs_max_message_bytes: int,
    encryption_key: bytes | None = None,
) -> SQSLogger:
    monkeypatch.setattr("botocore.auth.get_current_datetime", _fixed_signing_time)
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    return SQSLogger(
        sqs_queue_url=_QUEUE_URL,
        sqs_region_name="us-east-1",
        sqs_aws_access_key_id="unit-access-key",
        sqs_aws_secret_access_key="unit-secret-key",
        sqs_max_message_bytes=sqs_max_message_bytes,
        sqs_aws_use_application_level_encryption=encryption_key is not None,
        sqs_app_encryption_key_b64=None if encryption_key is None else base64.b64encode(encryption_key).decode(),
    )


def _sent_message_body(route: respx.Route) -> str:
    return parse_qs(route.calls.last.request.content.decode())["MessageBody"][0]


@pytest.mark.asyncio
async def test_async_send_message_logs_the_reason_sqs_rejected_the_message(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    sqs_error: Final = (
        '<?xml version="1.0"?>\n<ErrorResponse><Error><Type>Sender</Type><Code>InvalidParameterValue</Code>'
        "<Message>One or more parameters are invalid. Reason: Message must be shorter than 262144 bytes.</Message>"
        "</Error></ErrorResponse>"
    )
    route: Final = respx_mock.post(_QUEUE_URL).mock(return_value=httpx.Response(400, text=sqs_error))
    monkeypatch.setattr("botocore.auth.get_current_datetime", _fixed_signing_time)
    monkeypatch.setattr(asyncio, "create_task", MagicMock())
    logger: Final = SQSLogger(
        sqs_queue_url=_QUEUE_URL,
        sqs_region_name="us-east-1",
        sqs_aws_access_key_id="unit-access-key",
        sqs_aws_secret_access_key="unit-secret-key",
    )
    payload: Final = _chat_payload()

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        await logger.async_send_message(payload)

    assert json.loads(_sent_message_body(route)) == payload
    errors: Final = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "call-with-tools" in errors[0]
    assert "400" in errors[0]
    assert "InvalidParameterValue" in errors[0]
    assert "Message must be shorter than 262144 bytes" in errors[0]
    assert "\n" not in errors[0], "a line-based log pipeline would split the SQS reason from the call id"


@pytest.mark.parametrize(
    ("logged_response", "response", "model_parameters", "model_map_information"),
    [
        (_RESPONSES_API_RESPONSE, _RESPONSES_API_RESPONSE, _PARAMETERS, _MODEL_MAP),
        (_RESPONSES_API_RESPONSE, _RESPONSES_API_RESPONSE_WITHOUT_TOOLS, _PARAMETERS, _MODEL_MAP),
        (_RESPONSES_API_RESPONSE, _RESPONSES_API_RESPONSE_WITHOUT_TOOLS, _PARAMETERS, _MODEL_MAP_WITHOUT_VALUE),
        (
            _RESPONSES_API_RESPONSE,
            _RESPONSES_API_RESPONSE_WITHOUT_TOOLS,
            _PARAMETERS_WITHOUT_TOOLS,
            _MODEL_MAP_WITHOUT_VALUE,
        ),
        (_TEXT_RESPONSE, _TEXT_RESPONSE, _PARAMETERS_WITHOUT_TOOLS, _MODEL_MAP_WITHOUT_VALUE),
    ],
    ids=[
        "fits_untrimmed",
        "fits_without_the_tools_echoed_in_the_response",
        "fits_without_response_tools_and_model_map_value",
        "fits_without_every_droppable_field",
        "text_response_fits_without_model_map_value_and_tools",
    ],
)
@pytest.mark.asyncio
async def test_async_send_message_drops_the_fewest_optional_fields_that_fit_the_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    logged_response: object,
    response: object,
    model_parameters: object,
    model_map_information: object,
) -> None:
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    route: Final = respx_mock.post(_QUEUE_URL).mock(return_value=httpx.Response(200))
    expected: Final = _logged_payload(response, model_parameters, model_map_information)
    logger: Final = _sqs_logger(monkeypatch, sqs_max_message_bytes=_size(expected))

    await logger.async_send_message(_logged_payload(logged_response, _PARAMETERS, _MODEL_MAP))

    assert route.call_count == 1
    assert json.loads(_sent_message_body(route)) == expected


@pytest.mark.asyncio
async def test_async_send_message_warns_which_fields_it_dropped_to_fit(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    respx_mock.post(_QUEUE_URL).mock(return_value=httpx.Response(200))
    payload: Final = _chat_payload()
    logger: Final = _sqs_logger(
        monkeypatch, sqs_max_message_bytes=_size(_chat_payload_without_model_map_value_and_tools())
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        await logger.async_send_message(payload)

    warnings: Final = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "call-with-tools" in warnings[0]
    assert "model_map_information.model_map_value" in warnings[0]
    assert "model_parameters.tools" in warnings[0]
    assert "response.tools" not in warnings[0], "a chat completion response has no tools to drop"
    assert str(_size(payload)) in warnings[0]


@pytest.mark.asyncio
async def test_async_send_message_does_not_send_a_message_that_cannot_fit(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    route: Final = respx_mock.post(_QUEUE_URL).mock(return_value=httpx.Response(200))
    payload: Final = _chat_payload()
    logger: Final = _sqs_logger(
        monkeypatch, sqs_max_message_bytes=_size(_chat_payload_without_model_map_value_and_tools()) - 1
    )

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        await logger.async_send_message(payload)

    assert route.call_count == 0
    errors: Final = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "call-with-tools" in errors[0]
    assert str(_size(payload)) in errors[0]


@pytest.mark.asyncio
async def test_async_send_message_applies_the_size_limit_to_the_encrypted_message(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {})
    route: Final = respx_mock.post(_QUEUE_URL).mock(return_value=httpx.Response(200))
    key: Final = os.urandom(32)
    payload: Final = _chat_payload()
    limit: Final = _size(payload)
    logger: Final = _sqs_logger(
        monkeypatch,
        sqs_max_message_bytes=limit,
        encryption_key=key,
    )

    await logger.async_send_message(payload)

    message_body: Final = _sent_message_body(route)
    assert len(message_body.encode("utf-8")) <= limit
    encrypted: Final = _ENCRYPTED_MESSAGE.validate_json(message_body)["payload"]
    assert isinstance(encrypted, dict), message_body
    decrypted: Final = AppCrypto(key).decrypt_json(encrypted)
    assert decrypted == _chat_payload_without_model_map_value_and_tools()


@pytest.mark.asyncio
async def test_sqs_max_message_bytes_from_proxy_callback_params_is_honored(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    expected: Final = _chat_payload_without_model_map_value()
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {"sqs_max_message_bytes": str(_size(expected))})
    route: Final = respx_mock.post(_QUEUE_URL).mock(return_value=httpx.Response(200))
    logger: Final = _sqs_logger(monkeypatch, sqs_max_message_bytes=_size(_chat_payload()))

    await logger.async_send_message(_chat_payload())

    assert json.loads(_sent_message_body(route)) == expected


@pytest.mark.parametrize("configured", [0, -1, "not-a-number"])
def test_sqs_logger_rejects_an_invalid_max_message_bytes_at_startup(
    monkeypatch: pytest.MonkeyPatch, configured: object
) -> None:
    monkeypatch.setattr("litellm.aws_sqs_callback_params", {"sqs_max_message_bytes": configured})
    with pytest.raises(ValueError, match="sqs_max_message_bytes"):
        _sqs_logger(monkeypatch, sqs_max_message_bytes=262144)


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
