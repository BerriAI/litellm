from collections.abc import AsyncIterator
from typing import Final, Literal, assert_never, cast  # noqa: TID251  # narrows the native stream after isinstance

import pytest
from pydantic import TypeAdapter

from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.response_metadata import mark_rust_response
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES_EVENTS,
    MESSAGES_RESPONSE,
    OCR_DOCUMENT,
    OCR_RESPONSE,  # pyright: ignore[reportUnknownVariableType]  # shared fixture is an untyped dict literal
)

pytestmark = pytest.mark.requires_rust_extension
_HEADERS: Final = TypeAdapter(dict[str, str])
_ADDITIONAL: Final = TypeAdapter(dict[str, object])
RouteName = Literal["messages", "chat", "responses", "ocr", "transcription"]


def call_sync(route: RouteName, call: NativeCall) -> object:
    from litellm.rust_bridge import _native

    match route:
        case "messages":
            return _native.messages(call)
        case "chat":
            return _native.completion(call)
        case "responses":
            return _native.responses(call)
        case "ocr":
            return _native.ocr(call)
        case "transcription":
            return _native.transcription(call)
        case _:
            assert_never(route)


async def call_async(route: RouteName, call: NativeCall) -> object:
    from litellm.rust_bridge import _native

    match route:
        case "messages":
            return await _native.amessages(call)
        case "chat":
            return await _native.acompletion(call)
        case "responses":
            return await _native.aresponses(call)
        case "ocr":
            return await _native.aocr(call)
        case "transcription":
            return await _native.atranscription(call)
        case _:
            assert_never(route)


@pytest.mark.parametrize(
    ("route", "model", "fields", "body"),
    (
        (
            "messages",
            "anthropic/test-model",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            MESSAGES_RESPONSE,
        ),
        (
            "chat",
            "anthropic/test-model",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            MESSAGES_RESPONSE,
        ),
        (
            "responses",
            "openai/test-model",
            {"input": "hi"},
            {"id": "resp_test", "model": "test-model", "created_at": 0, "output": []},
        ),
        ("ocr", "mistral/test-model", {"document": OCR_DOCUMENT}, OCR_RESPONSE),
        (
            "transcription",
            "bedrock/test-model",
            {
                "audio": {"data": "YWJj", "format": "wav"},
                "optional_params": {
                    "aws_access_key_id": "test",
                    "aws_secret_access_key": "test",
                    "aws_region_name": "us-east-1",
                },
            },
            {"output": {"message": {"content": [{"text": "hello"}]}}},
        ),
    ),
)
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.asyncio
async def test_native_response_headers_reach_python_metadata(
    recording_server: RecordingServer,
    route: RouteName,
    model: str,
    fields: dict[str, object],
    body: object,
    asynchronous: bool,
) -> None:
    recording_server.enqueue(
        ResponseSpec(
            body=body,
            headers={
                "x-provider-trace": "trace",
                "retry-after": "7",
                "connection": "x-private",
                "x-private": "hop",
                "set-cookie": "session=private",
                "x-litellm-rust": "false",
            },
        )
    )
    supplied: Final = {
        "model": model,
        "api_key": "test-key",
        "api_base": recording_server.base_url,
        "custom_llm_provider": None,
        "timeout": None,
        "extra_headers": None,
        **fields,
    }
    call: Final = NativeCall(args=(), kwargs=supplied, base=supplied)
    response: Final = await call_async(route, call) if asynchronous else call_sync(route, call)
    hidden: Final = get_hidden_params_dict(mark_rust_response(response))
    headers: Final = _HEADERS.validate_python(hidden["headers"])
    assert headers["x-provider-trace"] == "trace"
    assert headers["retry-after"] == "7"
    additional: Final = _ADDITIONAL.validate_python(hidden["additional_headers"])
    assert additional["llm_provider-x-provider-trace"] == "trace"
    assert additional["llm_provider-retry-after"] == "7"
    assert additional["x-litellm-rust"] == "true"
    for forbidden in ("x-private", "connection", "set-cookie", "content-length"):
        assert forbidden not in headers
        assert f"llm_provider-{forbidden}" not in additional


@pytest.mark.asyncio
async def test_native_messages_stream_headers_use_the_same_python_metadata(recording_server: RecordingServer) -> None:
    recording_server.enqueue(
        ResponseSpec(
            body=None,
            events=MESSAGES_EVENTS,
            headers={
                "x-provider-trace": "stream",
                "connection": "x-private",
                "x-private": "hop",
                "x-litellm-rust": "false",
            },
        )
    )
    supplied: Final[dict[str, object]] = {
        "model": "anthropic/test-model",
        "api_key": "test-key",
        "api_base": recording_server.base_url,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "stream": True,
    }
    response: Final = await call_async("messages", NativeCall(args=(), kwargs=supplied, base=supplied))
    assert isinstance(response, AsyncIterator)
    stream: Final = cast(
        AsyncIterator[bytes], response
    )  # cast-ok: isinstance above only narrows to AsyncIterator[Unknown]
    additional: Final = _ADDITIONAL.validate_python(
        get_hidden_params_dict(mark_rust_response(response))["additional_headers"]
    )
    assert additional["llm_provider-x-provider-trace"] == "stream"
    assert additional["x-litellm-rust"] == "true"
    assert "llm_provider-x-private" not in additional
    assert b"message_stop" in b"".join([chunk async for chunk in stream])
