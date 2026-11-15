"""
Test for Groq streaming ASCII encoding issue fix.

This test verifies that the OpenAI-like handler correctly handles
UTF-8 encoded content in streaming responses, specifically fixing
the ASCII encoding error described in issue #12660.
"""

import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai_like.chat.handler import make_call, make_sync_call

_UNICODE_TEXTS: Final = (
    "The symbol µ represents micro",
    "Testing © character",
    "Testing ™ character",
    "Testing € character",
    "Testing 北京 character",
    "Testing 🚀 character",
    "Testing Ñoño character",
)


def _utf8_stream_transport(text: str) -> httpx.MockTransport:
    chunk: Final = {"choices": [{"index": 0, "delta": {"content": text}}]}
    body: Final = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
    return httpx.MockTransport(lambda request: httpx.Response(200, stream=httpx.ByteStream(body)))


def _call_args() -> dict[str, object]:
    return {
        "api_base": "https://test.com/v1/chat/completions",
        "headers": {"Authorization": "Bearer test"},
        "data": '{"model": "test", "messages": []}',
        "model": "test-model",
        "messages": [],
        "logging_obj": Mock(),
    }


@pytest.mark.parametrize("text", _UNICODE_TEXTS)
def test_utf8_streaming_sync(text: str) -> None:
    completion_stream: Final = make_sync_call(
        client=HTTPHandler(client=httpx.Client(transport=_utf8_stream_transport(text))), **_call_args()
    )

    assert "".join(chunk["text"] for chunk in completion_stream) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("text", _UNICODE_TEXTS)
async def test_utf8_streaming_async(text: str) -> None:
    completion_stream: Final = await make_call(
        client=AsyncHTTPHandler(transport=_utf8_stream_transport(text)), **_call_args()
    )

    assert "".join([chunk["text"] async for chunk in completion_stream]) == text
