import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm import acompletion, completion
from litellm.llms.cloudflare.chat.transformation import CloudflareChatResponseIterator
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

FAKE_API_BASE = (
    "https://fake-cloudflare.example.com/client/v4/accounts/fake-acct/ai/run/"
)
FAKE_API_KEY = "fake-cf-api-key"


def _make_mock_response(json_data: Dict[str, Any]) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    mock.json.return_value = json_data
    mock.text = json.dumps(json_data)
    return mock


def _chat_response(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "result": result,
        "success": True,
        "errors": [],
        "messages": [],
    }


def _streaming_chunks() -> list[str]:
    return [
        json.dumps({"response": "I am"}),
        json.dumps({"response": " a language"}),
        json.dumps({"response": " model."}),
    ]


def _streaming_chunks_response_text() -> list[str]:
    return [
        json.dumps({"response_text": "I am"}),
        json.dumps({"response_text": " a language"}),
        json.dumps({"response_text": " model."}),
    ]


def _streaming_chunks_openai_compatible() -> list[str]:
    return [
        json.dumps({"choices": [{"delta": {"content": "I am"}}]}),
        json.dumps({"choices": [{"delta": {"content": " a language"}}]}),
        json.dumps({"choices": [{"delta": {"content": " model."}}]}),
    ]


@pytest.mark.parametrize(
    ("result", "expected_content"),
    [
        (
            {"response": "I am a large language model created to assist you."},
            "I am a large language model created to assist you.",
        ),
        ({"response_text": "I am a language model."}, "I am a language model."),
        (
            {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]},
            "Hello",
        ),
        ({"response": ""}, ""),
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
def test_completion_cloudflare(sync_mode, result, expected_content):
    messages = [{"role": "user", "content": "what llm are you"}]
    mock_resp = _make_mock_response(_chat_response(result=result))

    if sync_mode:
        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                messages=messages,
                max_tokens=15,
                api_base=FAKE_API_BASE,
                api_key=FAKE_API_KEY,
            )
            mock_post.assert_called_once()
    else:
        with patch.object(
            AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            response = asyncio.run(
                acompletion(
                    model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                    messages=messages,
                    max_tokens=15,
                    api_base=FAKE_API_BASE,
                    api_key=FAKE_API_KEY,
                )
            )
            mock_post.assert_called_once()

    assert response is not None
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.content == expected_content


@pytest.mark.parametrize("sync_mode", [True, False])
def test_completion_cloudflare_stream(sync_mode):
    messages = [{"role": "user", "content": "what llm are you"}]
    raw_chunks = _streaming_chunks()

    if sync_mode:

        def _iter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = _iter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                messages=messages,
                max_tokens=15,
                stream=True,
                api_base=FAKE_API_BASE,
                api_key=FAKE_API_KEY,
            )
            chunks_received = list(response)
            mock_post.assert_called_once()
    else:

        async def _aiter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.aiter_lines.return_value = _aiter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def _run():
            with patch.object(
                AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                resp = await acompletion(
                    model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                    messages=messages,
                    max_tokens=15,
                    stream=True,
                    api_base=FAKE_API_BASE,
                    api_key=FAKE_API_KEY,
                )
                received = []
                async for chunk in resp:
                    received.append(chunk)
                mock_post.assert_called_once()
                return received

        chunks_received = asyncio.run(_run())

    assert len(chunks_received) > 0
    content = "".join(
        c.choices[0].delta.content
        for c in chunks_received
        if c.choices[0].delta.content
    )
    assert "language" in content.lower()


@pytest.mark.parametrize("sync_mode", [True, False])
def test_completion_cloudflare_stream_response_text(sync_mode):
    """Newer Cloudflare Workers AI models (e.g. Nemotron) emit `response_text`
    instead of `response` in streamed chunks. The iterator must surface that
    text so streaming output is not silently empty.
    """
    messages = [{"role": "user", "content": "what llm are you"}]
    raw_chunks = _streaming_chunks_response_text()

    if sync_mode:

        def _iter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = _iter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="cloudflare/@cf/nvidia/nemotron-mini-4b-instruct",
                messages=messages,
                max_tokens=15,
                stream=True,
                api_base=FAKE_API_BASE,
                api_key=FAKE_API_KEY,
            )
            chunks_received = list(response)
            mock_post.assert_called_once()
    else:

        async def _aiter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.aiter_lines.return_value = _aiter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def _run():
            with patch.object(
                AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                resp = await acompletion(
                    model="cloudflare/@cf/nvidia/nemotron-mini-4b-instruct",
                    messages=messages,
                    max_tokens=15,
                    stream=True,
                    api_base=FAKE_API_BASE,
                    api_key=FAKE_API_KEY,
                )
                received = []
                async for chunk in resp:
                    received.append(chunk)
                mock_post.assert_called_once()
                return received

        chunks_received = asyncio.run(_run())

    assert len(chunks_received) > 0
    content = "".join(
        c.choices[0].delta.content
        for c in chunks_received
        if c.choices[0].delta.content
    )
    assert "language" in content.lower()


@pytest.mark.parametrize("sync_mode", [True, False])
def test_completion_cloudflare_stream_openai_compatible(sync_mode):
    messages = [{"role": "user", "content": "what llm are you"}]
    raw_chunks = _streaming_chunks_openai_compatible()

    if sync_mode:

        def _iter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = _iter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="cloudflare/@cf/meta/llama-3.1-8b-instruct",
                messages=messages,
                max_tokens=15,
                stream=True,
                api_base=FAKE_API_BASE,
                api_key=FAKE_API_KEY,
            )
            chunks_received = list(response)
            mock_post.assert_called_once()
    else:

        async def _aiter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.aiter_lines.return_value = _aiter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def _run():
            with patch.object(
                AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_post:
                resp = await acompletion(
                    model="cloudflare/@cf/meta/llama-3.1-8b-instruct",
                    messages=messages,
                    max_tokens=15,
                    stream=True,
                    api_base=FAKE_API_BASE,
                    api_key=FAKE_API_KEY,
                )
                received = []
                async for chunk in resp:
                    received.append(chunk)
                mock_post.assert_called_once()
                return received

        chunks_received = asyncio.run(_run())

    assert len(chunks_received) > 0
    content = "".join(
        c.choices[0].delta.content
        for c in chunks_received
        if c.choices[0].delta.content
    )
    assert "language" in content.lower()


@pytest.mark.parametrize(
    ("raw_chunk", "expected_text"),
    [
        ({"response": "hello"}, "hello"),
        ({"response_text": "hello"}, "hello"),
        ({"choices": [{"delta": {"content": "hello"}}]}, "hello"),
    ],
)
def test_cloudflare_streaming_chunk_parser_content(raw_chunk, expected_text):
    iterator = CloudflareChatResponseIterator(
        streaming_response=[],
        sync_stream=True,
    )

    parsed_chunk = iterator.chunk_parser(raw_chunk)

    assert parsed_chunk["text"] == expected_text
    assert parsed_chunk["is_finished"] is False


def test_cloudflare_streaming_chunk_parser_finish_reason():
    iterator = CloudflareChatResponseIterator(
        streaming_response=[],
        sync_stream=True,
    )

    parsed_chunk = iterator.chunk_parser(
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    )

    assert parsed_chunk["text"] == ""
    assert parsed_chunk["is_finished"] is True
    assert parsed_chunk["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "raw_chunk",
    [
        {"response": "last token", "finish_reason": "stop"},
        {"response_text": "last token", "finish_reason": "stop"},
    ],
)
def test_cloudflare_legacy_streaming_chunk_parser_finish_reason(raw_chunk):
    iterator = CloudflareChatResponseIterator(
        streaming_response=[],
        sync_stream=True,
    )

    parsed_chunk = iterator.chunk_parser(raw_chunk)

    assert parsed_chunk["text"] == "last token"
    assert parsed_chunk["is_finished"] is True
    assert parsed_chunk["finish_reason"] == "stop"
