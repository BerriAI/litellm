import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm import acompletion, completion
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

FAKE_API_BASE = "https://fake-cloudflare.example.com/client/v4/accounts/fake-acct/ai/v1"
FAKE_API_KEY = "fake-cf-api-key"


def _make_mock_response(json_data: Dict[str, Any]) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    mock.json.return_value = json_data
    mock.text = json.dumps(json_data)
    return mock


def _chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-cf",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I am a large language model created to assist you.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 11, "total_tokens": 19},
    }


def _tool_call_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-cf-tools",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "New York"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
    }


def _streaming_chunks() -> list[str]:
    base = {
        "id": "chatcmpl-cf",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
    }
    return [
        json.dumps({**base, "choices": [{"index": 0, "delta": {"content": "I am"}}]}),
        json.dumps(
            {**base, "choices": [{"index": 0, "delta": {"content": " a language"}}]}
        ),
        json.dumps(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " model."},
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
def test_completion_cloudflare(sync_mode):
    messages = [{"role": "user", "content": "what llm are you"}]
    mock_resp = _make_mock_response(_chat_response())

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
    assert "language model" in response.choices[0].message.content.lower()

    called_url = mock_post.call_args.kwargs.get("url") or mock_post.call_args.args[0]
    assert called_url.endswith("/ai/v1/chat/completions")
    assert "/ai/run/" not in called_url


def test_completion_cloudflare_tool_calls_sent_to_openai_endpoint():
    messages = [{"role": "user", "content": "weather in New York?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    mock_resp = _make_mock_response(_tool_call_response())

    with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
        response = completion(
            model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            api_base=FAKE_API_BASE,
            api_key=FAKE_API_KEY,
        )
        mock_post.assert_called_once()

    sent_body = json.loads(mock_post.call_args.kwargs["data"])
    assert sent_body["tools"] == tools
    assert sent_body["tool_choice"] == "auto"

    assert response.choices[0].finish_reason == "tool_calls"
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


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
