import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm import acompletion, completion
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

FAKE_API_BASE = "https://fake-cloudflare.example.com/client/v4/accounts/fake-acct/ai/run/"
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
        "result": {
            "response": "I am a large language model created to assist you.",
        },
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
