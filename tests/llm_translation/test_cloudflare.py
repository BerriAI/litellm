import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm import acompletion, completion
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

FAKE_API_BASE = "https://fake-cloudflare.example.com/client/v4/accounts/fake-acct/ai/v1"
FAKE_API_KEY = "fake-cf-api-key"


def _streaming_chunks() -> list[str]:
    base = {
        "id": "chatcmpl-cf",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
    }
    return [
        json.dumps({**base, "choices": [{"index": 0, "delta": {"content": "I am"}}]}),
        json.dumps({**base, "choices": [{"index": 0, "delta": {"content": " a language"}}]}),
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


@pytest.mark.parametrize("sync_mode", [False])
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
            with patch.object(AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
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
    content = "".join(c.choices[0].delta.content for c in chunks_received if c.choices[0].delta.content)
    assert "language" in content.lower()
