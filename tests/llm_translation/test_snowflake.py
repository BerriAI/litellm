import asyncio
import json
import httpx
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm import completion, acompletion
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

FAKE_API_BASE = "https://fake-snowflake.example.com/api/v2/cortex/inference:chat"


def _make_mock_response(json_data: Dict[str, Any]) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    mock.json.return_value = json_data
    mock.text = json.dumps(json_data)
    return mock


def _chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-snowflake-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "mistral-7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The sky above is painted blue,\nWith clouds of white and morning dew.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 30,
            "total_tokens": 40,
        },
    }


def _streaming_chunks() -> List[str]:
    base = {
        "id": "chatcmpl-snowflake-stream-123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "mistral-7b",
    }
    deltas = [
        {"role": "assistant", "content": "The"},
        {"content": " sky"},
        {"content": " is blue"},
    ]
    chunks = []
    for i, delta in enumerate(deltas):
        finish = "stop" if i == len(deltas) - 1 else None
        chunks.append(
            json.dumps(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish}
                    ],
                }
            )
        )
    return chunks


@pytest.mark.parametrize("sync_mode", [True, False])
def test_chat_completion_snowflake(sync_mode):
    messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]
    mock_resp = _make_mock_response(_chat_response())

    if sync_mode:
        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                api_base=FAKE_API_BASE,
            )
            mock_post.assert_called_once()
    else:
        with patch.object(
            AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            response = asyncio.run(
                acompletion(
                    model="snowflake/mistral-7b",
                    messages=messages,
                    api_base=FAKE_API_BASE,
                )
            )
            mock_post.assert_called_once()

    assert response is not None
    assert response.choices[0].message.content is not None
    assert "sky" in response.choices[0].message.content.lower()
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 30


@pytest.mark.parametrize("sync_mode", [True, False])
def test_chat_completion_snowflake_stream(sync_mode):
    messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]
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
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
                api_base=FAKE_API_BASE,
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
                    model="snowflake/mistral-7b",
                    messages=messages,
                    max_tokens=100,
                    stream=True,
                    api_base=FAKE_API_BASE,
                )
                received = []
                async for chunk in resp:
                    received.append(chunk)
                mock_post.assert_called_once()
                return received

        chunks_received = asyncio.run(_run())

    assert len(chunks_received) > 0
    content = "".join(
        c.choices[0].delta.content for c in chunks_received if c.choices[0].delta.content
    )
    assert "sky" in content.lower()
