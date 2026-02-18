"""
Mock httpx transport that returns valid OpenAI ChatCompletion responses.

Activated via `litellm_settings: { network_mock: true }`.
Intercepts at the httpx transport layer — the lowest point before bytes hit the wire —
so the full proxy -> router -> OpenAI SDK -> httpx path is exercised.
"""

import json
import time
from typing import Iterator, List

import httpx


# ---------------------------------------------------------------------------
# Pre-built response templates
# ---------------------------------------------------------------------------

def _chat_completion_json(model: str) -> dict:
    """Return a minimal valid ChatCompletion object."""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Mock response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _streaming_sse_payloads(model: str) -> List[bytes]:
    """Pre-build the SSE byte payloads for a streaming response."""
    chunk = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Mock response"},
                "finish_reason": None,
            }
        ],
    }
    done_chunk = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return [
        b"data: " + json.dumps(chunk).encode() + b"\n\n",
        b"data: " + json.dumps(done_chunk).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]


# ---------------------------------------------------------------------------
# Byte-stream wrappers
# ---------------------------------------------------------------------------

class MockSSEAsyncStream(httpx.AsyncByteStream):
    """Async byte stream that yields pre-built SSE payloads."""

    def __init__(self, payloads: List[bytes]) -> None:
        self._payloads = payloads

    async def __aiter__(self):  # type: ignore[override]
        for payload in self._payloads:
            yield payload


class MockSSESyncStream(httpx.SyncByteStream):
    """Sync byte stream that yields pre-built SSE payloads."""

    def __init__(self, payloads: List[bytes]) -> None:
        self._payloads = payloads

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._payloads)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

_STREAM_HEADERS = {
    "content-type": "text/event-stream",
}

_JSON_HEADERS = {
    "content-type": "application/json",
}


class MockOpenAITransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    """
    httpx transport that returns canned OpenAI ChatCompletion responses.

    Supports both async (AsyncOpenAI) and sync (OpenAI) SDK paths.
    """

    @staticmethod
    def _parse_request(request: httpx.Request) -> tuple:
        """Extract (model, stream) from the request body."""
        body = json.loads(request.content)
        model = body.get("model", "mock-model")
        stream = body.get("stream", False)
        return model, stream

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        model, stream = self._parse_request(request)
        if stream:
            payloads = _streaming_sse_payloads(model)
            return httpx.Response(
                status_code=200,
                headers=_STREAM_HEADERS,
                stream=MockSSEAsyncStream(payloads),
            )
        body = json.dumps(_chat_completion_json(model)).encode()
        return httpx.Response(
            status_code=200,
            headers=_JSON_HEADERS,
            content=body,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        model, stream = self._parse_request(request)
        if stream:
            payloads = _streaming_sse_payloads(model)
            return httpx.Response(
                status_code=200,
                headers=_STREAM_HEADERS,
                stream=MockSSESyncStream(payloads),
            )
        body = json.dumps(_chat_completion_json(model)).encode()
        return httpx.Response(
            status_code=200,
            headers=_JSON_HEADERS,
            content=body,
        )
