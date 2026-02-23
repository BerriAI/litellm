"""
Mock httpx transport that returns valid OpenAI ChatCompletion responses.

Activated via `litellm_settings: { network_mock: true }`.
Intercepts at the httpx transport layer — the lowest point before bytes hit the wire —
so the full proxy -> router -> OpenAI SDK -> httpx path is exercised.
"""

import json
import time
import uuid
from typing import Tuple

import httpx


# ---------------------------------------------------------------------------
# Pre-built response templates
# ---------------------------------------------------------------------------

def _mock_id() -> str:
    return f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"


def _chat_completion_json(model: str) -> dict:
    """Return a minimal valid ChatCompletion object."""
    return {
        "id": _mock_id(),
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


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

_JSON_HEADERS = {
    "content-type": "application/json",
}


class MockOpenAITransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    """
    httpx transport that returns canned OpenAI ChatCompletion responses.

    Supports both async (AsyncOpenAI) and sync (OpenAI) SDK paths.
    """

    @staticmethod
    def _parse_request(request: httpx.Request) -> Tuple[str, bool]:
        """Extract model from the request body."""
        try:
            body = json.loads(request.content)
        except (json.JSONDecodeError, ValueError):
            return ("mock-model", False)
        model = body.get("model", "mock-model")
        return (model, False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        model, _ = self._parse_request(request)
        body = json.dumps(_chat_completion_json(model)).encode()
        return httpx.Response(
            status_code=200,
            headers=_JSON_HEADERS,
            content=body,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        model, _ = self._parse_request(request)
        body = json.dumps(_chat_completion_json(model)).encode()
        return httpx.Response(
            status_code=200,
            headers=_JSON_HEADERS,
            content=body,
        )
