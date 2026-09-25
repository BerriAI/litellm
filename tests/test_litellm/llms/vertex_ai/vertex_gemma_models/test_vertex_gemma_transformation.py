import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.main import vertex_gemma_chat_completion
from litellm.types.llms.openai import OutputTextDeltaEvent, ResponseCompletedEvent, ResponsesAPIStreamingResponse

_VERTEX_URL = "https://example.invalid/v1/projects/test/locations/us-central1/endpoints/test:predict"
_MESSAGES = [{"role": "user", "content": "Reply exactly READY"}]
_FAKE_CREDENTIALS = "gemma-test-credentials"


def _vertex_response():
    return {
        "predictions": {
            "id": "chatcmpl-stream-test",
            "created": 1759863903,
            "model": "google/gemma-3-12b-it",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "READY"}}],
            "usage": {"prompt_tokens": 14, "completion_tokens": 1, "total_tokens": 15},
        }
    }


@pytest.fixture(autouse=True)
def _cached_access_token():
    """Serve a fake token from the handler's credential cache so no auth round-trip runs."""
    cache = vertex_gemma_chat_completion._credentials_project_mapping
    key = (_FAKE_CREDENTIALS, "test")
    cache[key] = (SimpleNamespace(token="fake-token", expired=False), "test")
    yield
    cache.pop(key, None)


def test_sync_gemma_stream():
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_vertex_response())

    stream = litellm.completion(
        model="vertex_ai/gemma/test-model",
        messages=_MESSAGES,
        stream=True,
        api_base=_VERTEX_URL,
        vertex_project="test",
        vertex_location="us-central1",
        vertex_credentials=_FAKE_CREDENTIALS,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )

    assert isinstance(stream, CustomStreamWrapper)
    chunks = list(stream)

    assert "stream" not in captured["body"]["instances"][0]
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "READY"
    assert chunks[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_async_gemma_responses_stream():
    captured: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_vertex_response())

    response = await litellm.aresponses(
        model="vertex_ai/gemma/test-model",
        input="Reply exactly READY",
        stream=True,
        api_base=_VERTEX_URL,
        vertex_project="test",
        vertex_location="us-central1",
        vertex_credentials=_FAKE_CREDENTIALS,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    events = [event async for event in cast(AsyncIterator[ResponsesAPIStreamingResponse], response)]

    assert "stream" not in captured["body"]["instances"][0]
    assert "READY" in "".join(event.delta for event in events if isinstance(event, OutputTextDeltaEvent))
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert events[-1].response.usage is not None
    assert events[-1].response.usage.total_tokens == 15
