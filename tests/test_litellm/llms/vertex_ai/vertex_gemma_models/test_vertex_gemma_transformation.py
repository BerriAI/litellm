from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.llms.openai import OutputTextDeltaEvent, ResponseCompletedEvent, ResponsesAPIStreamingResponse

_VERTEX_URL = "https://example.invalid/v1/projects/test/locations/us-central1/endpoints/test:predict"
_MESSAGES = [{"role": "user", "content": "Reply exactly READY"}]


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


def test_sync_gemma_stream():
    reply = Mock(status_code=200)
    reply.json.return_value = _vertex_response()
    with (
        patch(
            "litellm.llms.vertex_ai.vertex_gemma_models.transformation.VertexGemmaConfig._sync_post", return_value=reply
        ),
        patch(
            "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
            return_value=("fake-token", "test"),
        ),
    ):
        stream = litellm.completion(
            model="vertex_ai/gemma/test-model",
            messages=_MESSAGES,
            stream=True,
            api_base=_VERTEX_URL,
            vertex_project="test",
            vertex_location="us-central1",
        )
        assert isinstance(stream, CustomStreamWrapper)
        chunks = list(stream)

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "READY"
    assert chunks[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_async_gemma_responses_stream():
    litellm.in_memory_llm_clients_cache.flush_cache()
    reply = Mock(status_code=200)
    reply.json.return_value = _vertex_response()
    client = Mock(post=AsyncMock(return_value=reply))
    with (
        patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client),
        patch(
            "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
            return_value=("fake-token", "test"),
        ),
    ):
        response = await litellm.aresponses(
            model="vertex_ai/gemma/test-model",
            input="Reply exactly READY",
            stream=True,
            api_base=_VERTEX_URL,
            vertex_project="test",
            vertex_location="us-central1",
        )
        events = [event async for event in cast(AsyncIterator[ResponsesAPIStreamingResponse], response)]

    assert "stream" not in client.post.call_args.kwargs["json"]["instances"][0]
    assert "READY" in "".join(event.delta for event in events if isinstance(event, OutputTextDeltaEvent))
    assert isinstance(events[-1], ResponseCompletedEvent)
    assert events[-1].response.usage is not None
    assert events[-1].response.usage.total_tokens == 15
