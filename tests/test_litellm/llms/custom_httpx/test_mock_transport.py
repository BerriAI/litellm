"""
Tests for MockOpenAITransport — verifies that the mock transport produces
responses parseable by the OpenAI SDK for both streaming and non-streaming paths.
"""

import json

import httpx
import pytest

from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


class TestNonStreaming:
    def test_sync_returns_valid_chat_completion(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
        )
        response = transport.handle_request(request)
        assert response.status_code == 200

        body = json.loads(response.content)
        assert body["object"] == "chat.completion"
        assert body["model"] == "gpt-4o"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "usage" in body

    @pytest.mark.asyncio
    async def test_async_returns_valid_chat_completion(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}),
        )
        response = await transport.handle_async_request(request)
        assert response.status_code == 200

        body = json.loads(response.content)
        assert body["object"] == "chat.completion"
        assert body["model"] == "gpt-4o-mini"

    def test_model_echoed_from_request(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "my-custom-model", "messages": []}),
        )
        response = transport.handle_request(request)
        body = json.loads(response.content)
        assert body["model"] == "my-custom-model"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_sync_streaming_returns_sse_events(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "gpt-4o", "stream": True, "messages": []}),
        )
        response = transport.handle_request(request)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        chunks = list(response.stream)
        # Should have: content chunk, finish chunk, [DONE]
        assert len(chunks) == 3
        assert chunks[-1] == b"data: [DONE]\n\n"

        # Parse the first chunk
        first_line = chunks[0].decode()
        assert first_line.startswith("data: ")
        data = json.loads(first_line[len("data: "):].strip())
        assert data["object"] == "chat.completion.chunk"
        assert data["model"] == "gpt-4o"
        assert data["choices"][0]["delta"]["content"] == "Mock response"

    @pytest.mark.asyncio
    async def test_async_streaming_returns_sse_events(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "gpt-4o", "stream": True, "messages": []}),
        )
        response = await transport.handle_async_request(request)
        assert response.status_code == 200

        chunks = []
        async for chunk in response.stream:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[-1] == b"data: [DONE]\n\n"

        # Parse finish chunk
        finish_line = chunks[1].decode()
        data = json.loads(finish_line[len("data: "):].strip())
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_streaming_model_echoed(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "custom-stream", "stream": True, "messages": []}),
        )
        response = transport.handle_request(request)
        first_chunk = next(iter(response.stream))
        data = json.loads(first_chunk.decode()[len("data: "):].strip())
        assert data["model"] == "custom-stream"


# ---------------------------------------------------------------------------
# Integration with httpx client
# ---------------------------------------------------------------------------


class TestHttpxClientIntegration:
    def test_sync_client_get(self):
        """Verify the transport works when wired into an httpx.Client."""
        client = httpx.Client(transport=MockOpenAITransport())
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        client.close()

    @pytest.mark.asyncio
    async def test_async_client_get(self):
        """Verify the transport works when wired into an httpx.AsyncClient."""
        client = httpx.AsyncClient(transport=MockOpenAITransport())
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        await client.aclose()
