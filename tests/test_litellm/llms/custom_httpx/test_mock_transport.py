"""
Tests for MockOpenAITransport — verifies that the mock transport produces
responses parseable by the OpenAI SDK.
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

    def test_unique_ids_per_response(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            content=json.dumps({"model": "gpt-4o", "messages": []}),
        )
        r1 = json.loads(transport.handle_request(request).content)
        r2 = json.loads(transport.handle_request(request).content)
        assert r1["id"] != r2["id"]

    def test_empty_body_does_not_crash(self):
        transport = MockOpenAITransport()
        request = httpx.Request(
            method="GET",
            url="https://api.openai.com/v1/models",
            content=b"",
        )
        response = transport.handle_request(request)
        assert response.status_code == 200
        body = json.loads(response.content)
        assert body["model"] == "mock-model"


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
