"""
Simple A2A agent tests - non-streaming and streaming.

These tests use a mocked A2A client to avoid network/env dependencies.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest


class MockA2AResponse:
    def __init__(self, text: str):
        self._payload = {
            "id": str(uuid4()),
            "jsonrpc": "2.0",
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": text}],
                    "messageId": uuid4().hex,
                }
            },
        }

    def model_dump(self, mode="json", exclude_none=True):
        return self._payload


class MockA2AStreamingChunk(MockA2AResponse):
    def __init__(self, text: str, state: str):
        super().__init__(text=text)
        self._payload["result"]["status"] = {"state": state}


class MockA2AClient:
    def __init__(self):
        self._litellm_agent_card = SimpleNamespace(
            name="mock-agent", url="http://mock-agent.local"
        )

    async def send_message(self, request):
        return MockA2AResponse(text="hello")

    def send_message_streaming(self, request):
        async def _stream():
            yield MockA2AStreamingChunk(text="hel", state="in_progress")
            yield MockA2AStreamingChunk(text="hello", state="completed")

        return _stream()


@pytest.fixture
def mock_a2a_client(monkeypatch):
    import litellm.a2a_protocol.main as a2a_main

    async def _fake_create_a2a_client(base_url, timeout=60.0, extra_headers=None):
        return MockA2AClient()

    monkeypatch.setattr(a2a_main, "create_a2a_client", _fake_create_a2a_client)


@pytest.mark.asyncio
async def test_a2a_non_streaming(mock_a2a_client):
    """Test non-streaming A2A request."""
    from a2a.types import MessageSendParams, SendMessageRequest
    from litellm.a2a_protocol import asend_message

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Say hello in one word"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    response = await asend_message(
        request=request,
        api_base="http://mock",
    )

    assert response is not None
    print(f"\nNon-streaming response: {response}")


@pytest.mark.asyncio
async def test_a2a_streaming(mock_a2a_client):
    """Test streaming A2A request."""
    from a2a.types import MessageSendParams, SendStreamingMessageRequest
    from litellm.a2a_protocol import asend_message_streaming

    request = SendStreamingMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Say hello in one word"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    chunks = []
    async for chunk in asend_message_streaming(
        request=request,
        api_base="http://mock",
    ):
        chunks.append(chunk)
        print(f"\nStreaming chunk: {chunk}")

    assert len(chunks) > 0, "Should receive at least one chunk"
    print(f"\nTotal chunks received: {len(chunks)}")
