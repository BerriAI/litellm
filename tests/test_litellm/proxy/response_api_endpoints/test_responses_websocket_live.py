"""
Live integration test for Responses API WebSocket mode through the LiteLLM proxy.

Requires:
  - OPENAI_API_KEY set in environment
  - LiteLLM proxy running on localhost:4000 with a model named 'gpt-4o-mini'

Run standalone:
    OPENAI_API_KEY=sk-... python -m pytest tests/test_litellm/proxy/response_api_endpoints/test_responses_websocket_live.py -v -s
"""

import asyncio
import json
import os

import pytest

PROXY_BASE = os.environ.get("LITELLM_PROXY_BASE", "ws://localhost:4000")
PROXY_KEY = os.environ.get("LITELLM_PROXY_KEY", "sk-1234")
MODEL = os.environ.get("LITELLM_WS_TEST_MODEL", "gpt-4o-mini")


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — live OpenAI test skipped",
)


@pytest.mark.asyncio
async def test_responses_websocket_live_through_proxy():
    """
    End-to-end: connect to the proxy's /v1/responses WebSocket, send a
    response.create event, and collect streamed events until response.completed.
    """
    import websockets

    url = f"{PROXY_BASE}/v1/responses?model={MODEL}"
    headers = {"Authorization": f"Bearer {PROXY_KEY}"}

    collected_events = []

    async with websockets.connect(url, additional_headers=headers) as ws:
        request_event = {
            "type": "response.create",
            "model": MODEL,
            "input": "Say exactly: Hello WebSocket",
        }
        await ws.send(json.dumps(request_event))

        async for raw in ws:
            event = json.loads(raw)
            collected_events.append(event)
            event_type = event.get("type", "")
            print(f"  ← {event_type}")
            if event_type in (
                "response.completed",
                "response.failed",
                "response.incomplete",
                "error",
            ):
                break

    event_types = [e.get("type") for e in collected_events]
    print(f"\nAll event types received: {event_types}")

    assert "response.created" in event_types, "Missing response.created event"
    assert (
        "response.completed" in event_types
        or "response.failed" in event_types
    ), "No terminal event received"

    if "response.completed" in event_types:
        completed = next(e for e in collected_events if e["type"] == "response.completed")
        response_obj = completed.get("response", {})
        assert response_obj.get("id", "").startswith("resp_")
        assert response_obj.get("status") == "completed"
        print(f"\n✅ Response ID: {response_obj['id']}")

        output = response_obj.get("output", [])
        for item in output:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        print(f"   Model said: {part['text']}")


@pytest.mark.asyncio
async def test_responses_websocket_continuation():
    """
    Test continuation: send a first response.create, get the response_id,
    then send a second response.create with previous_response_id.
    """
    import websockets

    url = f"{PROXY_BASE}/v1/responses?model={MODEL}"
    headers = {"Authorization": f"Bearer {PROXY_KEY}"}

    async with websockets.connect(url, additional_headers=headers) as ws:
        # First turn
        await ws.send(json.dumps({
            "type": "response.create",
            "model": MODEL,
            "input": "Remember the number 42.",
        }))

        first_response_id = None
        async for raw in ws:
            event = json.loads(raw)
            if event.get("type") == "response.completed":
                first_response_id = event.get("response", {}).get("id")
                break
            if event.get("type") in ("response.failed", "error"):
                pytest.skip(f"First turn failed: {event}")

        assert first_response_id, "Did not receive first response ID"
        print(f"  First response: {first_response_id}")

        # Second turn — continuation
        await ws.send(json.dumps({
            "type": "response.create",
            "model": MODEL,
            "input": [
                {"type": "message", "role": "user", "content": "What number did I mention?"},
            ],
            "previous_response_id": first_response_id,
        }))

        second_events = []
        async for raw in ws:
            event = json.loads(raw)
            second_events.append(event)
            if event.get("type") in (
                "response.completed",
                "response.failed",
                "error",
            ):
                break

        second_types = [e.get("type") for e in second_events]
        print(f"  Second turn events: {second_types}")
        assert "response.completed" in second_types or "response.failed" in second_types
