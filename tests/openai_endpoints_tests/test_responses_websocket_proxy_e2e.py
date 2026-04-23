"""
E2E tests for OpenAI Responses API WebSocket mode through the LiteLLM proxy.

Connects to ws://0.0.0.0:4000/v1/responses, sends response.create events,
and validates the streamed response events.

Requires:
  - Proxy running: python -m litellm.proxy.proxy_cli --config <config> --port 4000
  - Model configured in proxy (e.g. gpt-4o-mini)

See: https://developers.openai.com/api/docs/guides/websocket-mode/
"""

import asyncio
import json
import os

import httpx
import pytest

# ── Configuration ─────────────────────────────────────────────────────────────
PROXY_BASE_URL = os.environ.get("LITELLM_PROXY_BASE_URL", "ws://0.0.0.0:4000")
PROXY_MASTER_KEY = os.environ.get("LITELLM_PROXY_KEY", "sk-1234")
PROXY_MODEL = os.environ.get("LITELLM_PROXY_RESPONSES_MODEL", "gpt-4o-mini")
# ──────────────────────────────────────────────────────────────────────────────


def _generate_key() -> str:
    """Generate a key for testing via proxy key/generate endpoint."""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {PROXY_MASTER_KEY}",
        "Content-Type": "application/json",
    }
    response = httpx.post(url, headers=headers, json={}, timeout=10)
    if response.status_code != 200:
        raise Exception(
            f"Key generation failed with status: {response.status_code}. "
            "Is the proxy running?"
        )
    return response.json()["key"]


def _assert_basic_response(events: list[dict], label: str = "") -> None:
    """Assert that events contain response.created, response.completed, and usage."""
    prefix = f"[{label}] " if label else ""
    types = [e.get("type") for e in events]
    assert len(events) > 0, f"{prefix}no events received"
    assert "response.created" in types, f"{prefix}missing response.created, got: {types}"
    assert "response.completed" in types, (
        f"{prefix}missing response.completed, got: {types}"
    )
    completed = next(e for e in events if e.get("type") == "response.completed")
    resp = completed.get("response", {})
    assert resp.get("status") == "completed", (
        f"{prefix}status != completed: {resp.get('status')}"
    )
    usage = resp.get("usage", {})
    assert usage.get("input_tokens", 0) > 0, f"{prefix}input_tokens=0"
    assert usage.get("output_tokens", 0) > 0, f"{prefix}output_tokens=0"
    streaming_types = {
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_item.done",
    }
    found = streaming_types & set(types)
    assert found, f"{prefix}no streaming delta events found, got: {types}"


@pytest.mark.asyncio
async def test_responses_websocket_proxy_basic():
    """
    Sends a simple response.create event to the proxy WebSocket endpoint
    and validates response.created, response.completed, and streaming events.
    """
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets not installed")

    try:
        key = _generate_key()
    except Exception as e:
        pytest.skip(
            f"Proxy not available or key generation failed: {e}. "
            "Start proxy: python -m litellm.proxy.proxy_cli --config <config> --port 4000"
        )

    url = f"{PROXY_BASE_URL}/v1/responses?model={PROXY_MODEL}"
    headers = {"Authorization": f"Bearer {key}"}
    events: list[dict] = []

    try:
        async with websockets.connect(
            url, additional_headers=headers, open_timeout=5
        ) as ws:
            payload = {
                "type": "response.create",
                "model": PROXY_MODEL,
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Say hello in one word."}
                        ],
                    }
                ],
                "tools": [],
            }
            await ws.send(json.dumps(payload))
            for _ in range(50):
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                event = json.loads(msg)
                events.append(event)
                if event.get("type") in (
                    "response.completed",
                    "response.failed",
                    "error",
                ):
                    break
    except Exception as e:
        pytest.fail(
            f"WebSocket connection failed: {e}. "
            "Ensure proxy is running and model is configured."
        )

    _assert_basic_response(events, "proxy-basic")


@pytest.mark.asyncio
async def test_responses_websocket_proxy_multi_turn():
    """
    Sends two sequential response.create events with previous_response_id
    to validate multi-turn conversation over a single WebSocket.
    """
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets not installed")

    try:
        key = _generate_key()
    except Exception as e:
        pytest.skip(
            f"Proxy not available or key generation failed: {e}. "
            "Start proxy: python -m litellm.proxy.proxy_cli --config <config> --port 4000"
        )

    url = f"{PROXY_BASE_URL}/v1/responses?model={PROXY_MODEL}"
    headers = {"Authorization": f"Bearer {key}"}
    all_events: list[dict] = []
    completed: list[dict] = []
    first_id = None

    try:
        async with websockets.connect(
            url, additional_headers=headers, open_timeout=5
        ) as ws:
            # Turn 1
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "model": PROXY_MODEL,
                        "store": True,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "Remember the number 7. Just say OK.",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            for _ in range(50):
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                event = json.loads(msg)
                all_events.append(event)
                if event.get("type") == "response.completed":
                    completed.append(event)
                    first_id = event.get("response", {}).get("id")
                    break
                if event.get("type") in ("response.failed", "error"):
                    break

            assert first_id, "Turn 1 never completed"

            # Turn 2
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "model": PROXY_MODEL,
                        "store": True,
                        "previous_response_id": first_id,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "What number did I tell you to remember?",
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            for _ in range(50):
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                event = json.loads(msg)
                all_events.append(event)
                if event.get("type") == "response.completed":
                    completed.append(event)
                    break
                if event.get("type") in ("response.failed", "error"):
                    break

    except Exception as e:
        pytest.fail(
            f"WebSocket multi-turn failed: {e}. "
            "Ensure proxy is running and model is configured."
        )

    assert len(completed) >= 2, (
        f"Expected 2 response.completed events, got {len(completed)}"
    )
    assert completed[1].get("response", {}).get("status") == "completed"
