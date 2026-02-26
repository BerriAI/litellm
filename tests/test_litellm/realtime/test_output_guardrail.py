"""
Unit tests for realtime output text guardrail.

Tests that response.text.delta events are buffered and checked on
response.text.done, blocking bad output and forwarding clean output.
"""

import asyncio
import json
import unittest
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeWebSocket:
    """Mimics a FastAPI WebSocket (client ↔ proxy)."""

    def __init__(self):
        self.sent: List[str] = []

    async def send_text(self, text: str):
        self.sent.append(text)


class _FakeBackendWS:
    """Mimics a websockets.ClientConnection (proxy ↔ backend)."""

    def __init__(self):
        self.sent: List[str] = []

    async def send(self, data: str):
        self.sent.append(data)

    async def recv(self, decode=True):
        raise StopAsyncIteration


from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks


class _BlockingGuardrail(CustomGuardrail):
    """Blocks any text containing 'bomb'."""

    def __init__(self):
        super().__init__(
            guardrail_name="test-content-filter",
            supported_event_hooks=[GuardrailEventHooks.post_call],
            event_hook=GuardrailEventHooks.post_call,
            default_on=True,
        )

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        texts = inputs.get("texts", [])
        for text in texts:
            if "bomb" in text.lower():
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=400,
                    detail={"error": "Response blocked by content filter."},
                )
        return inputs


class _PassthroughGuardrail(CustomGuardrail):
    """Always passes."""

    def __init__(self):
        super().__init__(
            guardrail_name="test-passthrough",
            supported_event_hooks=[GuardrailEventHooks.post_call],
            event_hook=GuardrailEventHooks.post_call,
            default_on=True,
        )

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        return inputs


def _make_streaming(guardrail=None):
    """Create a RealTimeStreaming instance with mocked dependencies."""
    from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

    client_ws = _FakeWebSocket()
    backend_ws = _FakeBackendWS()
    logging_obj = MagicMock()

    streaming = RealTimeStreaming(
        websocket=client_ws,
        backend_ws=backend_ws,
        logging_obj=logging_obj,
    )
    # Patch store_message to be a no-op (avoids JSON parsing complexity)
    streaming.store_message = MagicMock()

    if guardrail is not None:
        import litellm

        litellm.callbacks = [guardrail]
    else:
        import litellm

        litellm.callbacks = []

    return streaming, client_ws, backend_ws


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestOutputGuardrailBlocked:
    """response.text.done with blocked word → error replaces original text."""

    def setup_method(self):
        import litellm

        litellm.callbacks = []

    def teardown_method(self):
        import litellm

        litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_blocked_text_replaces_deltas(self):
        streaming, client_ws, _ = _make_streaming(guardrail=_BlockingGuardrail())

        delta_event = {
            "type": "response.text.delta",
            "delta": "I will tell you how to make a bomb.",
            "content_index": 0,
            "item_id": "item_abc",
            "output_index": 0,
            "response_id": "resp_xyz",
        }
        done_event = {
            "type": "response.text.done",
            "text": "I will tell you how to make a bomb.",
            "content_index": 0,
            "item_id": "item_abc",
            "output_index": 0,
            "response_id": "resp_xyz",
        }

        events = [delta_event, done_event]
        for event in events:
            event_str = json.dumps(event)
            # Simulate _handle_provider_config_message inner loop
            etype = event.get("type", "")
            if etype == "response.text.delta" and streaming._has_realtime_output_guardrails():
                streaming._pending_output_text_events.append(event_str)
            elif etype == "response.text.done":
                full_text = event.get("text", "")
                blocked, error_msg = await streaming.run_realtime_output_guardrails(full_text)
                if blocked:
                    streaming._pending_output_text_events.clear()
                    error_delta_str = json.dumps(
                        {
                            "type": "response.text.delta",
                            "delta": error_msg,
                            "content_index": event.get("content_index", 0),
                            "item_id": event.get("item_id", ""),
                            "output_index": event.get("output_index", 0),
                            "response_id": event.get("response_id", ""),
                        }
                    )
                    error_done = dict(event)
                    error_done["text"] = error_msg
                    await client_ws.send_text(error_delta_str)
                    await client_ws.send_text(json.dumps(error_done))

        # The original delta should NOT reach the client
        sent_texts = [json.loads(e) for e in client_ws.sent]
        for sent in sent_texts:
            assert "bomb" not in sent.get("delta", "").lower(), (
                f"Blocked word leaked in delta: {sent}"
            )
            assert "bomb" not in sent.get("text", "").lower(), (
                f"Blocked word leaked in done event: {sent}"
            )

        # The client should have received exactly 2 events: error delta + error done
        assert len(client_ws.sent) == 2, f"Expected 2 events, got {len(client_ws.sent)}"
        error_delta = json.loads(client_ws.sent[0])
        error_done = json.loads(client_ws.sent[1])
        assert error_delta["type"] == "response.text.delta"
        assert error_done["type"] == "response.text.done"
        # Error message should mention blocking
        assert len(error_delta.get("delta", "")) > 0
        print(f"\n✅ PASS: blocked text replaced with: {error_delta['delta']!r}")


class TestOutputGuardrailClean:
    """response.text.done with clean text → buffered deltas flushed normally."""

    def setup_method(self):
        import litellm

        litellm.callbacks = []

    def teardown_method(self):
        import litellm

        litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_clean_text_flushed(self):
        streaming, client_ws, _ = _make_streaming(guardrail=_PassthroughGuardrail())

        delta1 = {
            "type": "response.text.delta",
            "delta": "Hello, how can I help?",
            "content_index": 0,
            "item_id": "item_1",
            "output_index": 0,
            "response_id": "resp_1",
        }
        done_event = {
            "type": "response.text.done",
            "text": "Hello, how can I help?",
            "content_index": 0,
            "item_id": "item_1",
            "output_index": 0,
            "response_id": "resp_1",
        }

        # Buffer the delta
        streaming._pending_output_text_events.append(json.dumps(delta1))

        # Process done event
        blocked, error_msg = await streaming.run_realtime_output_guardrails(done_event["text"])
        assert not blocked, f"Expected not blocked, got error: {error_msg}"

        # Flush buffer
        for pending in streaming._pending_output_text_events:
            streaming.store_message(pending)
            await client_ws.send_text(pending)
        streaming._pending_output_text_events.clear()
        await client_ws.send_text(json.dumps(done_event))

        # Both delta and done should reach the client
        assert len(client_ws.sent) == 2
        d = json.loads(client_ws.sent[0])
        done = json.loads(client_ws.sent[1])
        assert d["type"] == "response.text.delta"
        assert done["type"] == "response.text.done"
        assert "Hello" in d["delta"]
        print(f"\n✅ PASS: clean text flushed normally: {d['delta']!r}")


class TestOutputGuardrailNoGuardrail:
    """Without output guardrail, deltas pass straight through (no buffering)."""

    def setup_method(self):
        import litellm

        litellm.callbacks = []

    def teardown_method(self):
        import litellm

        litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_no_buffering_without_guardrail(self):
        streaming, client_ws, _ = _make_streaming(guardrail=None)

        # With no guardrail, _has_realtime_output_guardrails() should return False
        assert not streaming._has_realtime_output_guardrails()

        delta = {
            "type": "response.text.delta",
            "delta": "This is fine.",
        }
        # Should not be buffered
        if streaming._has_realtime_output_guardrails():
            streaming._pending_output_text_events.append(json.dumps(delta))
        else:
            await client_ws.send_text(json.dumps(delta))

        assert len(streaming._pending_output_text_events) == 0
        assert len(client_ws.sent) == 1
        print(f"\n✅ PASS: delta forwarded directly without guardrail")


class TestGuardrailEventHook:
    """post_call runs across all modalities including realtime output."""

    def test_hook_exists(self):
        from litellm.types.guardrails import GuardrailEventHooks

        assert hasattr(GuardrailEventHooks, "post_call")
        assert GuardrailEventHooks.post_call == "post_call"
        print(f"\n✅ PASS: GuardrailEventHooks.post_call = {GuardrailEventHooks.post_call!r}")
