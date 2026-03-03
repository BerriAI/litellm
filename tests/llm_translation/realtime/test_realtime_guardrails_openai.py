"""
Integration tests for RealTimeStreaming guardrails against a live OpenAI backend.

These tests require OPENAI_API_KEY and are skipped if not set.

They verify end-to-end that:
  1. A text message blocked by a guardrail -> error event sent to client, NO AI response.
  2. A voice transcript blocked by a guardrail -> error event sent, response.create NOT sent.
  3. A clean text message passes through and triggers a real OpenAI response.

Run with:
    poetry run pytest tests/llm_translation/realtime/test_realtime_guardrails_openai.py -v -s
"""

import asyncio
import json
import os
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.types.guardrails import GuardrailEventHooks

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_REALTIME_URL = (
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
)

pytestmark = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY not set - skipping OpenAI realtime integration tests",
)

# A unique phrase guaranteed NOT to appear in normal assistant output.
BLOCKED_PHRASE = "XSECRETBLOCKTESTPHRASEX"


class PhraseBlockingGuardrail(CustomGuardrail):
    """Blocks any message containing BLOCKED_PHRASE."""

    async def apply_guardrail(
        self, inputs, request_data, input_type, logging_obj=None
    ):
        for text in inputs.get("texts", []):
            if BLOCKED_PHRASE in text:
                raise ValueError(
                    "Content blocked: contains forbidden test phrase."
                )
        return inputs


def _make_guardrail(event_hook=GuardrailEventHooks.pre_call):
    return PhraseBlockingGuardrail(
        guardrail_name="integration-test-guard",
        event_hook=event_hook,
        default_on=True,
    )


async def _wait_for_event(
    client_events: List[dict], event_type: str, timeout: float = 15.0
) -> dict:
    """Poll client_events list until an event with matching type appears."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        matching = [e for e in client_events if e.get("type") == event_type]
        if matching:
            return matching[0]
        await asyncio.sleep(0.05)
    raise TimeoutError(
        f"Timed out waiting for '{event_type}'. Got so far: {[e.get('type') for e in client_events]}"
    )


async def _build_streaming(client_events: List[dict], backend_ws, request_data=None):
    """Create a RealTimeStreaming with a mock client WebSocket that captures events."""
    client_ws = MagicMock()
    input_queue: asyncio.Queue = asyncio.Queue()

    async def send_text(data: str):
        client_events.append(json.loads(data))

    client_ws.send_text = send_text
    client_ws.receive_text = input_queue.get

    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj.success_handler = MagicMock()
    logging_obj.model_call_details = {}

    streaming = RealTimeStreaming(
        websocket=client_ws,
        backend_ws=backend_ws,
        logging_obj=logging_obj,
        request_data=request_data or {"guardrails": ["integration-test-guard"]},
    )
    return streaming, input_queue


@pytest.mark.asyncio
async def test_text_message_blocked_by_guardrail_no_ai_response():
    """
    Send a text message containing the blocked phrase.
    Guardrail must:
      - Send error event (guardrail_violation) to client.
      - Send response.audio_transcript.delta with the block message to client.
      - NOT forward response.create to OpenAI (no AI response).
    """
    import websockets

    guardrail = _make_guardrail(GuardrailEventHooks.pre_call)
    litellm.callbacks = [guardrail]

    client_events: List[dict] = []

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as backend_ws:
            streaming, input_queue = await _build_streaming(client_events, backend_ws)

            # Start backend -> client forwarding
            backend_task = asyncio.create_task(
                streaming.backend_to_client_send_messages()
            )
            # Start client -> backend forwarding (reads from input_queue)
            client_task = asyncio.create_task(streaming.client_ack_messages())

            try:
                # Wait until session is ready
                await _wait_for_event(client_events, "session.created", timeout=15)

                # Send the blocked message + response.create
                blocked_item = json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"Hello {BLOCKED_PHRASE}",
                                }
                            ],
                        },
                    }
                )
                await input_queue.put(blocked_item)
                # Give guardrail time to process before the follow-up response.create
                await asyncio.sleep(0.3)
                await input_queue.put(json.dumps({"type": "response.create"}))

                # Allow time for guardrail round-trip
                await asyncio.sleep(3.0)

            finally:
                backend_task.cancel()
                client_task.cancel()
                await asyncio.gather(backend_task, client_task, return_exceptions=True)

        # --- Assertions ---
        event_types = [e.get("type") for e in client_events]

        # 1. Must have received guardrail error
        error_events = [e for e in client_events if e.get("type") == "error"]
        assert len(error_events) >= 1, (
            f"Expected at least one error event but got: {event_types}"
        )
        assert error_events[0]["error"]["type"] == "guardrail_violation", (
            f"Wrong error type: {error_events[0]}"
        )

        # 2. Must have the guardrail message surfaced as an AI transcript delta
        transcript_deltas = [
            e
            for e in client_events
            if e.get("type") == "response.audio_transcript.delta"
        ]
        assert len(transcript_deltas) >= 1, (
            f"Expected guardrail message in transcript delta, got: {event_types}"
        )

        # 3. No real AI response should have been generated - response.done would only
        #    appear if we sent a response.create and OpenAI replied. We allow it in the
        #    synthetic form (empty output=[]) but NOT with actual AI content.
        done_events = [e for e in client_events if e.get("type") == "response.done"]
        for done in done_events:
            output = done.get("response", {}).get("output", [])
            ai_texts = [
                c.get("text", "") or c.get("transcript", "")
                for item in output
                for c in item.get("content", [])
            ]
            real_ai_text = " ".join(ai_texts).strip()
            assert real_ai_text == "", (
                f"AI responded with real content even though message was blocked: {real_ai_text!r}"
            )

    finally:
        litellm.callbacks = []


@pytest.mark.asyncio
async def test_voice_transcript_blocked_by_guardrail():
    """
    Simulate a backend-side voice transcription event containing the blocked phrase.
    Guardrail must block it - no response.create sent to OpenAI.
    """
    from websockets.exceptions import ConnectionClosed

    guardrail = _make_guardrail(GuardrailEventHooks.realtime_input_transcription)
    litellm.callbacks = [guardrail]

    client_events: List[dict] = []

    # Build the transcript event that would come from the OpenAI backend
    transcript_event = json.dumps(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": f"This is {BLOCKED_PHRASE} in my voice message",
            "item_id": "item_integ_test",
        }
    ).encode()

    # Mock backend that delivers the transcript then closes
    backend_ws = MagicMock()
    backend_ws.recv = AsyncMock(
        side_effect=[
            transcript_event,
            ConnectionClosed(None, None),
        ]
    )
    backend_ws.send = AsyncMock()

    try:
        streaming, _ = await _build_streaming(client_events, backend_ws)
        await streaming.backend_to_client_send_messages()

        event_types = [e.get("type") for e in client_events]

        # 1. Error event must be sent to client
        error_events = [e for e in client_events if e.get("type") == "error"]
        assert len(error_events) >= 1, (
            f"Expected guardrail error event, got: {event_types}"
        )
        assert error_events[0]["error"]["type"] == "guardrail_violation"

        # 2. response.create must NOT have been sent to backend
        sent_to_backend = [
            json.loads(c.args[0])
            for c in backend_ws.send.call_args_list
            if c.args and isinstance(c.args[0], str)
        ]
        response_creates = [
            e for e in sent_to_backend if e.get("type") == "response.create"
        ]
        assert len(response_creates) == 0, (
            f"Guardrail should have stopped response.create, got: {sent_to_backend}"
        )

        # 3. Guardrail message surfaced as AI transcript delta
        transcript_deltas = [
            e
            for e in client_events
            if e.get("type") == "response.audio_transcript.delta"
        ]
        assert len(transcript_deltas) >= 1, (
            f"Expected guardrail message in transcript delta, got: {event_types}"
        )

    finally:
        litellm.callbacks = []


@pytest.mark.asyncio
async def test_clean_text_message_passes_through_to_openai():
    """
    A clean message (no blocked phrase) must pass the guardrail and result in a real
    AI response from OpenAI (response.done with non-empty output).
    """
    import websockets

    guardrail = _make_guardrail(GuardrailEventHooks.pre_call)
    litellm.callbacks = [guardrail]

    client_events: List[dict] = []

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as backend_ws:
            streaming, input_queue = await _build_streaming(client_events, backend_ws)

            backend_task = asyncio.create_task(
                streaming.backend_to_client_send_messages()
            )
            client_task = asyncio.create_task(streaming.client_ack_messages())

            try:
                await _wait_for_event(client_events, "session.created", timeout=15)

                # Send a clean message
                clean_item = json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Reply with just: OK"}
                            ],
                        },
                    }
                )
                await input_queue.put(clean_item)
                await asyncio.sleep(0.1)
                await input_queue.put(json.dumps({"type": "response.create"}))

                # Wait for OpenAI to respond
                await _wait_for_event(client_events, "response.done", timeout=30)

            finally:
                backend_task.cancel()
                client_task.cancel()
                await asyncio.gather(backend_task, client_task, return_exceptions=True)

        # No guardrail error should have been sent
        error_events = [e for e in client_events if e.get("type") == "error"]
        guardrail_errors = [
            e for e in error_events if e.get("error", {}).get("type") == "guardrail_violation"
        ]
        assert len(guardrail_errors) == 0, (
            f"Clean message should not trigger guardrail, got: {guardrail_errors}"
        )

        # AI response must be present
        done_events = [e for e in client_events if e.get("type") == "response.done"]
        assert len(done_events) >= 1, (
            f"Expected response.done from OpenAI, got: {[e.get('type') for e in client_events]}"
        )

    finally:
        litellm.callbacks = []
