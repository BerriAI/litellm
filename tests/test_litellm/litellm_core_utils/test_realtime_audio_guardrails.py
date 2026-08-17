import base64
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.types.guardrails import GuardrailEventHooks

AUDIO_B64 = base64.b64encode(b"\x00\x01" * 160).decode()


class RecordingAudioGuardrail(CustomGuardrail):
    """Records every audio frame handed to it; blocks model frames listed in block_seq."""

    def __init__(self, block_seq=(), **kwargs):
        self.seen: list[dict] = []
        self.block_seq = set(block_seq)
        super().__init__(guardrail_name="record-audio", **kwargs)

    def should_run_guardrail(self, data, event_type) -> bool:
        return event_type == GuardrailEventHooks.realtime_audio

    async def apply_guardrail(self, inputs, request_data=None, input_type=None, logging_obj=None):
        for frame in inputs.get("audio", []):
            self.seen.append(frame)
            if frame["speaker"] == "model" and frame["sequence"] in self.block_seq:
                raise ValueError("blocked by audio guardrail")
        return inputs


class TranscriptOnlyGuardrail(CustomGuardrail):
    """Opts into the transcript hook only, so it must never be handed audio."""

    def __init__(self, **kwargs):
        self.calls = 0
        super().__init__(guardrail_name="transcript-only", **kwargs)

    def should_run_guardrail(self, data, event_type) -> bool:
        return event_type == GuardrailEventHooks.realtime_input_transcription

    async def apply_guardrail(self, inputs, request_data=None, input_type=None, logging_obj=None):
        self.calls += 1
        return inputs


def _streaming() -> RealTimeStreaming:
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    websocket = MagicMock()
    websocket.send_text = AsyncMock()
    backend_ws = MagicMock()
    backend_ws.send = AsyncMock()
    return RealTimeStreaming(websocket, backend_ws, logging_obj, request_data={})


def _audio_delta(event_type: str = "response.audio.delta") -> tuple[dict, str]:
    event = {"type": event_type, "delta": AUDIO_B64}
    return event, json.dumps(event)


@pytest.fixture(autouse=True)
def _reset_callbacks():
    original = litellm.callbacks
    yield
    litellm.callbacks = original


@pytest.mark.asyncio
async def test_guardrail_receives_frame_with_speaker_rate_and_sequence():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    withheld = await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="user")

    assert withheld is False
    assert len(guardrail.seen) == 1
    frame = guardrail.seen[0]
    assert frame["speaker"] == "user"
    assert frame["encoding"] == "pcm16"
    assert frame["sequence"] == 0
    assert base64.b64decode(frame["audio"]) == b"\x00\x01" * 160


@pytest.mark.asyncio
async def test_default_sample_rates_differ_by_direction():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="user")
    await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="model")

    rates = {f["speaker"]: f["sample_rate_hz"] for f in guardrail.seen}
    assert rates == {"user": 16000, "model": 24000}


@pytest.mark.asyncio
async def test_declared_session_format_overrides_default_rate():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    streaming._remember_declared_audio_formats(
        {"session": {"input_audio_format": "g711_ulaw", "output_audio_format": "g711_alaw"}}
    )
    await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="user")
    await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="model")

    rates = {f["speaker"]: f["sample_rate_hz"] for f in guardrail.seen}
    assert rates == {"user": 8000, "model": 8000}


@pytest.mark.asyncio
async def test_sequence_is_monotonic_per_speaker():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    for _ in range(3):
        await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="user")
    await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="model")

    assert [f["sequence"] for f in guardrail.seen if f["speaker"] == "user"] == [0, 1, 2]
    assert [f["sequence"] for f in guardrail.seen if f["speaker"] == "model"] == [0]


@pytest.mark.asyncio
async def test_model_frame_is_withheld_and_session_stays_open():
    guardrail = RecordingAudioGuardrail(block_seq={1})
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    results = []
    for _ in range(3):
        event, event_str = _audio_delta()
        results.append(await streaming._send_event_to_client(event, event_str))

    # Frame seq=1 withheld; the other two reach the client.
    assert results == [True, False, True]
    assert streaming.websocket.send_text.await_count == 2
    assert len(guardrail.seen) == 3
    streaming.websocket.close.assert_not_called()


@pytest.mark.asyncio
async def test_ga_spelling_of_audio_delta_is_screened_too():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    event, event_str = _audio_delta("response.output_audio.delta")
    await streaming._send_event_to_client(event, event_str)

    assert len(guardrail.seen) == 1


@pytest.mark.asyncio
async def test_non_audio_events_are_untouched():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    event = {"type": "session.created", "session": {"id": "s1"}}
    sent = await streaming._send_event_to_client(event, json.dumps(event))

    assert sent is True
    assert guardrail.seen == []


@pytest.mark.asyncio
async def test_transcript_only_guardrail_never_sees_audio_and_adds_no_path():
    guardrail = TranscriptOnlyGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    assert streaming._has_realtime_audio_guardrails() is False

    event, event_str = _audio_delta()
    sent = await streaming._send_event_to_client(event, event_str)

    # Forwarded verbatim; the guardrail is never invoked. Zero cost when unused.
    assert sent is True
    assert guardrail.calls == 0


@pytest.mark.asyncio
async def test_crashing_guardrail_does_not_discard_audio():
    class Exploding(CustomGuardrail):
        def should_run_guardrail(self, data, event_type) -> bool:
            return event_type == GuardrailEventHooks.realtime_audio

        async def apply_guardrail(self, inputs, **kwargs):
            raise RuntimeError("bug in guardrail, not a policy block")

    litellm.callbacks = [Exploding(guardrail_name="boom")]
    streaming = _streaming()

    withheld = await streaming.run_realtime_audio_guardrails(AUDIO_B64, speaker="model")

    assert withheld is False


@pytest.mark.asyncio
async def test_empty_or_missing_delta_is_not_screened():
    guardrail = RecordingAudioGuardrail()
    litellm.callbacks = [guardrail]
    streaming = _streaming()

    for event in ({"type": "response.audio.delta"}, {"type": "response.audio.delta", "delta": ""}):
        assert await streaming._send_event_to_client(event, json.dumps(event)) is True
    assert guardrail.seen == []
