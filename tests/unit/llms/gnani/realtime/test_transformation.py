import asyncio
import base64
import json
import math
import struct
from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

from litellm import Router
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.base_llm.realtime.transcription_protocol import RealtimeTranscriptionProtocolError
from litellm.llms.gnani.realtime.audio import GnaniAudioStream
from litellm.llms.gnani.realtime.transformation import GnaniRealtimeConfig
from litellm.types.llms.openai import OpenAIRealtimeEvents
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize("model", ["prisma-v2.5", "prisma-v3.0"])
def test_client_pcm_is_framed_across_appends(model: str) -> None:
    config: Final = ProviderConfigManager.get_provider_realtime_config(model, LlmProviders.GNANI)
    assert config is not None
    config.transform_realtime_request(_session_update(16_000, {"language": "hi"}), model)
    samples: Final = struct.pack("<512h", *range(-256, 256))

    def append(pcm: bytes) -> tuple[str | bytes, ...]:
        return tuple(config.transform_realtime_request(_append(pcm), model))

    assert append(samples[:10]) == ()
    assert append(samples[10:]) == (samples,)
    assert tuple(config.transform_realtime_request('{"type":"input_audio_buffer.commit"}', model)) == (
        '{"type":"commit"}',
    )


@pytest.mark.parametrize(("rate", "alias_hz"), [(24_000, 11_000), (44_100, 10_000), (48_000, 10_000)])
def test_resampling_filters_out_tones_above_the_wire_nyquist_and_keeps_speech_tones(rate: int, alias_hz: int) -> None:
    full_scale_rms: Final = 10_000 / math.sqrt(2)
    assert _rms(_forwarded_tone(rate, alias_hz)) < 0.01 * full_scale_rms
    assert _rms(_forwarded_tone(rate, 1_000)) == pytest.approx(full_scale_rms, rel=0.02)


def _forwarded_tone(rate: int, hz: int) -> bytes:
    config: Final = GnaniRealtimeConfig()
    config.transform_realtime_request(_session_update(rate, {}), "prisma-v3.0")
    tone: Final = struct.pack(f"<{rate}h", *(round(10_000 * math.sin(2 * math.pi * hz * i / rate)) for i in range(rate)))
    chunk: Final = rate // 5 * 2
    appends: Final = (
        config.transform_realtime_request(_append(tone[offset : offset + chunk]), "prisma-v3.0")
        for offset in range(0, len(tone), chunk)
    )
    return _pcm(chain.from_iterable(appends))[3_200:]


def _pcm(frames: Iterable[str | bytes]) -> bytes:
    return b"".join(frame for frame in frames if isinstance(frame, bytes))


@pytest.mark.parametrize("rate", [24_000, 44_100, 48_000])
def test_commit_forwards_the_end_of_a_short_burst_held_back_by_the_resampling_filter(rate: int) -> None:
    config: Final = GnaniRealtimeConfig()
    config.transform_realtime_request(_session_update(rate, {}), "prisma-v3.0")
    burst: Final = struct.pack(
        f"<{rate // 500}h", *(round(10_000 * math.sin(2 * math.pi * 1_000 * i / rate)) for i in range(rate // 500))
    )
    assert config.transform_realtime_request(_append(burst), "prisma-v3.0") == ()
    committed: Final = _pcm(config.transform_realtime_request('{"type":"input_audio_buffer.commit"}', "prisma-v3.0"))
    burst_energy_at_wire_rate: Final = _energy(burst) * 16_000 / rate
    assert _energy(committed) == pytest.approx(burst_energy_at_wire_rate, rel=0.1)


def _energy(pcm: bytes) -> float:
    return sum(sample * sample for sample in struct.unpack(f"<{len(pcm) // 2}h", pcm))


def _rms(pcm: bytes) -> float:
    samples: Final = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _session_update(rate: int, transcription: Mapping[str, object], **audio_input: object) -> str:
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": rate},
                        "transcription": transcription,
                        **audio_input,
                    }
                },
            },
        }
    )


def _append(pcm: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})


@pytest.mark.parametrize(
    ("turn_detection", "strict_error"),
    [
        (
            {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500},
            "prompt, turn_detection.prefix_padding_ms, turn_detection.silence_duration_ms, turn_detection.threshold",
        ),
        ({"type": "semantic_vad", "eagerness": "auto"}, "server_vad only"),
        (None, "server_vad only"),
    ],
)
def test_turn_detection_other_than_server_vad_and_a_prompt_are_rejected_unless_drop_params_drops_them(
    turn_detection: Mapping[str, object] | None, strict_error: str
) -> None:
    update: Final = _session_update(
        24_000, {"model": "gnani-prisma-v3", "language": "en", "prompt": "Asha"}, turn_detection=turn_detection
    )
    strict: Final = ProviderConfigManager.get_provider_realtime_config("prisma-v3.0", LlmProviders.GNANI)
    assert strict is not None
    with pytest.raises(RealtimeTranscriptionProtocolError, match=strict_error):
        strict.transform_realtime_request(update, "prisma-v3.0")
    dropping: Final = ProviderConfigManager.get_provider_realtime_config(
        "prisma-v3.0", LlmProviders.GNANI, drop_params=True
    )
    assert dropping is not None
    (wire,) = dropping.transform_realtime_request(update, "prisma-v3.0")
    assert isinstance(wire, str)
    assert json.loads(wire) == {"type": "update_settings", "language_code": "en-IN"}


@pytest.mark.parametrize(
    ("session", "setting"),
    [
        ({"audio": {"input": {"noise_reduction": {"type": "near_field"}}}}, "noise_reduction"),
        ({"input_audio_noise_reduction": {"type": "far_field"}}, "noise_reduction"),
        ({"include": ["item.input_audio_transcription.logprobs"]}, "include"),
    ],
)
def test_noise_reduction_and_include_are_rejected_unless_drop_params_drops_them(
    session: Mapping[str, object], setting: str
) -> None:
    update: Final = json.dumps({"type": "session.update", "session": {"type": "transcription", **session}})
    with pytest.raises(RealtimeTranscriptionProtocolError, match=f"settings: {setting}\\. Set drop_params"):
        GnaniRealtimeConfig().transform_realtime_request(update, "prisma-v3.0")
    (wire,) = GnaniRealtimeConfig(drop_params=True).transform_realtime_request(update, "prisma-v3.0")
    assert isinstance(wire, str)
    assert json.loads(wire) == {"type": "update_settings"}


def test_noise_reduction_off_and_an_empty_include_are_accepted_without_drop_params() -> None:
    update: Final = json.dumps(
        {
            "type": "session.update",
            "session": {"type": "transcription", "audio": {"input": {"noise_reduction": None}}, "include": []},
        }
    )
    (wire,) = GnaniRealtimeConfig().transform_realtime_request(update, "prisma-v3.0")
    assert isinstance(wire, str)
    assert json.loads(wire) == {"type": "update_settings"}


@pytest.mark.parametrize("deployment_headers",[{}, {"lang_code": "indic-auto"}, {"lang_code": "hi, en"}])
def test_prisma_v25_needs_one_explicit_language_before_audio(deployment_headers: Mapping[str, str]) -> None:
    v25: Final = ProviderConfigManager.get_provider_realtime_config("prisma-v2.5", LlmProviders.GNANI)
    v3: Final = ProviderConfigManager.get_provider_realtime_config("prisma-v3.0", LlmProviders.GNANI)
    assert v25 is not None and v3 is not None
    v25.validate_environment(deployment_headers, "prisma-v2.5", api_key="deployment-key")
    v3.validate_environment(deployment_headers, "prisma-v3.0", api_key="deployment-key")
    audio: Final = _append(bytes(2))
    assert v3.transform_realtime_request(audio, "prisma-v3.0") == ()
    with pytest.raises(RealtimeTranscriptionProtocolError, match="one explicit language"):
        v25.transform_realtime_request(audio, "prisma-v2.5")
    with pytest.raises(RealtimeTranscriptionProtocolError, match="one explicit language"):
        v25.transform_realtime_request(_session_update(24_000, {"language": "indic-auto"}), "prisma-v2.5")
    (wire,) = v25.transform_realtime_request(_session_update(24_000, {"language": "en"}), "prisma-v2.5")
    assert isinstance(wire, str)
    assert json.loads(wire) == {"type": "update_settings", "language_code": "en-IN"}
    assert v25.transform_realtime_request(audio, "prisma-v2.5") == ()


def test_unknown_gnani_model_cannot_fall_back_to_another_prisma_version() -> None:
    assert ProviderConfigManager.get_provider_realtime_config("unknown-prisma", LlmProviders.GNANI) is None


def test_a_deployment_with_a_stored_transcription_only_endpoint_list_keeps_realtime_on() -> None:
    Router(
        model_list=[
            {
                "model_name": "stale-admin-ui-save",
                "litellm_params": {"model": "gnani/prisma-v3.0", "api_key": "deployment-key"},
                "model_info": {"supported_endpoints": ["/v1/audio/transcriptions"]},
            }
        ]
    )
    assert ProviderConfigManager.get_provider_realtime_config("prisma-v3.0", LlmProviders.GNANI) is not None


def test_a_priced_deployment_of_an_unknown_gnani_model_stays_unsupported() -> None:
    Router(
        model_list=[
            {
                "model_name": "typo",
                "litellm_params": {
                    "model": "gnani/prisma-v9.0",
                    "api_key": "deployment-key",
                    "input_cost_per_second": 0.01,
                },
                "model_info": {"supported_endpoints": ["/v1/realtime"]},
            }
        ]
    )
    assert ProviderConfigManager.get_provider_realtime_config("prisma-v9.0", LlmProviders.GNANI) is None


def test_opening_and_updating_a_session_preserves_the_client_rate_and_acknowledged_language() -> None:
    config: Final = GnaniRealtimeConfig()
    headers: Final = config.validate_environment(
        {"X-Model": "wrong", "X-API-Key-ID": "wrong", "lang_code": "hi, en", "topic": "reading"},
        "prisma-v3.0",
        api_key="deployment-key",
    )
    assert headers == {
        "model": "prisma-v3.0",
        "x-api-key-id": "deployment-key",
        "language_code": "hi-IN,en-IN",
        "topic": "reading",
        "x-sample-rate": "16000",
    }
    (opened,) = _events(config, {"type": "connected", "request_id": "session-one", "config": {"sample_rate": 16000}})
    assert opened["type"] == "session.created"
    update: Final = {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 8000}, "transcription": {"language": "en"}}},
        },
    }
    (wire,) = config.transform_realtime_request(json.dumps(update), "prisma-v3.0")
    assert json.loads(wire) == {"type": "update_settings", "language_code": "en-IN"}
    (acknowledged,) = _events(config, {"type": "settings_updated", "config": {"language_code": "en-IN"}})
    assert acknowledged["type"] == "session.updated"
    assert opened["session"]["id"] == acknowledged["session"]["id"] == "session-one"
    assert opened["session"]["audio"]["input"]["format"]["rate"] == 24000
    assert acknowledged["session"]["audio"]["input"]["format"]["rate"] == 8000
    assert acknowledged["session"]["audio"]["input"]["transcription"]["language"] == "en-IN"


def test_final_transcripts_link_consecutive_turns_across_a_provider_error() -> None:
    config: Final = GnaniRealtimeConfig()
    assert _events(config, {"type": "processing"}) == ()
    committed, final = _events(
        config, {"type": "transcript", "text": "one", "segment_id": "seg1", "detected_language_code": "en-IN"}
    )
    assert committed["type"] == "input_audio_buffer.committed"
    assert final["type"] == "conversation.item.input_audio_transcription.completed"
    assert committed["previous_item_id"] is None
    assert committed["item_id"] == final["item_id"] == "seg1"
    assert final["transcript"] == "one"
    assert final.get("detected_language_code") == "en-IN"
    (error,) = _events(config, {"type": "error", "message": "unclear audio"})
    assert error["type"] == "error"
    assert error["error"]["message"] == "unclear audio"
    assert error.get("event_id")
    next_item, next_final = _events(config, {"type": "transcript", "text": "two", "segment_id": "seg2"})
    assert next_item["type"] == "input_audio_buffer.committed"
    assert next_final["type"] == "conversation.item.input_audio_transcription.completed"
    assert next_item["previous_item_id"] == "seg1"
    assert next_item["item_id"] == next_final["item_id"] == "seg2"


def test_rejected_update_keeps_the_partial_audio_for_the_next_commit() -> None:
    config: Final = GnaniRealtimeConfig()
    config.transform_realtime_request(_session_update(16_000, {}), "prisma-v3.0")
    assert config.transform_realtime_request('{"type":"input_audio_buffer.append","audio":"AQI="}', "prisma-v3.0") == ()
    with pytest.raises(RealtimeTranscriptionProtocolError, match="change the audio rate"):
        config.transform_realtime_request(
            '{"type":"session.update","session":{"audio":{"input":{"format":{"type":"audio/pcm","rate":24000}}}}}',
            "prisma-v3.0",
        )
    assert config.transform_realtime_request('{"type":"input_audio_buffer.commit"}', "prisma-v3.0") == (
        b"\x01\x02" + bytes(1022),
        '{"type":"commit"}',
    )
    with pytest.raises(RealtimeTranscriptionProtocolError, match="empty buffer"):
        config.transform_realtime_request('{"type":"input_audio_buffer.commit"}', "prisma-v3.0")


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ({"type": "input_audio_buffer.append", "audio": "%%%"}, "base64"),
        ({"type": "input_audio_buffer.append", "audio": "AQ=="}, "complete samples"),
        ({"type": "session.update", "session": {"input_audio_format": "g711_ulaw"}}, "mono PCM16"),
        ({"type": "session.update", "session": {"turn_detection": None}}, "server_vad"),
        ({"type": "response.create"}, "does not support"),
    ],
)
def test_unsupported_client_events_report_protocol_errors(message: Mapping[str, object], reason: str) -> None:
    with pytest.raises(RealtimeTranscriptionProtocolError, match=reason):
        GnaniRealtimeConfig().transform_realtime_request(json.dumps(message), "prisma-v3.0")


@pytest.mark.parametrize(("rate", "accepted_bytes"), [(8000, 64000), (24000, 192000)])
def test_an_append_longer_than_four_seconds_at_the_client_rate_is_rejected_before_decoding(
    rate: int, accepted_bytes: int
) -> None:
    config: Final = GnaniRealtimeConfig()
    update: Final = {
        "type": "session.update",
        "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": rate}}}},
    }
    config.transform_realtime_request(json.dumps(update), "prisma-v3.0")

    def append(size: int) -> tuple[str | bytes, ...]:
        audio: Final = base64.b64encode(bytes(size)).decode()
        return tuple(
            config.transform_realtime_request(
                json.dumps({"type": "input_audio_buffer.append", "audio": audio}), "prisma-v3.0"
            )
        )

    four_seconds_at_wire_rate: Final = 4 * 32000
    assert four_seconds_at_wire_rate - 1024 <= len(b"".join(append(accepted_bytes))) <= four_seconds_at_wire_rate
    with pytest.raises(RealtimeTranscriptionProtocolError, match="backlog limit"):
        append(accepted_bytes + 4)


def test_realtime_url_preserves_a_deployment_prefix_without_duplicating_the_stream_route() -> None:
    config: Final = GnaniRealtimeConfig()
    assert (
        config.get_complete_url("https://example.test/proxy", "prisma-v3.0") == "wss://example.test/proxy/stt/v3/stream"
    )
    assert (
        config.get_complete_url("wss://example.test/stt/v3/stream", "prisma-v3.0") == "wss://example.test/stt/v3/stream"
    )


@pytest.mark.parametrize("api_base", ["http://example.test", "ws://example.test/stt/v3/stream"])
def test_realtime_url_refuses_a_plaintext_base_that_would_send_the_key_unencrypted(api_base: str) -> None:
    with pytest.raises(ValueError, match="https:// or wss://"):
        GnaniRealtimeConfig().get_complete_url(api_base, "prisma-v3.0")


def _events(config: GnaniRealtimeConfig, payload: Mapping[str, object]) -> tuple[OpenAIRealtimeEvents, ...]:
    result: Final = config.transform_realtime_response(
        json.dumps(payload),
        "prisma-v3.0",
        None,
        {
            "session_configuration_request": None,
            "current_output_item_id": None,
            "current_response_id": None,
            "current_delta_chunks": None,
            "current_conversation_id": None,
            "current_item_chunks": None,
            "current_delta_type": None,
        },
    )["response"]
    return tuple(result) if isinstance(result, list) else (result,)


@pytest.mark.asyncio
async def test_forwarded_pcm_duration_is_reported_once_across_transcripts_and_disconnect() -> None:
    config: Final = GnaniRealtimeConfig()
    inner: Final = Mock(send=AsyncMock(side_effect=(None, None, None, None, ConnectionError("closed"))))
    backend: Final = config.wrap_backend(inner)
    await backend.send(bytes(1024))
    await backend.send('{"type":"commit"}')
    _, first = _events(config, {"type": "transcript", "text": "first"})
    assert _usage(first) == {"type": "duration", "seconds": 0.032}
    await backend.send(bytes(2048))
    _, second = _events(config, {"type": "transcript", "text": "second"})
    assert _usage(second) == {"type": "duration", "seconds": 0.064}
    await backend.send(bytes(1024))
    with pytest.raises(ConnectionError):
        await backend.send(bytes(2048))
    assert config.unbilled_usage_on_session_close("prisma-v3.0") == {"type": "duration", "seconds": 0.032}
    assert config.unbilled_usage_on_session_close("prisma-v3.0") is None


@pytest.mark.asyncio
async def test_each_transcript_bills_its_spoken_duration_and_the_silence_left_bills_at_close() -> None:
    config: Final = GnaniRealtimeConfig()
    backend: Final = config.wrap_backend(Mock(send=AsyncMock()))
    one_second: Final = 2 * 16_000
    await backend.send(bytes(6 * one_second))
    _, first = _events(config, {"type": "transcript", "text": "first", "audio_duration_ms": 3000})
    _, second = _events(config, {"type": "transcript", "text": "second", "audio_duration_ms": 4000})
    await backend.send(bytes(2 * one_second))
    _, third = _events(config, {"type": "transcript", "text": "third", "audio_duration_ms": 500})
    assert _usage(first) == {"type": "duration", "seconds": 3.0}
    assert _usage(second) == {"type": "duration", "seconds": 3.0}
    assert _usage(third) == {"type": "duration", "seconds": 0.5}
    assert config.unbilled_usage_on_session_close("prisma-v3.0") == {"type": "duration", "seconds": 1.5}
    assert config.unbilled_usage_on_session_close("prisma-v3.0") is None


def _usage(event: Mapping[str, object]) -> object:
    return event.get("usage")


@pytest.mark.parametrize("rate", [24_000, 48_000])
def test_repeating_the_session_rate_mid_stream_leaves_the_forwarded_audio_unchanged(rate: int) -> None:
    samples: Final = rate // 5
    tone: Final = struct.pack(
        f"<{samples}h", *(round(10_000 * math.sin(2 * math.pi * 1_000 * i / rate)) for i in range(samples))
    )
    half: Final = len(tone) // 2

    def forwarded(*between: str) -> bytes:
        config: Final = GnaniRealtimeConfig()
        config.transform_realtime_request(_session_update(rate, {}), "prisma-v3.0")
        messages: Final = (_append(tone[:half]), *between, _append(tone[half:]), '{"type":"input_audio_buffer.commit"}')
        return _pcm(
            chain.from_iterable(config.transform_realtime_request(message, "prisma-v3.0") for message in messages)
        )

    assert forwarded(_session_update(rate, {"language": "en"})) == forwarded()


class _OvershootingClock:
    def __init__(self, overshoot: float) -> None:
        self.now: float = 100.0
        self._overshoot: Final = overshoot

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0) + self._overshoot

    def stall(self, seconds: float) -> None:
        self.now += seconds


async def _paced_send_times(stream: GnaniAudioStream, clock: _OvershootingClock, frames: int) -> tuple[float, ...]:
    return tuple([await _paced_send(stream, clock) for _ in range(frames)])


async def _paced_send(stream: GnaniAudioStream, clock: _OvershootingClock) -> float:
    await stream.pace()
    return clock.now


@pytest.mark.asyncio
async def test_pacing_holds_real_time_cadence_when_sleeps_overshoot_and_does_not_burst_after_a_stall() -> None:
    clock: Final = _OvershootingClock(overshoot=0.004)
    stream: Final = GnaniAudioStream(monotonic=clock.monotonic, sleep=clock.sleep)
    streamed: Final = await _paced_send_times(stream, clock, 1_000)
    assert streamed[-1] - streamed[0] == pytest.approx(999 * 0.032, abs=0.005)
    clock.stall(5.0)
    resumed: Final = await _paced_send_times(stream, clock, 3)
    assert resumed[1] - resumed[0] == pytest.approx(0.032, abs=0.001)
    assert resumed[2] - resumed[1] == pytest.approx(0.032, abs=0.001)


@pytest.mark.asyncio
async def test_a_rejected_client_event_reports_an_error_and_the_session_keeps_forwarding() -> None:
    rejected: Final = _session_update(24_000, {}, format={"type": "audio/pcmu"})
    accepted: Final = _session_update(24_000, {"language": "hi"})
    client_ws: Final = Mock(
        headers={}, receive_text=AsyncMock(side_effect=[rejected, accepted, RuntimeError]), send_text=AsyncMock()
    )
    backend_ws: Final = Mock(send=AsyncMock())
    streaming: Final = RealTimeStreaming(
        client_ws, backend_ws, Mock(), provider_config=GnaniRealtimeConfig(), model="prisma-v3.0"
    )

    await asyncio.wait_for(streaming.client_ack_messages(), timeout=2)

    sent_to_client: Final = tuple(json.loads(call.args[0]) for call in client_ws.send_text.await_args_list)
    assert tuple(event["error"]["message"] for event in sent_to_client if event["type"] == "error") == (
        "Gnani realtime requires mono PCM16 audio",
    )
    assert tuple(json.loads(call.args[0]) for call in backend_ws.send.await_args_list) == (
        {"type": "update_settings", "language_code": "hi-IN"},
    )


def test_speech_events_share_the_item_id_of_their_utterance_transcript() -> None:
    config: Final = GnaniRealtimeConfig()
    first_started: Final = _events(config, {"type": "speech_start", "utterance_id": "u1"})
    second_started: Final = _events(config, {"type": "speech_start", "utterance_id": "u2"})
    first: Final = _events(config, {"type": "transcript", "text": "one", "utterance_id": "u1", "segment_id": "s-1"})
    second: Final = _events(config, {"type": "transcript", "text": "two", "utterance_id": "u2", "segment_id": "s-2"})
    unannounced: Final = _events(config, {"type": "transcript", "text": "three", "segment_id": "s-3"})
    assert tuple(map(_type_and_item, first_started + second_started)) == (
        ("input_audio_buffer.speech_started", "u1"),
        ("input_audio_buffer.speech_started", "u2"),
    )
    assert tuple(map(_type_and_item, first + second + unannounced)) == (
        ("input_audio_buffer.speech_stopped", "u1"),
        ("input_audio_buffer.committed", "u1"),
        ("conversation.item.input_audio_transcription.completed", "u1"),
        ("input_audio_buffer.speech_stopped", "u2"),
        ("input_audio_buffer.committed", "u2"),
        ("conversation.item.input_audio_transcription.completed", "u2"),
        ("input_audio_buffer.committed", "s-3"),
        ("conversation.item.input_audio_transcription.completed", "s-3"),
    )


def _type_and_item(event: Mapping[str, object]) -> tuple[object, object]:
    return event.get("type"), event.get("item_id")
