import base64
import json
import struct
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

from litellm.llms.base_llm.realtime.transcription_protocol import RealtimeTranscriptionProtocolError
from litellm.llms.gnani.realtime.transformation import GnaniRealtimeConfig
from litellm.types.llms.openai import OpenAIRealtimeEvents
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize("model", ["prisma-v2.5", "prisma-v3.0"])
def test_client_pcm_is_resampled_and_framed_across_appends(model: str) -> None:
    config: Final = ProviderConfigManager.get_provider_realtime_config(model, LlmProviders.GNANI)
    assert config is not None
    samples: Final = struct.pack("<768h", *range(0, 1536, 2))

    def append(pcm: bytes) -> tuple[str | bytes, ...]:
        message: Final = json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        return tuple(config.transform_realtime_request(message, model))

    assert append(samples[:10]) == ()
    assert append(samples[10:]) == (struct.pack("<512h", *range(0, 1536, 3)),)
    assert tuple(config.transform_realtime_request('{"type":"input_audio_buffer.commit"}', model)) == (
        '{"type":"commit"}',
    )


def test_unknown_gnani_model_cannot_fall_back_to_another_prisma_version() -> None:
    assert ProviderConfigManager.get_provider_realtime_config("unknown-prisma", LlmProviders.GNANI) is None


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
    update: Final = '{"type":"transcription_session.update","session":{"input_audio_format":"pcm16"}}'
    config.transform_realtime_request(update, "prisma-v3.0")
    assert config.transform_realtime_request('{"type":"input_audio_buffer.append","audio":"AQI="}', "prisma-v3.0") == ()
    with pytest.raises(RealtimeTranscriptionProtocolError, match="change the audio rate"):
        config.transform_realtime_request(
            '{"type":"session.update","session":{"audio":{"input":{"format":{"type":"audio/pcm","rate":16000}}}}}',
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
        config.get_complete_url("ws://example.test/stt/v3/stream", "prisma-v3.0") == "ws://example.test/stt/v3/stream"
    )


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


def _usage(event: Mapping[str, object]) -> object:
    return event.get("usage")
