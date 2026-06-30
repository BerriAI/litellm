import base64
import json
import struct
from unittest.mock import MagicMock

from litellm.llms.volcengine.realtime.protocol import (
    COMP_NONE,
    FLAG_EVENT,
    EV_ASR_ENDED,
    EV_ASR_INFO,
    EV_ASR_RESPONSE,
    EV_CHAT_RESPONSE,
    EV_CONFIG_UPDATED,
    MSG_AUDIO_CLIENT,
    MSG_AUDIO_SERVER,
    MSG_FULL_SERVER,
    SER_JSON,
    SER_RAW,
    EV_SAY_HELLO,
    EV_CONNECTION_STARTED,
    EV_SESSION_FINISHED,
    EV_SESSION_STARTED,
    EV_START_CONNECTION,
    EV_TASK_REQUEST,
    EV_TTS_ENDED,
    EV_TTS_RESPONSE,
    EV_UPDATE_CONFIG,
    decode_realtime_frame,
    encode_audio_event,
    encode_event_frame,
    parse_json_payload,
    parse_payload,
)
from litellm.llms.volcengine.realtime.transformation import (
    VOLCENGINE_REALTIME_DEFAULT_APP_KEY,
    VOLCENGINE_REALTIME_DEFAULT_API_BASE,
    VOLCENGINE_REALTIME_DEFAULT_BOT_NAME,
    VOLCENGINE_REALTIME_DEFAULT_END_SMOOTH_WINDOW_MS,
    VOLCENGINE_REALTIME_DEFAULT_MODEL_VERSION,
    VOLCENGINE_REALTIME_DEFAULT_SPEAKER,
    VolcEngineRealtimeConfig,
    pick_realtime_resource_id,
)


def _server_event(event: int, payload=None, session_id=None) -> bytes:
    body = json.dumps(payload or {}).encode("utf-8")
    return encode_event_frame(
        message_type=MSG_FULL_SERVER,
        flags=FLAG_EVENT,
        serialization=SER_JSON,
        compression=COMP_NONE,
        event=event,
        session_id=session_id,
        payload=body,
    )


def _server_audio(payload: bytes, session_id="session-1") -> bytes:
    return encode_event_frame(
        message_type=MSG_AUDIO_SERVER,
        flags=FLAG_EVENT,
        serialization=SER_RAW,
        compression=COMP_NONE,
        event=EV_TTS_RESPONSE,
        session_id=session_id,
        payload=payload,
    )


def _empty_transform_input(**overrides):
    base = {
        "session_configuration_request": None,
        "current_output_item_id": None,
        "current_response_id": None,
        "current_delta_chunks": None,
        "current_conversation_id": None,
        "current_item_chunks": None,
        "current_delta_type": None,
    }
    base.update(overrides)
    return base


def test_realtime_event_frame_round_trip():
    payload = b"\x01\x02\x03"
    frame = decode_realtime_frame(
        encode_audio_event(
            event=EV_TASK_REQUEST,
            session_id="session-1",
            payload=payload,
        )
    )

    assert frame.message_type == MSG_AUDIO_CLIENT
    assert frame.event == EV_TASK_REQUEST
    assert frame.session_id == "session-1"
    assert frame.compression == COMP_NONE
    assert parse_payload(frame) == payload


def test_realtime_connection_setup_frame_is_uncompressed_json():
    config = VolcEngineRealtimeConfig()
    frame = decode_realtime_frame(
        config.session_configuration_request("volc.speech.dialog")
    )  # type: ignore[arg-type]

    assert frame.event == EV_START_CONNECTION
    assert frame.compression == COMP_NONE
    assert parse_json_payload(frame) == {}


def test_realtime_environment_prefers_specific_speech_key(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_API_KEY", "ark-chat-key")
    monkeypatch.setenv("VOLCENGINE_SPEECH_KEY", "speech-key")
    config = VolcEngineRealtimeConfig()

    headers = config.validate_environment(
        headers={"Authorization": "Bearer litellm-key"},
        model="volc.speech.dialog",
        api_key="ark-chat-key",
    )

    assert headers["X-Api-Key"] == "speech-key"
    assert headers["X-Api-Resource-Id"] == "volc.speech.dialog"
    assert headers["X-Api-App-Key"] == VOLCENGINE_REALTIME_DEFAULT_APP_KEY
    assert headers["X-Api-Connect-Id"]
    assert "Authorization" not in headers


def test_realtime_environment_prefers_specific_realtime_api_key(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_API_KEY", "ark-chat-key")
    monkeypatch.setenv("VOLCENGINE_SPEECH_KEY", "speech-key")
    monkeypatch.setenv("VOLCENGINE_REALTIME_API_KEY", "realtime-key")

    headers = VolcEngineRealtimeConfig().validate_environment(
        headers={},
        model="volc.speech.dialog",
        api_key="ark-chat-key",
    )

    assert headers["X-Api-Key"] == "realtime-key"


def test_realtime_environment_supports_official_app_credentials(monkeypatch):
    monkeypatch.delenv("VOLCENGINE_SPEECH_KEY", raising=False)
    monkeypatch.setenv("VOLCENGINE_REALTIME_APP_ID", "app-id")
    monkeypatch.setenv("VOLCENGINE_REALTIME_ACCESS_KEY", "access-key")
    monkeypatch.setenv("VOLCENGINE_REALTIME_APP_KEY", "app-key")

    headers = VolcEngineRealtimeConfig().validate_environment(
        headers={},
        model="volcengine/volc.speech.dialog",
        api_key=None,
    )

    assert headers["X-Api-App-ID"] == "app-id"
    assert headers["X-Api-Access-Key"] == "access-key"
    assert headers["X-Api-App-Key"] == "app-key"
    assert headers["X-Api-Resource-Id"] == "volc.speech.dialog"


def test_realtime_environment_preserves_explicit_x_api_headers(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_SPEECH_KEY", "speech-key")

    headers = VolcEngineRealtimeConfig().validate_environment(
        headers={
            "x-api-resource-id": "custom.resource",
            "x-api-app-key": "custom-app-key",
            "x-api-connect-id": "connect-id",
        },
        model="volcengine/volc.speech.dialog",
        api_key=None,
    )

    assert headers["x-api-resource-id"] == "custom.resource"
    assert headers["x-api-app-key"] == "custom-app-key"
    assert headers["x-api-connect-id"] == "connect-id"
    assert headers["X-Api-Key"] == "speech-key"


def test_realtime_url_and_resource_id_mapping():
    config = VolcEngineRealtimeConfig()

    assert config.get_complete_url(None, "volc.speech.dialog") == (
        VOLCENGINE_REALTIME_DEFAULT_API_BASE
    )
    assert (
        config.get_complete_url("wss://example.test/realtime", "volc.speech.dialog")
        == "wss://example.test/realtime"
    )
    assert pick_realtime_resource_id("volcengine/volc.speech.dialog") == (
        "volc.speech.dialog"
    )


def test_session_update_maps_to_start_session_frame():
    config = VolcEngineRealtimeConfig()

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "instructions": "Be concise.",
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice": "zh_female_vv_uranus_bigtts",
                        },
                    },
                    "volcengine": {"dialog": {"bot_name": "LiteLLM"}},
                },
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert len(frames) == 1
    frame = decode_realtime_frame(frames[0])  # type: ignore[arg-type]
    assert frame.event == 100
    assert frame.session_id
    payload = parse_json_payload(frame)
    assert payload["dialog"]["system_role"] == "Be concise."
    assert payload["dialog"]["bot_name"] == "LiteLLM"
    assert (
        payload["dialog"]["extra"]["model"] == VOLCENGINE_REALTIME_DEFAULT_MODEL_VERSION
    )
    assert payload["asr"]["extra"]["end_smooth_window_ms"] == (
        VOLCENGINE_REALTIME_DEFAULT_END_SMOOTH_WINDOW_MS
    )
    assert payload["asr"]["audio_info"] == {
        "format": "pcm",
        "sample_rate": 16000,
        "channel": 1,
    }
    assert payload["tts"]["speaker"] == "zh_female_vv_uranus_bigtts"
    assert payload["tts"]["audio_config"]["format"] == "pcm_s16le"
    assert payload["tts"]["audio_config"]["sample_rate"] == 24000


def test_session_update_allows_volcengine_overrides():
    config = VolcEngineRealtimeConfig()

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "volcengine": {
                        "tts": {
                            "speaker": "custom-speaker",
                            "extra": {"explicit_dialect": "sichuan"},
                        },
                        "asr": {"extra": {"end_smooth_window_ms": 900}},
                        "dialog": {"extra": {"model": "2.2.0.0"}},
                    }
                },
            }
        ),
        model="volcengine/volc.speech.dialog",
        session_configuration_request=None,
    )

    payload = parse_json_payload(decode_realtime_frame(frames[0]))  # type: ignore[arg-type]
    assert payload["tts"]["speaker"] == "custom-speaker"
    assert payload["tts"]["extra"]["explicit_dialect"] == "sichuan"
    assert payload["asr"]["extra"]["end_smooth_window_ms"] == 900
    assert payload["dialog"]["extra"]["model"] == "2.2.0.0"


def test_session_update_ignores_openai_default_voice_for_volcengine_speaker():
    config = VolcEngineRealtimeConfig()

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {"audio": {"output": {"voice": "marin"}}},
            }
        ),
        model="volcengine/volc.speech.dialog",
        session_configuration_request=None,
    )

    payload = parse_json_payload(decode_realtime_frame(frames[0]))  # type: ignore[arg-type]
    assert payload["tts"]["speaker"] == VOLCENGINE_REALTIME_DEFAULT_SPEAKER


def test_session_update_after_start_updates_dialog_system_role():
    config = VolcEngineRealtimeConfig()
    config.transform_realtime_request(
        json.dumps({"type": "session.update", "session": {}}),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": "You are a concise assistant.",
                },
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert len(frames) == 1
    frame = decode_realtime_frame(frames[0])  # type: ignore[arg-type]
    assert frame.event == EV_UPDATE_CONFIG
    payload = parse_json_payload(frame)
    assert payload == {
        "dialog": {
            "bot_name": VOLCENGINE_REALTIME_DEFAULT_BOT_NAME,
            "system_role": "You are a concise assistant.",
            "speaking_style": "",
        }
    }


def test_session_update_after_start_merges_partial_updates():
    config = VolcEngineRealtimeConfig()
    config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "instructions": "Original instructions.",
                    "volcengine": {
                        "dialog": {"bot_name": "LiteLLM"},
                        "tts": {"extra": {"explicit_dialect": "sichuan"}},
                    },
                },
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {"instructions": "Updated instructions."},
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    payload = parse_json_payload(decode_realtime_frame(frames[0]))  # type: ignore[arg-type]
    assert payload["dialog"]["bot_name"] == "LiteLLM"
    assert payload["dialog"]["system_role"] == "Updated instructions."


def test_session_update_after_start_ignores_tools_only_update():
    config = VolcEngineRealtimeConfig()
    config.transform_realtime_request(
        json.dumps({"type": "session.update", "session": {}}),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "volc.speech.dialog",
                    "tools": [{"type": "function", "name": "end_call"}],
                },
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert frames == []


def test_audio_append_maps_to_volcengine_audio_task_frame():
    config = VolcEngineRealtimeConfig()
    config.transform_realtime_request(
        json.dumps({"type": "session.update", "session": {}}),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )
    audio = b"\x01\x00\x02\x00"

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert len(frames) == 1
    frame = decode_realtime_frame(frames[0])  # type: ignore[arg-type]
    assert frame.message_type == MSG_AUDIO_CLIENT
    assert frame.event == EV_TASK_REQUEST
    assert frame.compression == COMP_NONE
    assert parse_payload(frame) == audio


def test_audio_append_resamples_openai_24khz_pcm16_to_volcengine_16khz():
    config = VolcEngineRealtimeConfig()
    config.transform_realtime_request(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}
                },
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )
    audio = struct.pack("<hhh", 0, 6000, 12000)

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    frame = decode_realtime_frame(frames[0])  # type: ignore[arg-type]
    assert parse_payload(frame) == struct.pack("<hh", 0, 9000)


def test_response_create_with_instructions_maps_to_say_hello():
    config = VolcEngineRealtimeConfig()
    config.transform_realtime_request(
        json.dumps({"type": "session.update", "session": {}}),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "response.create",
                "response": {"instructions": "你好，请简单打个招呼。"},
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert len(frames) == 1
    frame = decode_realtime_frame(frames[0])  # type: ignore[arg-type]
    assert frame.event == EV_SAY_HELLO
    assert parse_json_payload(frame) == {"content": "你好，请简单打个招呼。"}


def test_response_create_without_instructions_is_not_forwarded():
    config = VolcEngineRealtimeConfig()

    frames = config.transform_realtime_request(
        json.dumps({"type": "response.create"}),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert frames == []


def test_response_create_metadata_is_echoed_on_next_response_created():
    config = VolcEngineRealtimeConfig()

    frames = config.transform_realtime_request(
        json.dumps(
            {
                "type": "response.create",
                "response": {"metadata": {"client_event_id": "response_create_test"}},
            }
        ),
        model="volc.speech.dialog",
        session_configuration_request=None,
    )

    assert frames == []

    transformed = config.transform_realtime_response(
        _server_audio(b"\x01\x00\x02\x00"),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    assert transformed["response"][0]["type"] == "response.created"
    assert transformed["response"][0]["response"]["metadata"] == {
        "client_event_id": "response_create_test"
    }


def test_connection_and_session_events_map_to_openai_session_events():
    config = VolcEngineRealtimeConfig()

    created = config.transform_realtime_response(
        _server_event(EV_CONNECTION_STARTED),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )
    updated = config.transform_realtime_response(
        _server_event(EV_SESSION_STARTED),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )
    config_updated = config.transform_realtime_response(
        _server_event(EV_CONFIG_UPDATED),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    assert created["response"][0]["type"] == "session.created"
    assert updated["response"][0]["type"] == "session.updated"
    assert config_updated["response"][0]["type"] == "session.updated"


def test_server_audio_frame_maps_to_openai_audio_delta_and_done():
    config = VolcEngineRealtimeConfig()
    pcm16_bytes_that_look_like_float32 = struct.pack("<ff", 0.5, -0.5)

    first = config.transform_realtime_response(
        _server_audio(pcm16_bytes_that_look_like_float32),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    events = first["response"]
    assert isinstance(events, list)
    assert [event["type"] for event in events[:4]] == [
        "response.created",
        "response.output_item.added",
        "conversation.item.added",
        "response.content_part.added",
    ]
    assert events[1]["item"]["content"][0]["type"] == "output_audio"
    assert events[3]["part"]["type"] == "output_audio"
    assert events[-1]["type"] == "response.output_audio.delta"
    delta_audio = base64.b64decode(events[-1]["delta"])
    delta_samples = struct.unpack("<hhhh", delta_audio)
    assert delta_samples == (0, 34, 0, -104)

    done = config.transform_realtime_response(
        _server_event(EV_SESSION_FINISHED, session_id="session-1"),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(
            current_output_item_id=first["current_output_item_id"],
            current_response_id=first["current_response_id"],
            current_conversation_id=first["current_conversation_id"],
            current_delta_type=first["current_delta_type"],
        ),
    )

    done_events = done["response"]
    assert isinstance(done_events, list)
    assert [event["type"] for event in done_events] == [
        "response.output_audio.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    assert done_events[-1]["response"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert done_events[1]["part"]["type"] == "output_audio"
    assert done_events[2]["item"]["content"][0]["type"] == "output_audio"
    for event in done_events:
        json.dumps(event)


def test_server_audio_startup_suppresses_full_scale_sentinel_and_fades_in():
    config = VolcEngineRealtimeConfig()
    provider_audio = struct.pack(
        "<hhhhhhhhh",
        0,
        0,
        -32768,
        -32768,
        1000,
        1000,
        1000,
        1000,
        1000,
    )

    transformed = config.transform_realtime_response(
        _server_audio(provider_audio),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    delta = transformed["response"][-1]["delta"]
    smoothed = base64.b64decode(delta)
    samples = struct.unpack(f"<{len(smoothed) // 2}h", smoothed)

    assert samples[:4] == (0, 0, 0, 0)
    assert 0 < samples[4] < samples[5] < samples[6] < 1000


def test_tts_ended_closes_openai_audio_response_before_session_finished():
    config = VolcEngineRealtimeConfig()
    first = config.transform_realtime_response(
        _server_audio(b"\x01\x00\x02\x00"),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    done = config.transform_realtime_response(
        _server_event(EV_TTS_ENDED, session_id="session-1"),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(
            current_output_item_id=first["current_output_item_id"],
            current_response_id=first["current_response_id"],
            current_conversation_id=first["current_conversation_id"],
            current_delta_type=first["current_delta_type"],
        ),
    )

    assert done["current_output_item_id"] is None
    assert done["current_response_id"] is None
    assert [event["type"] for event in done["response"]] == [
        "response.output_audio.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    assert done["response"][-1]["response"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for event in done["response"]:
        json.dumps(event)


def test_chat_response_maps_to_assistant_audio_transcript_not_user_stt():
    config = VolcEngineRealtimeConfig()

    transformed = config.transform_realtime_response(
        _server_event(EV_CHAT_RESPONSE, {"content": "你好呀"}),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    event_types = [event["type"] for event in transformed["response"]]
    assert "conversation.item.input_audio_transcription.completed" not in event_types
    assert event_types == [
        "response.created",
        "response.output_item.added",
        "conversation.item.added",
        "response.content_part.added",
        "response.output_audio_transcript.delta",
    ]
    assert transformed["response"][-1]["delta"] == "你好呀"

    done = config.transform_realtime_response(
        _server_event(EV_TTS_ENDED, session_id="session-1"),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(
            current_output_item_id=transformed["current_output_item_id"],
            current_response_id=transformed["current_response_id"],
            current_conversation_id=transformed["current_conversation_id"],
            current_delta_type=transformed["current_delta_type"],
        ),
    )

    assert done["response"][1]["part"]["transcript"] == "你好呀"
    assert done["response"][2]["item"]["content"][0]["transcript"] == "你好呀"


def test_asr_info_maps_to_openai_speech_started_event():
    config = VolcEngineRealtimeConfig()

    started = config.transform_realtime_response(
        _server_event(EV_ASR_INFO, {"question_id": "question-1"}),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    event = started["response"][0]
    assert event["type"] == "input_audio_buffer.speech_started"
    assert event["audio_start_ms"] == 0
    assert event["item_id"].startswith("item_")

    config.transform_realtime_response(
        _server_event(
            EV_ASR_RESPONSE,
            {"results": [{"text": "你好", "is_interim": False}]},
        ),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )
    ended = config.transform_realtime_response(
        _server_event(EV_ASR_ENDED),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    assert ended["response"][0]["type"] == (
        "conversation.item.input_audio_transcription.completed"
    )
    assert ended["response"][0]["item_id"] == event["item_id"]


def test_asr_response_emits_user_transcript_only_after_asr_ended():
    config = VolcEngineRealtimeConfig()

    interim = config.transform_realtime_response(
        _server_event(
            EV_ASR_RESPONSE,
            {"results": [{"text": "喂", "is_interim": True}]},
        ),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )
    final_piece = config.transform_realtime_response(
        _server_event(
            EV_ASR_RESPONSE,
            {"results": [{"text": "你好", "is_interim": False}]},
        ),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )
    ended = config.transform_realtime_response(
        _server_event(EV_ASR_ENDED),
        model="volc.speech.dialog",
        logging_obj=MagicMock(),
        realtime_response_transform_input=_empty_transform_input(),
    )

    assert interim["response"] == []
    assert final_piece["response"] == []
    assert ended["response"][0]["type"] == (
        "conversation.item.input_audio_transcription.completed"
    )
    assert ended["response"][0]["transcript"] == "你好"
