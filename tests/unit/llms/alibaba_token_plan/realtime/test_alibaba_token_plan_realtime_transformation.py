import base64
import datetime
import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import JsonValue

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.alibaba_token_plan.realtime.transformation import AlibabaTokenPlanRealtimeConfig
from litellm.llms.base_llm.realtime.transcription_protocol import RealtimeTranscriptionProtocolError, json_object
from litellm.types.realtime import RealtimeResponseTransformInput
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL: Final = "qwen-audio-3.0-realtime-plus"
EMPTY_STATE: Final[RealtimeResponseTransformInput] = {
    "session_configuration_request": None,
    "current_output_item_id": None,
    "current_response_id": None,
    "current_delta_chunks": None,
    "current_item_chunks": None,
    "current_conversation_id": None,
    "current_delta_type": None,
}


def _logging() -> Logging:
    return Logging(
        model=MODEL,
        messages=[],
        stream=True,
        call_type="arealtime",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="realtime-test",
        function_id="realtime-test",
    )


def _request(
    config: AlibabaTokenPlanRealtimeConfig, event: Mapping[str, object]
) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        json_object(frame)
        for frame in config.transform_realtime_request(json.dumps(event), MODEL)
        if isinstance(frame, str)
    )


def _response(config: AlibabaTokenPlanRealtimeConfig, event: Mapping[str, object]) -> Mapping[str, JsonValue]:
    result: Final = config.transform_realtime_response(json.dumps(event), MODEL, _logging(), EMPTY_STATE)
    return json_object(json.dumps(result["response"]))


def _audio_frames(config: AlibabaTokenPlanRealtimeConfig, audio: bytes) -> tuple[Mapping[str, JsonValue], ...]:
    return _request(config, {"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode()})


def _audio_bytes(events: tuple[Mapping[str, JsonValue], ...]) -> bytes:
    return b"".join(
        base64.b64decode(str(event["audio"])) for event in events if event["type"] == "input_audio_buffer.append"
    )


@pytest.mark.parametrize("path", ["api-ws/v1/realtime", "compatible-mode/v1", "apps/anthropic"])
def test_discovered_provider_uses_websocket_host_path_query_and_dedicated_key(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ALIBABA_TOKEN_PLAN_API_BASE",
        f"https://gateway.example/plan/{path}?tenant=test&model=stale",
    )
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    config: Final = ProviderConfigManager.get_provider_realtime_config(MODEL, LlmProviders.ALIBABA_TOKEN_PLAN)
    assert config is not None
    url: Final = urlsplit(config.get_complete_url(None, MODEL))
    assert (url.scheme, url.netloc, url.path) == ("wss", "gateway.example", "/plan/api-ws/v1/realtime")
    assert parse_qs(url.query) == {"tenant": ["test"], "model": [MODEL]}
    assert config.validate_environment({"x-trace-id": "trace"}, MODEL)["Authorization"] == "Bearer token-plan-key"
    assert config.validate_environment({}, MODEL, api_key="explicit")["Authorization"] == "Bearer explicit"


@pytest.mark.parametrize("scheme", ["http", "ws"])
def test_realtime_rejects_plaintext_official_host_but_allows_local_gateway(scheme: str) -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    with pytest.raises(ValueError, match="require HTTPS or WSS"):
        config.get_complete_url(f"{scheme}://token-plan.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime", MODEL)
    assert config.get_complete_url(f"{scheme}://localhost:8080/token-plan/api-ws/v1/realtime", MODEL) == (
        f"ws://localhost:8080/token-plan/api-ws/v1/realtime?model={MODEL}"
    )


@pytest.mark.parametrize("layout", ("beta", "ga"))
def test_session_translates_formats_modalities_tools_and_manual_control(layout: str) -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    session: Final = {
        "instructions": "Keep answers brief",
        "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}],
        **(
            {
                "input_audio_format": "pcm",
                "output_audio_format": "pcm16",
                "voice": "longanqian",
                "modalities": ["audio", "text"],
                "turn_detection": None,
            }
            if layout == "beta"
            else {
                "type": "realtime",
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 16000}, "turn_detection": None},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "longanqian"},
                },
                "output_modalities": ["audio"],
            }
        ),
    }
    (out,) = _request(config, {"type": "session.update", "session": session})
    assert out["session"] == {
        "instructions": "Keep answers brief",
        "tools": [
            {"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {}}}}
        ],
        "input_audio_format": "pcm",
        "output_audio_format": "pcm",
        "voice": "longanqian",
        "modalities": ["text", "audio"],
        "turn_detection": None,
    }


def test_16khz_audio_passes_through_unchanged() -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    raw: Final = b"\x00\x01\x00\x02" * 320
    encoded: Final = base64.b64encode(raw).decode()
    assert _audio_bytes(_audio_frames(config, raw)) == raw
    (item,) = _request(
        config,
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": encoded}]},
        },
    )
    assert item["item"]["content"][0]["audio"] == encoded


@pytest.mark.parametrize(
    "session",
    (
        {"input_audio_format": "pcm16"},
        {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
        {"audio": {"input": {"format": {"type": "audio/pcm"}}}},
    ),
)
def test_24khz_input_is_rejected_with_a_16khz_hint(session: Mapping[str, object]) -> None:
    with pytest.raises(RealtimeTranscriptionProtocolError, match="16000 Hz"):
        _request(AlibabaTokenPlanRealtimeConfig(), {"type": "session.update", "session": session})


@pytest.mark.parametrize(
    "session",
    (
        {"input_audio_format": "g711_ulaw"},
        {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 48000}}}},
        {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}},
        {"turn_detection": {"type": "semantic_vad"}},
        {"turn_detection": {"type": "server_vad", "create_response": False}},
        {"turn_detection": {"type": "server_vad", "interrupt_response": False}},
        {"tools": [{"type": "mcp"}]},
        {"tools": [{"type": "function", "name": "weather"}], "enable_search": True},
        {"tool_choice": "required"},
        {"temperature": 0.7},
    ),
)
def test_unsupported_capabilities_fail_explicitly(session: Mapping[str, object]) -> None:
    with pytest.raises(RealtimeTranscriptionProtocolError):
        _request(AlibabaTokenPlanRealtimeConfig(), {"type": "session.update", "session": session})


@pytest.mark.parametrize(
    "event",
    (
        {"type": "response.cancel", "event_id": "cancel"},
        {"type": "input_audio_buffer.clear"},
        {"type": "conversation.item.delete", "item_id": "item-1"},
        {"type": "conversation.item.retrieve", "item_id": "item-1"},
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call-1", "output": '{"temperature":18}'},
        },
    ),
)
def test_manual_turn_and_function_result_events_reach_provider(event: Mapping[str, object]) -> None:
    assert _request(AlibabaTokenPlanRealtimeConfig(), event) == (event,)


def test_server_audio_tool_cancel_and_error_events_preserve_payloads() -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    audio: Final = base64.b64encode(b"\x11\x22" * 160).decode()
    assert _response(config, {"type": "response.audio.delta", "delta": audio, "item_id": "voice"}) == {
        "type": "response.output_audio.delta",
        "delta": audio,
        "item_id": "voice",
    }
    tool: Final = {"type": "response.function_call_arguments.delta", "call_id": "call-1", "delta": '{"city":'}
    assert _response(config, tool) == tool
    cancelled: Final = {
        "type": "response.done",
        "response": {"id": "r1", "status": "cancelled", "status_details": {"reason": "turn_detected"}, "output": []},
    }
    assert _response(config, cancelled) == cancelled
    error: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": "No active response"}}
    assert _response(config, error) == error
    session: Final = _response(
        config,
        {
            "type": "session.created",
            "session": {"id": "session", "model": MODEL, "modalities": ["audio", "text"], "voice": "longanqian"},
        },
    )
    assert session["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 16000}
    assert session["session"]["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}


@pytest.mark.asyncio
@pytest.mark.parametrize("beta", (False, True))
async def test_injected_websockets_relay_session_audio_barge_in_and_tool_events(beta: bool) -> None:
    client: Final = AsyncMock()
    client.scope = {"headers": [(b"openai-beta", b"realtime=v1")] if beta else []}
    backend: Final = AsyncMock()
    config: Final = AlibabaTokenPlanRealtimeConfig()
    streaming: Final = RealTimeStreaming(client, backend, _logging(), config, MODEL)
    for event in (
        {
            "type": "session.created",
            "session": {"id": "session", "model": MODEL, "modalities": ["audio", "text"], "voice": "longanqian"},
        },
        {"type": "response.audio.delta", "delta": "AAAA", "item_id": "item"},
        {"type": "input_audio_buffer.speech_started", "item_id": "user", "audio_start_ms": 10},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "user", "transcript": "Hello"},
        {"type": "response.function_call_arguments.done", "call_id": "call", "arguments": "{}", "name": "weather"},
        {
            "type": "response.done",
            "response": {
                "id": "r1",
                "status": "cancelled",
                "status_details": {"reason": "turn_detected"},
                "output": [],
            },
        },
    ):
        await streaming._handle_provider_config_message(json.dumps(event))
    events: Final = tuple(json.loads(call.args[0]) for call in client.send_text.await_args_list)
    assert events[0]["type"] == "session.created"
    assert events[1]["type"] == ("response.audio.delta" if beta else "response.output_audio.delta")
    assert events[1]["delta"] == "AAAA"
    assert events[2]["type"] == "input_audio_buffer.speech_started"
    assert events[4]["call_id"] == "call"
    assert events[5]["response"]["status"] == "cancelled"
    backend.send.assert_awaited_once_with(json.dumps({"type": "response.create"}))


def test_cumulative_transcription_emits_only_new_confirmed_suffix_per_item() -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    first: Final = _response(
        config,
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "one",
            "content_index": 0,
            "text": "Hello",
            "stash": "there",
        },
    )
    second: Final = _response(
        config,
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "one",
            "content_index": 0,
            "text": "Hello world",
            "stash": "today",
        },
    )
    repeat: Final = _response(
        config,
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "one",
            "content_index": 0,
            "text": "Hello world",
            "stash": "tomorrow",
        },
    )
    other: Final = _response(
        config,
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "two",
            "content_index": 0,
            "text": "Hello",
        },
    )
    completed: Final = _response(
        config,
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "one",
            "content_index": 0,
            "transcript": "Hello world!",
        },
    )
    assert first["delta"] == "Hello"
    assert second["delta"] == " world"
    assert repeat["delta"] == ""
    assert other["delta"] == "Hello"
    assert completed["transcript"] == "Hello world!"
    assert second["stash"] == "today"


@pytest.mark.parametrize(
    "event",
    (
        {"type": "conversation.item.truncate", "item_id": "item", "content_index": 0, "audio_end_ms": 100},
        {"type": "response.cancel", "response_id": "old-response"},
        {"type": "response.create", "response": {"conversation": "none"}},
    ),
)
def test_unsupported_conversation_operations_are_not_silently_ignored(event: Mapping[str, object]) -> None:
    with pytest.raises(RealtimeTranscriptionProtocolError):
        _request(AlibabaTokenPlanRealtimeConfig(), event)


def test_provider_session_tools_audio_parts_and_usage_are_openai_shaped() -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    session: Final = _response(
        config,
        {
            "type": "session.updated",
            "session": {
                "id": "session",
                "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            },
        },
    )
    assert session["session"]["tools"] == [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]
    part: Final = _response(config, {"type": "response.content_part.done", "part": {"type": "audio", "text": "Hello"}})
    assert part["part"]["type"] == "audio"
    assert part["part"]["transcript"] == "Hello"
    done: Final = _response(
        config,
        {
            "type": "response.done",
            "response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 15,
                    "total_tokens": 25,
                    "input_tokens_details": {"audio_tokens": 9, "text_tokens": 1},
                    "output_tokens_details": {"audio_tokens": 12, "text_tokens": 3},
                }
            },
        },
    )
    assert done["response"]["usage"]["input_token_details"]["audio_tokens"] == 9
    assert done["response"]["usage"]["output_token_details"]["audio_tokens"] == 12
    assert done["response"]["usage"]["total_tokens"] == 25


def test_ga_response_audio_settings_and_item_content_preserve_content_part_types() -> None:
    config: Final = AlibabaTokenPlanRealtimeConfig()
    result: Final = _response(
        config,
        {
            "type": "response.done",
            "response": {
                "id": "response",
                "modalities": ["text", "audio"],
                "voice": "longanqian",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "audio", "transcript": "Hello"}]}
                ],
            },
        },
    )
    assert result["response"]["output_modalities"] == ["audio"]
    assert result["response"]["audio"]["output"]["voice"] == "longanqian"
    assert result["response"]["output"][0]["content"][0]["type"] == "output_audio"
    session: Final = _response(config, {"type": "session.created", "session": {"id": "session", "voice": "longanqian"}})
    assert session["session"]["input_audio_format"] == "pcm"
    assert session["session"]["modalities"] == ["text", "audio"]
