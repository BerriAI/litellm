import asyncio
import base64
import itertools
import json
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Final, get_args
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs

import pytest

import litellm
from litellm.llms.base_llm.realtime.transformation import RealtimeBackend
from litellm.llms.elevenlabs.common_utils import ElevenLabsException
from litellm.llms.elevenlabs.realtime.backend import ElevenLabsRealtimeBackend
from litellm.llms.elevenlabs.realtime.transformation import (
    DEFAULT_ELEVENLABS_REALTIME_URL,
    MAX_AUDIO_CHUNK_BYTES,
    ElevenLabsRealtimeConfig,
    build_elevenlabs_realtime_url,
    elevenlabs_realtime_config,
    SCRIBE_REALTIME_MODEL,
    SUPPORTED_SAMPLE_RATES,
    ElevenLabsEventTransformer,
    ElevenLabsProtocolError,
    ElevenLabsSessionConfig,
    default_session_config,
    elevenlabs_realtime_health_check,
    normalize_language,
    parse_server_message,
    parse_session_update,
    realtime_options_from_params,
    session_created_event,
)
from litellm.types.llms.elevenlabs import ElevenLabsErrorType, ElevenLabsRealtimeConfigure, ElevenLabsRealtimeOptions
from litellm.types.realtime import RealtimeResponseTransformInput
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

NO_OPTIONS: Final = ElevenLabsRealtimeOptions()


def _session_update(
    rate: int = 24_000,
    turn_detection: str | None = "server_vad",
    language: str | None = None,
    model: str = SCRIBE_REALTIME_MODEL,
    audio_type: str = "audio/pcm",
    channels: int | None = None,
) -> str:
    audio_format: Final = {"type": audio_type, "rate": rate} | ({} if channels is None else {"channels": channels})
    transcription: Final = {"model": model} | ({} if language is None else {"language": language})
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": audio_format,
                        "turn_detection": None if turn_detection is None else {"type": turn_detection},
                        "transcription": transcription,
                    }
                },
            },
        }
    )


def _query(config: ElevenLabsSessionConfig) -> dict[str, list[str]]:
    return parse_qs(config.query())


@pytest.mark.parametrize("rate", sorted(SUPPORTED_SAMPLE_RATES))
def test_session_rate_selects_the_matching_pcm_audio_format(rate: int):
    config: Final = parse_session_update(_session_update(rate=rate), NO_OPTIONS)

    assert _query(config)["audio_format"] == [f"pcm_{rate}"]
    assert config.bytes_per_second == rate * 2


def test_server_vad_selects_vad_and_null_turn_detection_selects_manual_commit():
    vad: Final = parse_session_update(_session_update(turn_detection="server_vad"), NO_OPTIONS)
    manual: Final = parse_session_update(_session_update(turn_detection=None), NO_OPTIONS)

    assert _query(vad)["commit_strategy"] == ["vad"]
    assert _query(manual)["commit_strategy"] == ["manual"]


def test_semantic_vad_falls_back_to_vad_only_under_drop_params():
    with pytest.raises(ElevenLabsProtocolError, match="drop_params"):
        parse_session_update(_session_update(turn_detection="semantic_vad"), NO_OPTIONS)

    fallback: Final = parse_session_update(_session_update(turn_detection="semantic_vad"), NO_OPTIONS, drop_params=True)
    assert _query(fallback)["commit_strategy"] == ["vad"]


@pytest.mark.parametrize("model", [SCRIBE_REALTIME_MODEL, f"elevenlabs/{SCRIBE_REALTIME_MODEL}"])
def test_query_pins_the_scribe_realtime_model(model: str):
    assert _query(parse_session_update(_session_update(model=model), NO_OPTIONS))["model_id"] == [SCRIBE_REALTIME_MODEL]


def test_beta_pcm16_session_uses_the_default_rate_and_vad():
    beta: Final = json.dumps(
        {
            "type": "session.update",
            "session": {"input_audio_format": "pcm16", "input_audio_transcription": {"model": SCRIBE_REALTIME_MODEL}},
        }
    )
    query: Final = _query(parse_session_update(beta, NO_OPTIONS))

    assert query["audio_format"] == ["pcm_24000"]
    assert query["commit_strategy"] == ["vad"]


@pytest.mark.parametrize(("source", "expected"), [("en", "en"), ("EN-us", "en"), ("pt_BR", "pt"), ("fil", "fil")])
def test_language_becomes_an_iso_language_code(source: str, expected: str):
    assert normalize_language(source) == expected
    assert _query(parse_session_update(_session_update(language=source), NO_OPTIONS))["language_code"] == [expected]


@pytest.mark.parametrize("language", ["", " ", "english", "e1"])
def test_invalid_language_is_rejected(language: str):
    with pytest.raises(ElevenLabsProtocolError, match="ISO 639"):
        normalize_language(language)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (_session_update(audio_type="audio/pcmu"), "audio/pcm"),
        (_session_update(channels=2), "mono"),
        (_session_update(rate=11_025), "8000, 16000"),
        (_session_update(model="gpt-4o-transcribe"), "cannot be changed"),
        (json.dumps({"type": "session.update", "session": {"type": "conversation"}}), "transcription sessions only"),
    ],
)
def test_session_rejects_unsupported_settings(update: str, message: str):
    with pytest.raises(ElevenLabsProtocolError, match=message):
        parse_session_update(update, NO_OPTIONS)


def test_options_are_appended_to_the_query():
    options: Final = ElevenLabsRealtimeOptions(
        keyterms=("Scribe", "realtime"),
        secondary_languages=("es",),
        no_verbatim=True,
        filter_background_audio=False,
        vad_threshold=0.4,
        vad_silence_threshold_secs=1.2,
        min_speech_duration_ms=100,
        min_silence_duration_ms=250,
        enable_logging=False,
    )
    query: Final = _query(parse_session_update(_session_update(), options))

    assert query["keyterms"] == ["Scribe", "realtime"]
    assert query["secondary_languages"] == ["es"]
    assert query["no_verbatim"] == ["true"]
    assert query["filter_background_audio"] == ["false"]
    assert query["vad_threshold"] == ["0.4"]
    assert query["vad_silence_threshold_secs"] == ["1.2"]
    assert query["min_speech_duration_ms"] == ["100"]
    assert query["min_silence_duration_ms"] == ["250"]
    assert query["enable_logging"] == ["false"]


def test_unset_options_are_not_sent():
    assert set(_query(parse_session_update(_session_update(), NO_OPTIONS))) == {
        "model_id",
        "audio_format",
        "commit_strategy",
    }


def test_options_from_params_ignore_unrelated_litellm_params():
    options: Final = realtime_options_from_params(
        {"api_key": "secret", "rpm": 10, "keyterms": ["Scribe"], "no_verbatim": True, "vad_threshold": None}
    )

    assert options == ElevenLabsRealtimeOptions(keyterms=("Scribe",), no_verbatim=True)


@pytest.mark.parametrize(
    "params",
    [
        {"keyterms": "Scribe"},
        {"no_verbatim": "yes"},
        {"min_speech_duration_ms": -1},
        {"vad_threshold": -0.1},
        {"vad_threshold": float("inf")},
        {"vad_silence_threshold_secs": float("nan")},
    ],
)
def test_options_from_params_reject_invalid_values(params: dict[str, object]):
    with pytest.raises(ValueError, match="Invalid ElevenLabs realtime options"):
        realtime_options_from_params(params)


def test_session_created_event_reports_the_effective_transcription_session():
    config: Final = parse_session_update(_session_update(rate=16_000, turn_detection=None, language="es"), NO_OPTIONS)
    event: Final = session_created_event(config, "sess_1")

    assert event["type"] == "session.created"
    assert event["session"]["id"] == "sess_1"
    audio_input: Final = event["session"]["audio"]["input"]
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 16_000}
    assert audio_input["transcription"] == {"model": SCRIBE_REALTIME_MODEL, "language": "es"}
    assert audio_input["turn_detection"] is None


DEFAULT_CONFIG: Final = default_session_config(NO_OPTIONS)


def _item_ids() -> Callable[[], str]:
    counter: Final = itertools.count()
    return lambda: f"item_{next(counter)}"


def _frame(message_type: str, **fields: object) -> str:
    return json.dumps({"message_type": message_type, **fields})


def _transform(transformer: ElevenLabsEventTransformer, frame: str) -> tuple[object, ...]:
    message: Final = parse_server_message(frame)
    return () if message is None else transformer.transform(message, DEFAULT_CONFIG)


def _events(transformer: ElevenLabsEventTransformer, *frames: str) -> list[dict[str, object]]:
    transformed: Final = itertools.chain.from_iterable(_transform(transformer, frame) for frame in frames)
    return [dict(event) for event in transformed]


def _summary(events: list[dict[str, object]]) -> list[tuple[object, ...]]:
    return [(event["type"], event.get("item_id"), event.get("delta", event.get("transcript"))) for event in events]


def test_live_partials_with_trailing_word_revisions_continue_emitting_stable_words():
    # Captured from the live ElevenLabs realtime API on 2026-10-28.
    partials: Final = (
        "The quick brown-",
        "The quick brown fox jumps",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog. Light LLM.",
        "The quick brown fox jumps over the lazy dog. Light LLM now streams-",
        "The quick brown fox jumps over the lazy dog. Light LLM now streams audio to ElevenLabs.",
        "The quick brown fox jumps over the lazy dog. Light LLM now streams audio to ElevenLabs Scribe.",
    )
    committed: Final = "The quick brown fox jumps over the lazy dog. Light LLM now streams audio to ElevenLabs Scribe."
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        *(_frame("partial_transcript", text=text) for text in partials),
        _frame("committed_transcript", text=committed),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "The quick"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " brown fox"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " jumps over the lazy"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " dog. Light"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " LLM now"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " streams audio to"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " ElevenLabs"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " Scribe."),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", committed),
    ]


def test_cumulative_partials_start_the_turn_once_and_emit_only_new_text():
    transformer: Final = ElevenLabsEventTransformer(new_item_id=_item_ids())
    first: Final = _events(transformer, _frame("partial_transcript", text="Hello"))
    later: Final = _events(
        transformer,
        _frame("partial_transcript", text="Hello wor"),
        _frame("partial_transcript", text="Hello world"),
        _frame("partial_transcript", text="Hello world again"),
    )

    assert _summary(first) == [("input_audio_buffer.speech_started", "item_0", None)]
    assert _summary([*first, *later]) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "Hello"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " world"),
    ]


def test_a_revised_emitted_word_keeps_streaming_from_the_same_word_position_and_the_commit_flushes_the_tail():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text="Hello world today"),
        _frame("partial_transcript", text="Hallo world tomorrow soon"),
        _frame("partial_transcript", text="Hello world tomorrow soon"),
        _frame("committed_transcript", text="Hello world tomorrow soon."),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "Hello world"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " tomorrow"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " soon."),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "Hello world tomorrow soon."),
    ]


def test_case_and_punctuation_changes_in_emitted_words_allow_new_stable_words():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text="Hello world today"),
        _frame("partial_transcript", text="hello, world today again"),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "Hello world"),
        ("conversation.item.input_audio_transcription.delta", "item_0", " today"),
    ]


def test_each_commit_closes_its_turn_and_the_next_partial_opens_a_new_item():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text="One"),
        _frame("committed_transcript", text="One."),
        _frame("partial_transcript", text="Two"),
        _frame("committed_transcript", text="Two."),
    )

    completed: Final = [
        event for event in events if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert [(event["item_id"], event["transcript"]) for event in completed] == [("item_0", "One."), ("item_1", "Two.")]


def test_commit_without_partials_still_reports_a_whole_turn():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()), _frame("committed_transcript", text="Hi")
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "Hi"),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "Hi"),
    ]


def test_empty_partials_and_empty_commits_outside_a_turn_are_ignored():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        _frame("partial_transcript", text=""),
        _frame("committed_transcript", text=""),
    )

    assert events == []


def test_each_completion_bills_only_the_audio_recorded_since_the_previous_one():
    transformer: Final = ElevenLabsEventTransformer(new_item_id=_item_ids())
    transformer.record_audio(1.5)
    first: Final = _events(transformer, _frame("committed_transcript", text="One."))
    transformer.record_audio(0.25)
    second: Final = _events(transformer, _frame("committed_transcript", text="Two."))

    assert first[-1]["usage"] == {"type": "duration", "seconds": 1.5}
    assert second[-1]["usage"] == {"type": "duration", "seconds": 0.25}
    assert transformer.take_unbilled_usage() is None


def test_trailing_audio_is_billed_once_at_close():
    transformer: Final = ElevenLabsEventTransformer()
    transformer.record_audio(2.0)

    assert transformer.take_unbilled_usage() == {"type": "duration", "seconds": 2.0}
    assert transformer.take_unbilled_usage() is None


def test_session_started_reports_session_created_with_the_upstream_session_id():
    events: Final = _events(ElevenLabsEventTransformer(), _frame("session_started", session_id="el_sess_1", config={}))

    assert [event["type"] for event in events] == ["session.created"]
    session: Final = events[0]["session"]
    assert isinstance(session, dict)
    assert session["id"] == "el_sess_1"


@pytest.mark.parametrize("error_type", get_args(ElevenLabsErrorType))
def test_provider_errors_name_the_error_type_without_upstream_detail(error_type: str):
    events: Final = _events(ElevenLabsEventTransformer(), _frame(error_type, error="private upstream detail"))

    assert len(events) == 1
    assert events[0]["type"] == "error"
    message: Final = json.dumps(events[0])
    assert error_type in message
    assert "private upstream detail" not in message


def test_unspaced_chinese_partials_stream_the_prefix_kept_across_two_partials():
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        *(_frame("partial_transcript", text=text) for text in ("今天", "今天天气", "今天天气很好", "今天天气很好。")),
        _frame("committed_transcript", text="今天天气很好。"),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "今天"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "天气"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "很好"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "。"),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "今天天气很好。"),
    ]


def test_unspaced_japanese_partials_stream_the_prefix_kept_across_two_partials():
    partials: Final = ("こんにちは", "こんにちは世界", "こんにちは世界です", "こんにちは世界です。")
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        *(_frame("partial_transcript", text=text) for text in partials),
        _frame("committed_transcript", text="こんにちは世界です。"),
    )

    assert _summary(events) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "こんにちは"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "世界"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "です"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "。"),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "こんにちは世界です。"),
    ]


def test_a_revision_inside_unspaced_text_emits_nothing_until_the_emitted_prefix_is_kept_again():
    transformer: Final = ElevenLabsEventTransformer(new_item_id=_item_ids())
    first: Final = _events(
        transformer, _frame("partial_transcript", text="今天"), _frame("partial_transcript", text="今天天气")
    )
    revised: Final = _events(
        transformer,
        _frame("partial_transcript", text="明天天气很好"),
        _frame("partial_transcript", text="今天天气很好"),
    )
    resumed: Final = _events(
        transformer,
        _frame("partial_transcript", text="今天天气很好啊"),
        _frame("committed_transcript", text="今天天气很好啊。"),
    )

    assert revised == []
    assert _summary([*first, *resumed]) == [
        ("input_audio_buffer.speech_started", "item_0", None),
        ("conversation.item.input_audio_transcription.delta", "item_0", "今天"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "天气很好"),
        ("conversation.item.input_audio_transcription.delta", "item_0", "啊。"),
        ("input_audio_buffer.speech_stopped", "item_0", None),
        ("conversation.item.input_audio_transcription.completed", "item_0", "今天天气很好啊。"),
    ]


def _streamed_deltas(partials: tuple[str, ...], committed: str) -> tuple[str, ...]:
    events: Final = _events(
        ElevenLabsEventTransformer(new_item_id=_item_ids()),
        *(_frame("partial_transcript", text=text) for text in partials),
        _frame("committed_transcript", text=committed),
    )
    assert events[-1]["transcript"] == committed
    deltas: Final = tuple(
        str(event["delta"]) for event in events if event["type"] == "conversation.item.input_audio_transcription.delta"
    )
    assert "".join(deltas) == committed, deltas
    return deltas


def test_a_turn_that_moves_from_unspaced_to_spaced_text_only_extends_the_emitted_text():
    partials: Final = ("我喜欢", "我喜欢用", "我喜欢用 LiteLLM 网关", "我喜欢用 LiteLLM 网关 很好")

    assert _streamed_deltas(partials, "我喜欢用 LiteLLM 网关 很好。") == ("我喜欢", "用", " LiteLLM 网关", " 很好。")


def test_japanese_with_spaced_latin_words_streams_by_character_instead_of_waiting_for_a_space():
    partials: Final = ("今日は", "今日は LiteLLM", "今日は LiteLLM を使って", "今日は LiteLLM を使ってテストします")

    assert _streamed_deltas(partials, "今日は LiteLLM を使ってテストします。") == (
        "今日は",
        " LiteLLM",
        " を使って",
        "テストします。",
    )


def test_live_chinese_partials_with_provisional_trailing_punctuation_stream_only_kept_text():
    # Captured from the live ElevenLabs realtime API (scribe_v2_realtime, language_code=zh) on 2026-11-21.
    partials: Final = (
        "今天天气-",
        "今天天气很好。",
        "今天天气很好，我们一起去公园。",
        "今天天气很好，我们一起去公园散步吧。",
        "今天天气很好，我们一起去公园散步吧。下午我们",
        "今天天气很好，我们一起去公园散步吧，下午我们再回来喝茶。",
    )
    committed: Final = "今天天气很好，我们一起去公园散步吧，下午我们再回来喝茶。"

    assert _streamed_deltas(partials, committed) == ("今天天气", "很好", "，我们一起去公园", "散步吧", "，下午我们再回来喝茶。")


@pytest.mark.parametrize(
    ("partials", "committed", "expected"),
    [
        pytest.param(
            ("今天天汽", "今天天汽", "今天天气很好"), "今天天气很好。", ("今天天", "气很好。"), id="repeated-unspaced"
        ),
        pytest.param(
            ("Lite", "Lite", "Light LLM now"), "Light LLM now.", ("Light LLM", " now."), id="repeated-ascii-word"
        ),
        pytest.param(("Hel", "Hello", "Hello world"), "Hello world.", ("Hello", " world."), id="growing-ascii-word"),
    ],
)
def test_a_repeated_or_growing_partial_never_emits_its_unconfirmed_last_character(
    partials: tuple[str, ...], committed: str, expected: tuple[str, ...]
):
    assert _streamed_deltas(partials, committed) == expected


def test_warnings_and_unknown_messages_are_not_forwarded():
    assert _events(ElevenLabsEventTransformer(), _frame("warning", warning="slow down")) == []
    assert parse_server_message(_frame("committed_transcript_with_timestamps", text="x")) is None
    assert parse_server_message(json.dumps({"message_type": ["not", "a", "string"]})) is None


@pytest.mark.parametrize(
    "payload", [_frame("partial_transcript"), "[]", "not json", b'{"message_type":"partial_transcript","text":"\xff"}']
)
def test_malformed_server_messages_raise_the_protocol_error(payload: str | bytes):
    with pytest.raises(ElevenLabsProtocolError):
        parse_server_message(payload)


def test_bytes_payloads_are_decoded():
    assert parse_server_message(_frame("partial_transcript", text="hi").encode()) is not None

EMPTY_TRANSFORM_INPUT: Final[RealtimeResponseTransformInput] = {
    "session_configuration_request": None,
    "current_output_item_id": None,
    "current_response_id": None,
    "current_delta_chunks": None,
    "current_item_chunks": None,
    "current_conversation_id": None,
    "current_delta_type": None,
}


def _append(audio: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode()})


def _configured(update: str | None = None) -> ElevenLabsRealtimeConfig:
    config: Final = ElevenLabsRealtimeConfig()
    config.transform_realtime_request(update or _session_update(), SCRIBE_REALTIME_MODEL)
    return config


def _response(config: ElevenLabsRealtimeConfig, payload: str) -> list[dict[str, object]]:
    response: Final = config.transform_realtime_response(payload, SCRIBE_REALTIME_MODEL, MagicMock(), EMPTY_TRANSFORM_INPUT)
    events: Final = response["response"]
    assert isinstance(events, list)
    return [dict(event) for event in events]


def test_session_update_becomes_one_configure_message_and_later_updates_are_ignored():
    config: Final = ElevenLabsRealtimeConfig(options=ElevenLabsRealtimeOptions(keyterms=("keyword",)))
    sent: Final = config.transform_realtime_request(_session_update(rate=16_000, turn_detection=None), SCRIBE_REALTIME_MODEL)

    assert len(sent) == 1
    assert isinstance(sent[0], str)
    configure: Final = ElevenLabsRealtimeConfigure.model_validate_json(sent[0])
    assert config.is_setup_message(json.loads(sent[0]))
    query: Final = parse_qs(configure.query)
    assert query["audio_format"] == ["pcm_16000"]
    assert query["commit_strategy"] == ["manual"]
    assert query["keyterms"] == ["keyword"]
    assert config.transform_realtime_request(_session_update(rate=48_000), SCRIBE_REALTIME_MODEL) == ()


def test_audio_before_session_update_is_rejected():
    with pytest.raises(ElevenLabsProtocolError, match=r"session\.update"):
        ElevenLabsRealtimeConfig().transform_realtime_request(_append(b"\x00\x00"), SCRIBE_REALTIME_MODEL)


def test_append_becomes_input_audio_chunks_that_reassemble_the_audio():
    config: Final = _configured(_session_update(rate=16_000))
    audio: Final = bytes(range(256)) * ((MAX_AUDIO_CHUNK_BYTES // 256) + 1)
    chunks: Final = [json.loads(sent) for sent in config.transform_realtime_request(_append(audio), SCRIBE_REALTIME_MODEL)]

    assert len(chunks) == 2
    assert all(chunk["message_type"] == "input_audio_chunk" for chunk in chunks)
    assert all(chunk["commit"] is False and chunk["sample_rate"] == 16_000 for chunk in chunks)
    assert all(len(base64.b64decode(chunk["audio_base_64"])) <= MAX_AUDIO_CHUNK_BYTES for chunk in chunks)
    assert b"".join(base64.b64decode(chunk["audio_base_64"]) for chunk in chunks) == audio


def test_response_create_has_no_elevenlabs_equivalent_and_is_dropped():
    assert _configured().transform_realtime_request(json.dumps({"type": "response.create"}), SCRIBE_REALTIME_MODEL) == ()


def test_clear_without_uncommitted_audio_sends_nothing_and_keeps_the_next_turn():
    config: Final = _configured()
    cleared: Final = config.transform_realtime_request(
        json.dumps({"type": "input_audio_buffer.clear"}), SCRIBE_REALTIME_MODEL
    )
    events: Final = _response(config, _frame("committed_transcript", text="Hello."))

    assert cleared == ()
    assert events[-1]["transcript"] == "Hello."


@pytest.mark.parametrize(("options", "addon_count"), [(ElevenLabsRealtimeOptions(keyterms=("LiteLLM",)), 1), (NO_OPTIONS, 0)])
def test_keyterms_add_a_priced_addon_model_to_the_session(options: ElevenLabsRealtimeOptions, addon_count: int):
    addons: Final = ElevenLabsRealtimeConfig(options=options).transcription_addon_pricing_models()

    assert len(addons) == addon_count
    assert all((litellm.get_model_info(addon)["input_cost_per_second"] or 0) > 0 for addon in addons)


async def _billing_session(rate: int, upstream: RealtimeBackend) -> tuple[ElevenLabsRealtimeConfig, RealtimeBackend]:
    async def connect(url: str, headers: Mapping[str, str]) -> RealtimeBackend:
        return upstream

    def factory(
        url: str, headers: Mapping[str, str], on_audio_sent: Callable[[float], None]
    ) -> RealtimeBackend:
        return ElevenLabsRealtimeBackend(url, headers, connect=connect, on_audio_sent=on_audio_sent)

    config: Final = ElevenLabsRealtimeConfig(backend_factory=factory)
    backend: Final = await config.open_backend("wss://h/p", {"xi-api-key": "k"})
    assert backend is not None
    for message in config.transform_realtime_request(_session_update(rate=rate), SCRIBE_REALTIME_MODEL):
        await backend.send(message)
    return config, backend


async def _send_audio(config: ElevenLabsRealtimeConfig, backend: RealtimeBackend, audio: bytes) -> None:
    for message in config.transform_realtime_request(_append(audio), SCRIBE_REALTIME_MODEL):
        await backend.send(message)


async def _audio_session() -> tuple[ElevenLabsRealtimeConfig, RealtimeBackend]:
    upstream: Final = MagicMock(spec=RealtimeBackend)
    upstream.send = AsyncMock()
    return await _billing_session(24_000, upstream)


@pytest.mark.parametrize("event_type", ["input_audio_buffer.commit", "input_audio_buffer.end"])
async def test_commit_and_end_commit_the_audio_sent_since_the_last_commit_once(event_type: str):
    config, backend = await _audio_session()
    commit: Final = json.dumps({"type": event_type})
    before_audio: Final = config.transform_realtime_request(commit, SCRIBE_REALTIME_MODEL)
    await _send_audio(config, backend, b"\x00\x00" * 2_400)
    after_audio: Final = config.transform_realtime_request(commit, SCRIBE_REALTIME_MODEL)
    repeated: Final = config.transform_realtime_request(commit, SCRIBE_REALTIME_MODEL)

    assert before_audio == ()
    assert [json.loads(message) for message in after_audio] == [
        {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True, "sample_rate": 24_000}
    ]
    assert repeated == ()


async def test_clear_discards_the_uncommitted_turn_and_bills_its_audio_with_the_next_one():
    config, backend = await _audio_session()
    await _send_audio(config, backend, b"\x00\x00" * 24_000)
    abandoned: Final = _response(config, _frame("partial_transcript", text="never mind this"))
    cleared: Final = config.transform_realtime_request(
        json.dumps({"type": "input_audio_buffer.clear"}), SCRIBE_REALTIME_MODEL
    )
    discarded: Final = [
        *_response(config, _frame("partial_transcript", text="never mind this part")),
        *_response(config, _frame("committed_transcript", text="Never mind this part.")),
    ]
    await _send_audio(config, backend, b"\x00\x00" * 12_000)
    kept: Final = [
        *_response(config, _frame("partial_transcript", text="Hello there")),
        *_response(config, _frame("committed_transcript", text="Hello there.")),
    ]

    assert [json.loads(message)["commit"] for message in cleared] == [True]
    assert discarded == []
    assert kept[0]["item_id"] != abandoned[0]["item_id"]
    assert [event.get("delta") for event in kept if "delta" in event] == ["Hello", " there."]
    assert (kept[-1]["transcript"], kept[-1]["usage"]) == ("Hello there.", {"type": "duration", "seconds": 1.5})


@pytest.mark.parametrize("rate", [16_000, 48_000])
async def test_completed_transcripts_bill_the_seconds_of_audio_sent(rate: int):
    upstream: Final = MagicMock(spec=RealtimeBackend)
    upstream.send = AsyncMock()
    config, backend = await _billing_session(rate, upstream)
    for audio in (b"\x00\x00" * rate, b"\x00\x00" * (rate // 2)):
        for message in config.transform_realtime_request(_append(audio), SCRIBE_REALTIME_MODEL):
            await backend.send(message)
    events: Final = _response(config, _frame("committed_transcript", text="hello"))

    assert events[-1]["usage"] == {"type": "duration", "seconds": 1.5}
    assert config.unbilled_usage_on_session_close(SCRIBE_REALTIME_MODEL) is None


async def test_audio_without_a_transcript_is_billed_at_close():
    upstream: Final = MagicMock(spec=RealtimeBackend)
    upstream.send = AsyncMock()
    config, backend = await _billing_session(24_000, upstream)
    for message in config.transform_realtime_request(_append(b"\x00\x00" * 12_000), SCRIBE_REALTIME_MODEL):
        await backend.send(message)

    assert config.unbilled_usage_on_session_close(SCRIBE_REALTIME_MODEL) == {"type": "duration", "seconds": 0.5}


async def test_failed_audio_chunks_do_not_contribute_to_completed_or_close_usage():
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    upstream: Final = MagicMock(spec=RealtimeBackend)
    upstream.send = AsyncMock(side_effect=[None, ConnectionClosedError(rcvd=Close(1006, ""), sent=None)])
    config, backend = await _billing_session(16_000, upstream)
    chunks: Final = config.transform_realtime_request(_append(b"\x00\x00" * 32_000), SCRIBE_REALTIME_MODEL)
    await backend.send(chunks[0])
    with pytest.raises(ConnectionClosedError):
        await backend.send(chunks[1])
    events: Final = _response(config, _frame("committed_transcript", text="hello"))
    assert events[-1]["usage"] == {"type": "duration", "seconds": 1.0}

    upstream.send = AsyncMock()
    for message in config.transform_realtime_request(_append(b"\x00\x00" * 8_000), SCRIBE_REALTIME_MODEL):
        await backend.send(message)
    assert config.unbilled_usage_on_session_close(SCRIBE_REALTIME_MODEL) == {"type": "duration", "seconds": 0.5}


_REFUSAL_STATUS_CODES: Final = {
    "auth_error": 401,
    "quota_exceeded": 429,
    "rate_limited": 429,
    "invalid_request": 400,
    "input_error": 400,
}


@pytest.mark.parametrize("error_type", get_args(ElevenLabsErrorType))
def test_an_error_before_session_started_refuses_the_session_with_the_mapped_status(error_type: str):
    config: Final = _configured()
    events: Final = _response(config, _frame(error_type, error="private upstream detail"))

    refusal: Final = config.session_refusal()
    assert isinstance(refusal, ElevenLabsException)
    assert (refusal.status_code, refusal.message) == (
        _REFUSAL_STATUS_CODES.get(error_type, 500),
        f"ElevenLabs realtime transcription failed: {error_type}",
    )
    expected_type: Final = "invalid_request_error" if refusal.status_code < 500 else "server_error"
    assert events == [{"type": "error", "error": {"type": expected_type, "message": refusal.message}}]


def test_an_error_after_session_started_is_an_error_event_and_not_a_refusal():
    config: Final = _configured()
    _response(config, _frame("session_started", session_id="el_sess_1"))
    events: Final = _response(config, _frame("auth_error", error="private upstream detail"))

    assert events == [
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "ElevenLabs realtime transcription failed: auth_error",
            },
        }
    ]
    assert config.session_refusal() is None


async def test_a_refused_session_bills_nothing_at_close():
    upstream: Final = MagicMock(spec=RealtimeBackend)
    upstream.send = AsyncMock()
    config, backend = await _billing_session(24_000, upstream)
    for message in config.transform_realtime_request(_append(b"\x00\x00" * 12_000), SCRIBE_REALTIME_MODEL):
        await backend.send(message)
    _response(config, _frame("auth_error", error="Invalid API key"))

    assert config.unbilled_usage_on_session_close(SCRIBE_REALTIME_MODEL) is None


def test_session_started_without_a_synthetic_event_reports_the_upstream_id_and_configured_session():
    config: Final = _configured(_session_update(rate=16_000, language="es"))
    events: Final = _response(config, _frame("session_started", session_id="el_sess_1"))

    session: Final = events[0]["session"]
    assert isinstance(session, dict)
    assert session["id"] == "el_sess_1"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 16_000}
    assert session["audio"]["input"]["transcription"] == {"model": SCRIBE_REALTIME_MODEL, "language": "es"}


def test_synthetic_session_created_uses_the_defaults():
    event: Final = ElevenLabsRealtimeConfig().transform_session_created_event(SCRIBE_REALTIME_MODEL, "trace_1")

    assert event["session"]["id"] == "trace_1"
    assert event["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24_000}


def test_session_started_keeps_the_synthetic_id_and_reports_the_configured_audio_rate():
    config: Final = ElevenLabsRealtimeConfig()
    synthetic: Final = config.transform_session_created_event(SCRIBE_REALTIME_MODEL, "trace_1")
    config.transform_realtime_request(_session_update(rate=16_000), SCRIBE_REALTIME_MODEL)
    events: Final = _response(config, _frame("session_started", session_id="el_sess_1"))

    assert synthetic["session"]["id"] == "trace_1"
    assert synthetic["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24_000}
    assert events[0]["session"]["id"] == "trace_1"
    assert events[0]["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 16_000}


def test_api_key_travels_only_in_the_xi_api_key_header():
    config: Final = ElevenLabsRealtimeConfig()
    headers: Final = config.validate_environment({"x-existing": "1"}, SCRIBE_REALTIME_MODEL, api_key=" secret ")
    url: Final = config.get_complete_url(None, SCRIBE_REALTIME_MODEL)

    assert headers == {"x-existing": "1", "xi-api-key": "secret"}
    assert "secret" not in url


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        ElevenLabsRealtimeConfig().validate_environment({}, SCRIBE_REALTIME_MODEL)


def test_api_base_selects_a_regional_host_and_unsupported_models_are_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELEVENLABS_API_BASE", raising=False)
    config: Final = ElevenLabsRealtimeConfig()

    assert config.get_complete_url(None, f"elevenlabs/{SCRIBE_REALTIME_MODEL}") == DEFAULT_ELEVENLABS_REALTIME_URL
    assert (
        config.get_complete_url("https://api.eu.residency.elevenlabs.io", SCRIBE_REALTIME_MODEL)
        == "wss://api.eu.residency.elevenlabs.io/v1/speech-to-text/realtime"
    )
    with pytest.raises(ValueError, match="Unsupported ElevenLabs realtime model"):
        config.get_complete_url(None, "scribe_v2")


def test_build_url_defaults_to_the_global_endpoint():
    assert build_elevenlabs_realtime_url(None) == DEFAULT_ELEVENLABS_REALTIME_URL


async def test_open_backend_returns_the_injected_backend_for_the_url_and_headers():
    calls: Final[list[tuple[str, Mapping[str, str]]]] = []
    sentinel: Final = MagicMock(spec=RealtimeBackend)

    def factory(url: str, headers: Mapping[str, str], on_audio_sent: Callable[[float], None]) -> RealtimeBackend:
        calls.append((url, headers))
        return sentinel

    backend: Final = await ElevenLabsRealtimeConfig(backend_factory=factory).open_backend("wss://h/p", {"xi-api-key": "k"})

    assert backend is sentinel
    assert calls == [("wss://h/p", {"xi-api-key": "k"})]


def test_config_built_from_litellm_params_applies_the_options():
    config: Final = elevenlabs_realtime_config({"api_key": "k", "keyterms": ["keyword"]}, drop_params=False)
    sent: Final = config.transform_realtime_request(_session_update(), SCRIBE_REALTIME_MODEL)

    assert parse_qs(ElevenLabsRealtimeConfigure.model_validate_json(sent[0]).query)["keyterms"] == ["keyword"]


def test_provider_config_manager_builds_the_elevenlabs_config_from_deployment_params():
    config: Final = ProviderConfigManager.get_provider_realtime_config(
        model=SCRIBE_REALTIME_MODEL,
        provider=LlmProviders.ELEVENLABS,
        litellm_params={"api_key": "k", "keyterms": ["Scribe"]},
    )
    assert config is not None
    sent: Final = config.transform_realtime_request(
        json.dumps({"type": "transcription_session.update", "session": {"type": "transcription"}}),
        SCRIBE_REALTIME_MODEL,
    )

    assert parse_qs(ElevenLabsRealtimeConfigure.model_validate_json(sent[0]).query)["keyterms"] == ["Scribe"]


class _ScriptedUpstream:
    def __init__(self, *frames: str | Exception) -> None:
        self.frames = list(frames)
        self.sent: list[str | bytes] = []
        self.closed = False

    async def __aenter__(self) -> "_ScriptedUpstream":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame

    async def close(self) -> None:
        self.closed = True


class _RecordingConnect:
    def __init__(self, upstream: _ScriptedUpstream) -> None:
        self.upstream = upstream
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    async def __call__(self, url: str, headers: Mapping[str, str]) -> RealtimeBackend:
        self.calls.append((url, dict(headers)))
        return self.upstream


async def test_health_check_opens_a_default_session_without_audio_and_is_healthy_on_session_started():
    options: Final = ElevenLabsRealtimeOptions(keyterms=("Scribe",))
    upstream: Final = _ScriptedUpstream(_frame("session_started", session_id="el_sess_1"))
    connect: Final = _RecordingConnect(upstream)

    healthy: Final = await elevenlabs_realtime_health_check(
        f"elevenlabs/{SCRIBE_REALTIME_MODEL}", None, "secret", options, connect=connect
    )

    assert healthy is True
    assert connect.calls == [
        (f"{DEFAULT_ELEVENLABS_REALTIME_URL}?{default_session_config(options).query()}", {"xi-api-key": "secret"})
    ]
    assert (upstream.sent, upstream.closed) == ([], True)


async def test_health_check_raises_the_mapped_refusal_when_elevenlabs_rejects_the_key():
    upstream: Final = _ScriptedUpstream(_frame("auth_error", error="Invalid API key"))

    with pytest.raises(ElevenLabsException) as refused:
        await elevenlabs_realtime_health_check(
            SCRIBE_REALTIME_MODEL, None, "bad-key", NO_OPTIONS, connect=_RecordingConnect(upstream)
        )

    assert (refused.value.status_code, refused.value.message) == (
        401,
        "ElevenLabs realtime transcription failed: auth_error",
    )
    assert upstream.closed is True


async def test_health_check_raises_a_gateway_timeout_when_elevenlabs_never_starts_the_session():
    class _SilentUpstream(_ScriptedUpstream):
        async def recv(self, decode: bool | None = None) -> str | bytes:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    upstream: Final = _SilentUpstream()

    with pytest.raises(ElevenLabsException) as timed_out:
        await elevenlabs_realtime_health_check(
            SCRIBE_REALTIME_MODEL, None, "secret", NO_OPTIONS, connect=_RecordingConnect(upstream), start_timeout=0.01
        )

    assert (timed_out.value.status_code, timed_out.value.message) == (
        504,
        "ElevenLabs did not start the realtime session in time",
    )
    assert upstream.closed is True


async def test_health_check_raises_a_bad_gateway_when_elevenlabs_closes_before_starting_the_session():
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close

    upstream: Final = _ScriptedUpstream(ConnectionClosedOK(rcvd=Close(1000, ""), sent=None))

    with pytest.raises(ElevenLabsException) as failed:
        await elevenlabs_realtime_health_check(
            SCRIBE_REALTIME_MODEL, None, "secret", NO_OPTIONS, connect=_RecordingConnect(upstream)
        )

    assert failed.value.status_code == 502
    assert upstream.closed is True
