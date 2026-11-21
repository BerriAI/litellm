"""`gemini/` chat: error mapping branches round one left out, retries, and audio output.

Errors. Every AI Studio failure goes through `_map_vertex_exception`
(`litellm/litellm_core_utils/exception_mapping_utils.py:1273`), which checks the error TEXT before the
HTTP status: Google's "exceeds the maximum number of tokens allowed" is a context window error, "The
response was blocked." a content policy error, "Content has no parts." an internal error, and "Resource
exhausted" a rate limit even on a 500. Only then do the status branches apply: 401, 408 (timeout), 502
(connection error, answered 500) and 503. A body `code: 429` inside a 400 is NOT a rate limit; that rewrite
is scoped to 5xx. Retries follow the deployment's `num_retries`: a 5xx is retried, a 400 is not.

Audio output. On a TTS model (`tts` in the name) the OpenAI `audio` param becomes
`generationConfig.speechConfig` and forces `responseModalities` to include `AUDIO`; any `format` other than
`pcm16` is refused unless params are dropped (`unsupported_gemini_tts_audio_format`,
`vertex_and_google_ai_studio_gemini.py:507`). An `inlineData` audio part comes back as `message.audio`
with `content: null`, and on a stream as `delta.audio` (#51750, `_create_streaming_choice`).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.providers.dec1_gemini2_support import (
    TTS_MODEL,
    deltas,
    error_message,
    fails_then_answers,
    frame,
    generation_config,
    sse_reply,
    stream_or_json,
)
from integration.providers.dec1_gemini_support import (
    gemini,
    gemini_echo,
    gemini_reply,
    google_error,
    marker,
    message_of,
    usage_metadata,
)
from pydantic import JsonValue

_PCM_B64: Final = "AAABAAIAAwA="
_AUDIO_PARTS: Final[list[JsonValue]] = [
    {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": _PCM_B64}}
]


# ---------------------------------------------------------------- error text before status


def test_a_too_long_prompt_is_a_context_window_error(gateway: Gateway) -> None:
    """Google's 400 `The input token count (1300000) exceeds the maximum number of tokens allowed (1048576).`
    is answered 400 with a message naming `ContextWindowExceededError`, the class routers use for context
    window fallbacks (`is_error_str_context_window_exceeded`, Gemini substring)."""
    message: Final = "The input token count (1300000) exceeds the maximum number of tokens allowed (1048576)."
    with gemini(gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", message), num_retries=0) as rig:
        response: Final = rig.chat()
        assert response.status_code == 400, response.text
        assert "ContextWindowExceededError" in error_message(response)


def test_a_blocked_response_text_is_a_content_policy_error(gateway: Gateway) -> None:
    """A 400 whose message contains `The response was blocked.` is answered 400 with a message naming
    `ContentPolicyViolationError` (`exception_mapping_utils.py:1350`)."""
    with gemini(
        gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", "The response was blocked."), num_retries=0
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 400, response.text
        assert "ContentPolicyViolationError" in error_message(response)


def test_content_has_no_parts_is_an_internal_error(gateway: Gateway) -> None:
    """A 400 whose message is `Content has no parts.` is answered 500, not 400: the text check raises
    `InternalServerError` before the status branch (`exception_mapping_utils.py:1312`)."""
    with gemini(
        gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", "Content has no parts."), num_retries=0
    ) as rig:
        assert rig.chat().status_code == 500


def test_resource_exhausted_text_on_a_500_is_a_rate_limit(gateway: Gateway) -> None:
    """A 500 whose message contains `Resource exhausted` is answered 429, so clients back off instead of
    treating it as a server fault (`exception_mapping_utils.py:1369`)."""
    with gemini(
        gateway, lambda _r: google_error(500, "INTERNAL", "Resource exhausted, retry later"), num_retries=0
    ) as rig:
        assert rig.chat().status_code == 429


def test_a_400_with_a_429_body_code_stays_a_bad_request(gateway: Gateway) -> None:
    """HTTP 400 with `{"error": {"code": 429, ...}}` and a neutral message stays 400: the body code rewrite
    to a rate limit only applies to 5xx envelopes (`exception_mapping_utils.py:1389`)."""
    body: Final = b'{"error": {"code": 429, "message": "dec1 neutral text", "status": "INVALID_ARGUMENT"}}'
    with gemini(gateway, lambda _r: Reply(status=400, body=body), num_retries=0) as rig:
        assert rig.chat().status_code == 400


@pytest.mark.parametrize(
    ("google_status", "status_name", "expected"),
    [(401, "UNAUTHENTICATED", 401), (408, "DEADLINE_EXCEEDED", 408), (502, "UNAVAILABLE", 500)],
)
def test_status_branches_for_401_408_and_502(
    gateway: Gateway, google_status: int, status_name: str, expected: int
) -> None:
    """With neutral messages, Google 401 is answered 401 (`AuthenticationError`), 408 is answered 408
    (`Timeout`), and 502 is answered 500 (`APIConnectionError`, whose status is 500)
    (`exception_mapping_utils.py` status branches). UNVERIFIED: the router may retry a 408 or 502 before
    answering; `num_retries: 0` is set to avoid that."""
    with gemini(
        gateway, lambda _r: google_error(google_status, status_name, "dec1 neutral text"), num_retries=0
    ) as rig:
        assert rig.chat().status_code == expected


# ---------------------------------------------------------------- retries


def test_a_500_is_retried_when_the_deployment_allows_retries(gateway: Gateway) -> None:
    """With `num_retries: 1` a first Google 500 and a second normal answer give the caller 200 "pong", and
    the peer saw exactly two `generateContent` calls. UNVERIFIED: retry backoff timing; the counts are the
    contract."""
    failure: Final = google_error(500, "INTERNAL", "dec1 transient")
    with gemini(gateway, fails_then_answers(failure), num_retries=1) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert message_of(response)["content"] == "pong"
        assert len(rig.peer.generate_calls()) == 2


def test_a_400_is_not_retried_even_with_retries_allowed(gateway: Gateway) -> None:
    """With `num_retries: 2` a Google 400 `INVALID_ARGUMENT` is answered 400 after exactly one call: a client
    error cannot succeed on retry."""
    with gemini(gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", "dec1 bad field"), num_retries=2) as rig:
        assert rig.chat().status_code == 400
        assert len(rig.peer.generate_calls()) == 1


# ---------------------------------------------------------------- audio output


def test_audio_on_a_tts_model_becomes_speech_config_and_audio_modality(gateway: Gateway) -> None:
    """`audio: {"voice": "Kore", "format": "pcm16"}` on gemini-2.5-flash-preview-tts reaches Google as
    `generationConfig.speechConfig: {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}` and
    `responseModalities: ["AUDIO"]`, added because no modality was given (`map_openai_params` tail)."""
    with gemini(gateway, lambda _r: gemini_reply(parts=_AUDIO_PARTS, model=TTS_MODEL), backend=TTS_MODEL) as rig:
        assert rig.chat(audio={"voice": "Kore", "format": "pcm16"}).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert config["speechConfig"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
        assert config["responseModalities"] == ["AUDIO"]


def test_text_and_audio_modalities_are_sent_once_each(gateway: Gateway) -> None:
    """`modalities: ["text", "audio"]` with an `audio` param gives `responseModalities: ["TEXT", "AUDIO"]`,
    with AUDIO not appended a second time."""
    with gemini(gateway, lambda _r: gemini_reply(parts=_AUDIO_PARTS, model=TTS_MODEL), backend=TTS_MODEL) as rig:
        assert rig.chat(modalities=["text", "audio"], audio={"voice": "Kore"}).status_code == 200
        assert generation_config(rig.peer.one_body())["responseModalities"] == ["TEXT", "AUDIO"]


def test_an_mp3_audio_format_is_refused_before_google(gateway: Gateway) -> None:
    """`audio: {"voice": "Kore", "format": "mp3"}` on the TTS model is answered 400 with
    `does not take the audio format 'mp3'` and Google is not called: Gemini TTS only returns PCM."""
    with gemini(gateway, gemini_echo, backend=TTS_MODEL) as rig:
        response: Final = rig.chat(audio={"voice": "Kore", "format": "mp3"})
        assert response.status_code == 400, response.text
        assert "does not take the audio format 'mp3'" in error_message(response)
        assert rig.peer.generate_calls() == ()


def test_an_audio_answer_comes_back_as_message_audio(gateway: Gateway) -> None:
    """An `inlineData` part with an `audio/...` mime type comes back as `message.audio` with `data` equal to the
    base64 Google sent and `transcript: ""`, and `message.content` is null as the OpenAI audio spec requires
    (`_extract_audio_response_from_parts`)."""
    with gemini(gateway, lambda _r: gemini_reply(parts=_AUDIO_PARTS, model=TTS_MODEL), backend=TTS_MODEL) as rig:
        message: Final = message_of(rig.chat(audio={"voice": "Kore"}))
        audio: Final = object_value(message["audio"])
        assert (audio["data"], audio["transcript"]) == (_PCM_B64, "")
        assert message.get("content") is None


def test_a_streamed_audio_answer_arrives_in_delta_audio(gateway: Gateway) -> None:
    """A stream whose frames carry `inlineData` audio parts delivers the base64 in `delta.audio.data`
    (#51750); no audio bytes leak into `delta.content`."""
    frames: Final = [frame(_AUDIO_PARTS), frame(_AUDIO_PARTS, finish="STOP", usage=usage_metadata())]
    with gemini(gateway, stream_or_json(sse_reply(frames), gemini_reply(parts=_AUDIO_PARTS)), backend=TTS_MODEL) as rig:
        response, text = rig.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": f"say hi {marker()}"}], "audio": {"voice": "Kore"}},
        )
        assert response.status_code == 200, text
        audio: Final = [object_value(delta["audio"]).get("data") for delta in deltas(text) if delta.get("audio")]
        assert audio == [_PCM_B64, _PCM_B64]
        assert not any(delta.get("content") for delta in deltas(text))
