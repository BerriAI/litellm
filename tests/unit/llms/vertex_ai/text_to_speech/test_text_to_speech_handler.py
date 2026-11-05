import base64
import binascii
import json
from datetime import datetime
from typing import Final

import google.auth
import google.auth.credentials
import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.vertex_ai.text_to_speech.text_to_speech_handler import VertexTextToSpeechAPI

SYNTHESIZE_URL: Final = "https://texttospeech.googleapis.com/v1/text:synthesize"
REGIONAL_SYNTHESIZE_URL: Final = "https://us-central1-texttospeech.googleapis.com/v1/text:synthesize"
AUDIO: Final = b"hello audio"
ENCODED_AUDIO: Final = base64.b64encode(AUDIO).decode()
BOTH_CALL_STYLES: Final = pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])


class _StaticGoogleCredentials(google.auth.credentials.Credentials):
    def refresh(self, request: object) -> None:
        self.token = "test-token"


@pytest.fixture
def synthesize_endpoint(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> respx.Route:
    monkeypatch.setattr(google.auth, "default", lambda scopes: (_StaticGoogleCredentials(), "test-project"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return respx_mock.post(REGIONAL_SYNTHESIZE_URL)


def _speech_logging() -> Logging:
    return Logging(
        model="vertex_ai/",
        messages=[],
        stream=False,
        call_type="speech",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="speech-call",
        function_id="speech-function",
    )


async def _synthesized_audio(
    run_async: bool,
    endpoint: respx.Route,
    payload: object,
    *,
    vertex_location: str | None = "us-central1",
    api_base: str | None = None,
) -> bytes:
    endpoint.mock(return_value=httpx.Response(200, content=json.dumps(payload).encode()))
    pending: Final = VertexTextToSpeechAPI().audio_speech(
        logging_obj=_speech_logging(),
        vertex_project="test-project",
        vertex_location=vertex_location,
        vertex_credentials=None,
        api_base=api_base,
        timeout=10.0,
        model="vertex_ai/",
        input="hello world",
        _is_async=run_async,
    )
    speech: Final = await pending if run_async else pending
    return speech.content


@BOTH_CALL_STYLES
@pytest.mark.parametrize(
    ("payload", "audio"),
    [
        ({"audioContent": ENCODED_AUDIO}, AUDIO),
        ({"audioContent": ENCODED_AUDIO, "timepoints": [], "audioConfig": {"audioEncoding": "MP3"}}, AUDIO),
        ({"audioContent": ENCODED_AUDIO[:8] + "\n" + ENCODED_AUDIO[8:] + "\n"}, AUDIO),
        ({"audioContent": ""}, b""),
    ],
)
async def test_vertex_speech_returns_the_decoded_audio_content(
    run_async: bool, synthesize_endpoint: respx.Route, payload: object, audio: bytes
) -> None:
    assert await _synthesized_audio(run_async, synthesize_endpoint, payload) == audio


@BOTH_CALL_STYLES
async def test_vertex_speech_sends_the_default_voice_request_with_the_google_token(
    run_async: bool, synthesize_endpoint: respx.Route
) -> None:
    await _synthesized_audio(run_async, synthesize_endpoint, {"audioContent": ENCODED_AUDIO})

    sent: Final = synthesize_endpoint.calls.last.request
    assert (json.loads(sent.content), sent.headers["authorization"], sent.headers["x-goog-user-project"]) == (
        {
            "input": {"text": "hello world"},
            "voice": {"languageCode": "en-US", "name": "en-US-Studio-O"},
            "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": "1"},
        },
        "Bearer test-token",
        "test-project",
    )


@BOTH_CALL_STYLES
@pytest.mark.parametrize(
    "body",
    [[], [{"audioContent": ENCODED_AUDIO}], "audioContent", 7, 403, 2.5, True, None, ["vertex-secret"]],
)
async def test_vertex_speech_rejects_a_synthesize_body_that_is_not_an_object_without_echoing_it(
    run_async: bool, synthesize_endpoint: respx.Route, body: object
) -> None:
    with pytest.raises(ValidationError) as rejection:
        await _synthesized_audio(run_async, synthesize_endpoint, body)

    rejection_text: Final = str(rejection.value)
    assert (
        [error["type"] for error in rejection.value.errors()],
        "vertex-secret" in rejection_text,
        "403" in rejection_text,
    ) == (["dict_type"], False, False), rejection_text


@BOTH_CALL_STYLES
@pytest.mark.parametrize("audio_content", [None, 5, 403, 1.5, True, [ENCODED_AUDIO], ["vertex-secret"], {"data": "x"}])
async def test_vertex_speech_rejects_audio_content_that_is_not_a_string_without_echoing_it(
    run_async: bool, synthesize_endpoint: respx.Route, audio_content: object
) -> None:
    with pytest.raises(ValidationError) as rejection:
        await _synthesized_audio(run_async, synthesize_endpoint, {"audioContent": audio_content})

    rejection_text: Final = str(rejection.value)
    assert (
        [error["type"] for error in rejection.value.errors()],
        "vertex-secret" in rejection_text,
        "403" in rejection_text,
    ) == (["string_type"], False, False), rejection_text


@BOTH_CALL_STYLES
@pytest.mark.parametrize(
    "payload",
    [{}, {"audiocontent": ENCODED_AUDIO}, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}],
)
async def test_vertex_speech_reports_a_synthesize_body_without_audio_content_as_a_key_error(
    run_async: bool, synthesize_endpoint: respx.Route, payload: object
) -> None:
    with pytest.raises(KeyError, match="audioContent"):
        await _synthesized_audio(run_async, synthesize_endpoint, payload)


@BOTH_CALL_STYLES
async def test_vertex_speech_reports_badly_padded_audio_content_with_the_base64_error(
    run_async: bool, synthesize_endpoint: respx.Route
) -> None:
    with pytest.raises(binascii.Error, match="Incorrect padding"):
        await _synthesized_audio(run_async, synthesize_endpoint, {"audioContent": "abc"})


@BOTH_CALL_STYLES
async def test_vertex_speech_reports_non_ascii_audio_content_with_the_base64_error(
    run_async: bool, synthesize_endpoint: respx.Route
) -> None:
    with pytest.raises(ValueError, match="string argument should contain only ASCII characters"):
        await _synthesized_audio(run_async, synthesize_endpoint, {"audioContent": "é中"})


@BOTH_CALL_STYLES
@pytest.mark.parametrize(
    ("vertex_location", "synthesize_url"),
    [
        (None, SYNTHESIZE_URL),
        ("global", SYNTHESIZE_URL),
        ("eu", "https://eu-texttospeech.googleapis.com/v1/text:synthesize"),
        ("us-central1", REGIONAL_SYNTHESIZE_URL),
    ],
)
async def test_vertex_speech_posts_to_the_location_host(
    run_async: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    vertex_location: str | None,
    synthesize_url: str,
) -> None:
    monkeypatch.setattr(google.auth, "default", lambda scopes: (_StaticGoogleCredentials(), "test-project"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    endpoint: Final = respx_mock.post(synthesize_url)

    await _synthesized_audio(
        run_async,
        endpoint,
        {"audioContent": ENCODED_AUDIO},
        vertex_location=vertex_location,
    )

    assert endpoint.call_count == 1
    assert str(endpoint.calls.last.request.url) == synthesize_url


@BOTH_CALL_STYLES
async def test_vertex_speech_prefers_api_base_over_vertex_location(
    run_async: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(google.auth, "default", lambda scopes: (_StaticGoogleCredentials(), "test-project"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    api_base: Final = "https://tts.gateway.internal/v1/text:synthesize"
    endpoint: Final = respx_mock.post(api_base)

    await _synthesized_audio(
        run_async,
        endpoint,
        {"audioContent": ENCODED_AUDIO},
        vertex_location="eu",
        api_base=api_base,
    )

    assert endpoint.call_count == 1
    assert str(endpoint.calls.last.request.url) == api_base
