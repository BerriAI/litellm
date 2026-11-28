"""Azure audio on the wire: Azure OpenAI speech and transcription, and Azure AI Speech (Cognitive Services).

Two different Azure audio products sit behind the same proxy routes:

- Azure OpenAI audio (`azure/tts-1`, `azure/whisper-1`, `azure/gpt-4o-transcribe`): the OpenAI SDK's Azure client
  (`AzureChatCompletion.audio_speech`, `AzureAudioTranscription.async_audio_transcriptions`) posts to
  `{api_base}/openai/deployments/{deployment}/audio/speech` or `/audio/transcriptions` with `?api-version=<v>`
  and the key in `api-key`. Speech is JSON; transcription is multipart with the deployment as `model`.
- Azure AI Speech, selected by a `azure/speech/...` model (`litellm/main.py` speech branch, and the cost map
  row `azure/speech/azure-stt` carrying `audio_transcription_config: azure_speech`). Text to speech posts SSML to
  `{api_base}/cognitiveservices/v1` with `Ocp-Apim-Subscription-Key`, `Content-Type: application/ssml+xml` and
  `X-Microsoft-OutputFormat` (`litellm/llms/azure/text_to_speech/transformation.py`). Speech to text posts the
  raw audio to `{api_base}/speech/recognition/conversation/cognitiveservices/v1?language=..&format=..`
  (`litellm/llms/azure/audio_transcription/transformation.py`). A loopback `api_base` is neither a Cognitive
  Services nor a regional speech host, so the path is joined onto it unchanged.

A Rust port of these routes must keep the binary passthrough, the SSML rendering (voice mapping, prosody rate,
XML escaping) and the per-product auth header.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure3_support import (
    SPEECH_BYTES,
    WAV_BYTES,
    binary_reply,
    multipart_fields,
    multipart_file,
)
from integration.providers.dec1_azure_support import (
    API_KEY,
    DATED_API_VERSION,
    azure,
    body_of,
    error_message,
    json_reply,
    marker,
    path_of,
    query_of,
    spend_rows_for_group,
)

TTS_DEPLOYMENT: Final = "tts-1"
WHISPER_DEPLOYMENT: Final = "whisper-1"
AVA_TTS_MODEL: Final = "azure/speech/azure-tts"
AVA_STT_MODEL: Final = "azure/speech/azure-stt"
TTS_PATH: Final = "/cognitiveservices/v1"
STT_PATH: Final = "/speech/recognition/conversation/cognitiveservices/v1"


def _speech_ok(_: Request) -> Reply:
    return binary_reply(SPEECH_BYTES, "audio/mpeg")


def _transcribed(_: Request) -> Reply:
    return json_reply({"text": "synthetic transcript"})


def _recognized(_: Request) -> Reply:
    return json_reply({"RecognitionStatus": "Success", "DisplayText": "Synthetic recognized text.", "Offset": 0})


def _transcribe(gateway: Gateway, model: str, **fields: str):  # noqa: ANN202  # httpx.Response
    return gateway.request_multipart(
        "/v1/audio/transcriptions", {"model": model, **fields}, {"file": ("clip.wav", WAV_BYTES, "audio/wav")}
    )


def test_azure_openai_speech_posts_json_to_the_deployment_speech_route(gateway: Gateway) -> None:
    """An `azure/tts-1` deployment with `api_version: 2024-10-21`; the caller asks for speech with voice `alloy`.

    Azure must see exactly one `POST /openai/deployments/tts-1/audio/speech?api-version=2024-10-21` whose JSON body
    has `model: tts-1`, the caller's `input` and `voice: alloy`, with the key in `api-key`.
    """
    text: Final = marker()
    with azure(gateway, _speech_ok, model=f"azure/{TTS_DEPLOYMENT}") as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/audio/speech", {"model": rig.model, "input": text, "voice": "alloy"}
        )
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (
            f"/openai/deployments/{TTS_DEPLOYMENT}/audio/speech",
            {"api-version": DATED_API_VERSION},
        )
        body: Final = body_of(request)
        assert (body["model"], body["input"], body["voice"]) == (TTS_DEPLOYMENT, text, "alloy")
        assert request.headers.get("api-key") == API_KEY


def test_azure_openai_speech_bytes_reach_the_caller_unchanged(gateway: Gateway) -> None:
    """Azure answers `audio/mpeg` bytes.

    The caller must get 200, exactly those bytes, and a `content-type` starting with `audio/mpeg`
    (`resolve_speech_media_type` keeps an upstream `audio/*` type).
    """
    with azure(gateway, _speech_ok, model=f"azure/{TTS_DEPLOYMENT}") as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/audio/speech", {"model": rig.model, "input": marker(), "voice": "alloy"}
        )
        assert response.status_code == 200, response.text
        assert response.content == SPEECH_BYTES
        assert response.headers["content-type"].startswith("audio/mpeg")


def test_azure_openai_speech_forwards_response_format_and_speed(gateway: Gateway) -> None:
    """The caller sends `response_format: wav` and `speed: 1.25`.

    Both must reach Azure's JSON body unchanged (`response_format: "wav"`, `speed: 1.25`).
    """
    with azure(gateway, lambda _: binary_reply(SPEECH_BYTES, "audio/wav"), model=f"azure/{TTS_DEPLOYMENT}") as rig:
        rig.gateway.request(
            "POST",
            "/v1/audio/speech",
            {"model": rig.model, "input": marker(), "voice": "nova", "response_format": "wav", "speed": 1.25},
        )
        body: Final = body_of(rig.one_post())
        assert (body["response_format"], body["speed"]) == ("wav", 1.25)


def test_azure_openai_speech_is_billed_per_input_character(gateway: Gateway) -> None:
    """An `azure/tts-1` deployment with no custom prices speaks a 40 character input.

    The cost map row `azure/tts-1` prices `input_cost_per_character: 1.5e-05`, so the one SpendLogs row for the
    group must have `spend == 40 * 1.5e-05 = 0.0006` and `call_type` `aspeech`. UNVERIFIED: call_type literal.
    """
    text: Final = "x" * 40
    with azure(gateway, _speech_ok, model=f"azure/{TTS_DEPLOYMENT}") as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/audio/speech", {"model": rig.model, "input": text, "voice": "alloy"}
        )
        assert response.status_code == 200, response.text
        (row,) = spend_rows_for_group(rig.model)
        assert float(str(row["spend"])) == pytest.approx(40 * 1.5e-05)
        assert row["call_type"] == "aspeech"


def test_azure_ai_speech_posts_ssml_to_cognitiveservices_v1_with_the_subscription_key(gateway: Gateway) -> None:
    """An `azure/speech/azure-tts` deployment on a loopback `api_base`; the caller sends plain text, voice `alloy`.

    Azure must see one `POST /cognitiveservices/v1` with `Ocp-Apim-Subscription-Key: synthetic-azure-key`,
    `Content-Type: application/ssml+xml`, `X-Microsoft-OutputFormat: audio-24khz-48kbitrate-mono-mp3` (the default)
    and `User-Agent: litellm`, and no `api-key` header.
    """
    with azure(gateway, _speech_ok, model=AVA_TTS_MODEL, api_version=None) as rig:
        response: Final = rig.gateway.request(
            "POST", "/v1/audio/speech", {"model": rig.model, "input": "hello", "voice": "alloy"}
        )
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert path_of(request) == TTS_PATH
        assert request.headers.get("ocp-apim-subscription-key") == API_KEY
        assert request.headers.get("content-type", "").startswith("application/ssml+xml")
        assert request.headers.get("x-microsoft-outputformat") == "audio-24khz-48kbitrate-mono-mp3"
        assert request.headers.get("user-agent") == "litellm"
        assert "api-key" not in request.headers


def test_azure_ai_speech_maps_openai_voices_to_azure_neural_voices(gateway: Gateway) -> None:
    """The caller asks for voice `alloy`.

    `VOICE_MAPPINGS` maps it to `en-US-JennyNeural`, so the SSML body must contain
    `<voice name='en-US-JennyNeural'>` and a `<prosody rate='0%'>` wrapper around the text (no speed given).
    """
    with azure(gateway, _speech_ok, model=AVA_TTS_MODEL, api_version=None) as rig:
        rig.gateway.request("POST", "/v1/audio/speech", {"model": rig.model, "input": "hello", "voice": "alloy"})
        ssml: Final = rig.one_post().body.decode()
        assert "<voice name='en-US-JennyNeural'>" in ssml, ssml
        assert "<prosody rate='0%'>hello</prosody>" in ssml, ssml


def test_azure_ai_speech_turns_speed_into_a_prosody_rate_and_wav_into_riff(gateway: Gateway) -> None:
    """The caller sends `speed: 1.5` and `response_format: wav` with an Azure voice name `en-GB-SoniaNeural`.

    The rate must be `+50%` (`_convert_speed_to_azure_rate`), the output format header
    `riff-24khz-16bit-mono-pcm`, and an Azure voice name must pass through unmapped.
    """
    with azure(
        gateway, lambda _: binary_reply(SPEECH_BYTES, "audio/wav"), model=AVA_TTS_MODEL, api_version=None
    ) as rig:
        rig.gateway.request(
            "POST",
            "/v1/audio/speech",
            {"model": rig.model, "input": "hi", "voice": "en-GB-SoniaNeural", "speed": 1.5, "response_format": "wav"},
        )
        request: Final = rig.one_post()
        ssml: Final = request.body.decode()
        assert "<voice name='en-GB-SoniaNeural'>" in ssml and "<prosody rate='+50%'>hi</prosody>" in ssml, ssml
        assert request.headers.get("x-microsoft-outputformat") == "riff-24khz-16bit-mono-pcm"


def test_azure_ai_speech_escapes_xml_in_plain_text_input(gateway: Gateway) -> None:
    """The caller's input is `a < b & "c"`.

    Plain text is escaped before it is placed in SSML, so the body must contain
    `a &lt; b &amp; &quot;c&quot;` and never the raw `<` from the input.
    """
    with azure(gateway, _speech_ok, model=AVA_TTS_MODEL, api_version=None) as rig:
        rig.gateway.request("POST", "/v1/audio/speech", {"model": rig.model, "input": 'a < b & "c"', "voice": "alloy"})
        ssml: Final = rig.one_post().body.decode()
        assert "a &lt; b &amp; &quot;c&quot;" in ssml, ssml


def test_azure_ai_speech_passes_caller_ssml_through_verbatim(gateway: Gateway) -> None:
    """The caller's input already starts with `<speak ...>`.

    `is_ssml_input` detects it, so Azure must receive exactly the caller's SSML bytes as the body, with no
    second `<speak>` wrapper and no escaping.
    """
    ssml: Final = "<speak version='1.0' xml:lang='en-US'><voice name='en-US-AriaNeural'>raw</voice></speak>"
    with azure(gateway, _speech_ok, model=AVA_TTS_MODEL, api_version=None) as rig:
        rig.gateway.request("POST", "/v1/audio/speech", {"model": rig.model, "input": ssml, "voice": "alloy"})
        assert rig.one_post().body.decode() == ssml


def test_azure_openai_transcription_posts_multipart_to_the_deployment_route(gateway: Gateway) -> None:
    """An `azure/whisper-1` deployment with `api_version: 2024-10-21`; the caller uploads `clip.wav`.

    Azure must see one `POST /openai/deployments/whisper-1/audio/transcriptions?api-version=2024-10-21` with the
    key in `api-key`, a multipart `model` field `whisper-1` and the audio bytes unchanged in the `file` part.

    BUG (run 44176682961): Azure receives `api-version=2025-02-01-preview`. The async branch of
    `AzureAudioTranscription.audio_transcriptions` calls `async_audio_transcriptions` without `api_version`
    (litellm/llms/azure/audio_transcriptions.py:55-69, the parameter defaults to None at :131 and is passed on at
    :142), so the client is built from `os.getenv("AZURE_API_VERSION", AZURE_DEFAULT_API_VERSION)`
    (litellm/llms/azure/common_utils.py:937-938) and the deployment's `api_version: 2024-10-21` is lost. The
    sync branch passes it (:71-79), and text to speech on the same deployment shape keeps it.
    """
    pytest.skip("BUG: async Azure transcription drops the deployment's api_version and sends the process default")
    with azure(gateway, _transcribed, model=f"azure/{WHISPER_DEPLOYMENT}") as rig:
        response: Final = _transcribe(rig.gateway, rig.model)
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (
            f"/openai/deployments/{WHISPER_DEPLOYMENT}/audio/transcriptions",
            {"api-version": DATED_API_VERSION},
        )
        assert request.headers.get("api-key") == API_KEY
        assert multipart_fields(request)["model"] == WHISPER_DEPLOYMENT
        assert multipart_file(request) == WAV_BYTES


def test_azure_openai_transcription_returns_the_text(gateway: Gateway) -> None:
    """Azure answers `{"text": "synthetic transcript"}`.

    The caller must get 200 and `text == "synthetic transcript"`.
    """
    with azure(gateway, _transcribed, model=f"azure/{WHISPER_DEPLOYMENT}") as rig:
        response: Final = _transcribe(rig.gateway, rig.model)
        assert response.status_code == 200, response.text
        assert response.json()["text"] == "synthetic transcript"


def test_azure_openai_transcription_forwards_language_prompt_and_response_format(gateway: Gateway) -> None:
    """The caller sends `language: de`, `prompt: glossary`, `response_format: json` and `temperature: 0`.

    Each must reach Azure as its own multipart field with the same string value.
    """
    with azure(gateway, _transcribed, model=f"azure/{WHISPER_DEPLOYMENT}") as rig:
        _transcribe(rig.gateway, rig.model, language="de", prompt="glossary", response_format="json", temperature="0")
        fields: Final = multipart_fields(rig.one_post())
        assert {name: fields.get(name) for name in ("language", "prompt", "response_format", "temperature")} == {
            "language": "de",
            "prompt": "glossary",
            "response_format": "json",
            "temperature": "0",
        }


def test_azure_ai_speech_to_text_posts_raw_audio_to_the_short_audio_route(gateway: Gateway) -> None:
    """An `azure/speech/azure-stt` deployment on a loopback `api_base`; the caller uploads `clip.wav`.

    Azure must see one `POST /speech/recognition/conversation/cognitiveservices/v1` with query exactly
    `{"language": "en-US", "format": "simple"}`, `Ocp-Apim-Subscription-Key: synthetic-azure-key`,
    `Accept: application/json`, and the raw audio bytes as the body (not multipart).
    """
    with azure(gateway, _recognized, model=AVA_STT_MODEL, api_version=None) as rig:
        response: Final = _transcribe(rig.gateway, rig.model)
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == (STT_PATH, {"language": "en-US", "format": "simple"})
        assert request.headers.get("ocp-apim-subscription-key") == API_KEY
        assert request.headers.get("accept") == "application/json"
        assert request.body == WAV_BYTES


def test_azure_ai_speech_to_text_returns_the_display_text(gateway: Gateway) -> None:
    """Azure answers `{"RecognitionStatus": "Success", "DisplayText": "Synthetic recognized text."}`.

    The caller must get 200 with `text == "Synthetic recognized text."` (`_extract_text`).
    """
    with azure(gateway, _recognized, model=AVA_STT_MODEL, api_version=None) as rig:
        response: Final = _transcribe(rig.gateway, rig.model)
        assert response.status_code == 200, response.text
        assert response.json()["text"] == "Synthetic recognized text."


def test_azure_ai_speech_to_text_maps_language_and_verbose_json(gateway: Gateway) -> None:
    """The caller sends `language: de-DE` and `response_format: verbose_json`.

    The query must become exactly `{"language": "de-DE", "format": "detailed"}` (`_get_azure_response_format`).
    """
    with azure(gateway, _recognized, model=AVA_STT_MODEL, api_version=None) as rig:
        _transcribe(rig.gateway, rig.model, language="de-DE", response_format="verbose_json")
        assert query_of(rig.one_post()) == {"language": "de-DE", "format": "detailed"}


def test_azure_ai_speech_to_text_reads_nbest_when_display_text_is_absent(gateway: Gateway) -> None:
    """Azure answers detailed format: no `DisplayText`, `NBest: [{"Display": "From n best.", "Lexical": "..."}]`.

    The caller's `text` must be `From n best.` (first NBest entry's `Display`).
    """
    nbest: Final = json_reply(
        {"RecognitionStatus": "Success", "NBest": [{"Display": "From n best.", "Lexical": "from n best"}]}
    )
    with azure(gateway, lambda _: nbest, model=AVA_STT_MODEL, api_version=None) as rig:
        response: Final = _transcribe(rig.gateway, rig.model, response_format="verbose_json")
        assert response.status_code == 200, response.text
        assert response.json()["text"] == "From n best."


def test_azure_ai_speech_to_text_no_match_is_an_error_naming_the_status(gateway: Gateway) -> None:
    """Azure answers HTTP 200 with `{"RecognitionStatus": "NoMatch"}`.

    The transformation raises, so the caller must not get 200, and the error message must contain
    `Azure AI Speech transcription failed with RecognitionStatus=NoMatch.`. UNVERIFIED: the status code the
    proxy picks for an exception carrying status 200.
    """
    with azure(
        gateway, lambda _: json_reply({"RecognitionStatus": "NoMatch"}), model=AVA_STT_MODEL, api_version=None
    ) as rig:
        response: Final = _transcribe(rig.gateway, rig.model)
        assert response.status_code != 200, response.text
        assert "Azure AI Speech transcription failed with RecognitionStatus=NoMatch." in error_message(response)
