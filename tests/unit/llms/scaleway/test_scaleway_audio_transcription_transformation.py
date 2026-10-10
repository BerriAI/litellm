import os
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.scaleway.audio_transcription.transformation import (
    ScalewayAudioTranscriptionConfig,
    ScalewayAudioTranscriptionException,
)
from litellm.types.utils import TranscriptionResponse


# ---------------------------------------------------------------------------
# get_complete_url
# ---------------------------------------------------------------------------


def test_scaleway_get_complete_url_default_base():
    """With no api_base supplied, Scaleway's Generative API endpoint is used."""
    url = ScalewayAudioTranscriptionConfig().get_complete_url(
        api_base=None,
        api_key="fake",
        model="whisper-large-v3",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.scaleway.ai/v1/audio/transcriptions"


def test_scaleway_get_complete_url_custom_base_strips_trailing_slash():
    """Caller-supplied api_base is respected; trailing slash is normalized."""
    url = ScalewayAudioTranscriptionConfig().get_complete_url(
        api_base="https://custom.example.com/v1/",
        api_key="fake",
        model="whisper-large-v3",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.example.com/v1/audio/transcriptions"


# ---------------------------------------------------------------------------
# validate_environment
# ---------------------------------------------------------------------------


def test_scaleway_validate_environment_explicit_api_key():
    headers = ScalewayAudioTranscriptionConfig().validate_environment(
        headers={},
        model="whisper-large-v3",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="explicit-key",
    )
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["accept"] == "application/json"


def test_scaleway_validate_environment_reads_scw_secret_key(monkeypatch):
    monkeypatch.setenv("SCW_SECRET_KEY", "env-secret")
    headers = ScalewayAudioTranscriptionConfig().validate_environment(
        headers={},
        model="whisper-large-v3",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Authorization"] == "Bearer env-secret"


def test_scaleway_validate_environment_explicit_api_key_wins_over_env(monkeypatch):
    """Caller-supplied api_key must win over the SCW_SECRET_KEY env var."""
    monkeypatch.setenv("SCW_SECRET_KEY", "env-secret")
    headers = ScalewayAudioTranscriptionConfig().validate_environment(
        headers={},
        model="whisper-large-v3",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="explicit-wins",
    )
    assert headers["Authorization"] == "Bearer explicit-wins"


# ---------------------------------------------------------------------------
# transform_audio_transcription_request
# ---------------------------------------------------------------------------


def _open_test_audio():
    """Shared helper: open the repo's canonical speech fixture."""
    wav_path = os.path.join(
        os.path.dirname(__file__),
        "../../../..",
        "tests",
        "llm_translation",
        "gettysburg.wav",
    )
    return open(wav_path, "rb")


def test_scaleway_transform_request_builds_multipart_with_supported_params():
    with _open_test_audio() as audio_file:
        result = (
            ScalewayAudioTranscriptionConfig().transform_audio_transcription_request(
                model="whisper-large-v3",
                audio_file=audio_file,
                optional_params={
                    "language": "en",
                    "temperature": 0.0,
                    "response_format": "verbose_json",
                },
                litellm_params={},
            )
        )

    assert isinstance(result.data, dict)
    assert result.data["model"] == "whisper-large-v3"
    assert result.data["language"] == "en"
    assert result.data["temperature"] == 0.0
    assert result.data["response_format"] == "verbose_json"
    assert result.files is not None
    assert "file" in result.files
    assert len(result.files["file"]) == 3  # (filename, content, content_type)


def test_scaleway_transform_request_drops_unsupported_params():
    """Only params in get_supported_openai_params() should land in the form."""
    with _open_test_audio() as audio_file:
        result = (
            ScalewayAudioTranscriptionConfig().transform_audio_transcription_request(
                model="whisper-large-v3",
                audio_file=audio_file,
                optional_params={
                    "language": "en",
                    "stream": True,  # not supported
                    "diarize": True,  # not supported
                },
                litellm_params={},
            )
        )

    assert "stream" not in result.data
    assert "diarize" not in result.data
    assert result.data["language"] == "en"


# ---------------------------------------------------------------------------
# transform_audio_transcription_response
# ---------------------------------------------------------------------------


def test_scaleway_transform_response_parses_text():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"text": "Four score and seven years ago"}

    response = (
        ScalewayAudioTranscriptionConfig().transform_audio_transcription_response(
            mock_response
        )
    )

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "Four score and seven years ago"


def test_scaleway_transform_response_preserves_segments_and_language():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "text": "hello world",
        "language": "en",
        "segments": [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.6, "end": 1.1},
        ],
    }

    response = (
        ScalewayAudioTranscriptionConfig().transform_audio_transcription_response(
            mock_response
        )
    )

    assert response.text == "hello world"
    assert response["language"] == "en"
    assert len(response["segments"]) == 2


def test_scaleway_transform_response_raises_typed_exception_on_non_json():
    """Malformed upstream body must raise the Scaleway-typed exception so
    error handlers downstream can classify it as a Scaleway failure."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json.side_effect = ValueError("not json")
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = "upstream 502 bad gateway"
    mock_response.status_code = 502

    with pytest.raises(ScalewayAudioTranscriptionException):
        ScalewayAudioTranscriptionConfig().transform_audio_transcription_response(
            mock_response
        )


def test_scaleway_transform_response_returns_plain_text_for_non_json_content_type():
    """When Scaleway responds with text/srt/vtt (response_format="text" etc.),
    the content-type is not application/json — return the body as plain text
    rather than exploding on .json()."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/plain; charset=utf-8"}
    mock_response.text = "Four score and seven years ago"

    response = (
        ScalewayAudioTranscriptionConfig().transform_audio_transcription_response(
            mock_response
        )
    )

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "Four score and seven years ago"


def test_scaleway_validate_environment_raises_when_no_key(monkeypatch):
    """Missing credential should fail fast with a typed exception rather than
    silently emitting 'Bearer None'."""
    monkeypatch.delenv("SCW_SECRET_KEY", raising=False)

    with pytest.raises(ScalewayAudioTranscriptionException) as excinfo:
        ScalewayAudioTranscriptionConfig().validate_environment(
            headers={},
            model="whisper-large-v3",
            messages=[],
            optional_params={},
            litellm_params={},
        )

    assert "SCW_SECRET_KEY" in str(excinfo.value)
