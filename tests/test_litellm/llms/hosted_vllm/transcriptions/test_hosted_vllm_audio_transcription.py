from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.hosted_vllm.transcriptions.transformation import (
    HostedVLLMAudioTranscriptionConfig,
)


def _transcription_response() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        json={"text": "Test transcription"},
    )


def test_transcription_passes_custom_ca_to_sync_http_client() -> None:
    client = MagicMock()
    client.post.return_value = _transcription_response()

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=client,
    ) as get_httpx_client:
        response = litellm.transcription(
            model="hosted_vllm/whisper-1",
            file=("audio.wav", b"audio", "audio/wav"),
            api_base="https://vllm.example.com",
            ssl_verify="/path/to/ca-cert.crt",
        )

    assert response.text == "Test transcription"
    assert get_httpx_client.call_args.kwargs["params"]["ssl_verify"] == "/path/to/ca-cert.crt"
    request = client.post.call_args.kwargs
    assert request["data"] == {"model": "whisper-1"}
    assert request["files"] == {"file": ("audio.wav", b"audio", "audio/wav")}


@pytest.mark.asyncio
async def test_transcription_passes_custom_ca_to_async_http_client() -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=_transcription_response())

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
        return_value=client,
    ) as get_async_httpx_client:
        response = await litellm.atranscription(
            model="hosted_vllm/whisper-1",
            file=("audio.wav", b"audio", "audio/wav"),
            api_base="https://vllm.example.com",
            ssl_verify="/path/to/ca-cert.crt",
        )

    assert response.text == "Test transcription"
    assert get_async_httpx_client.call_args.kwargs["params"]["ssl_verify"] == "/path/to/ca-cert.crt"
    request = client.post.call_args.kwargs
    assert request["data"] == {"model": "whisper-1"}
    assert request["files"] == {"file": ("audio.wav", b"audio", "audio/wav")}


def test_transform_request_preserves_explicit_content_type() -> None:
    config = HostedVLLMAudioTranscriptionConfig()

    request_data = config.transform_audio_transcription_request(
        model="whisper-1",
        audio_file=("recording", b"audio", "audio/ogg"),
        optional_params={"language": "en", "extra_body": {"temperature": 0.1}},
        litellm_params={},
    )

    assert request_data.data == {"model": "whisper-1", "language": "en", "temperature": 0.1}
    assert request_data.files == {"file": ("recording", b"audio", "audio/ogg")}


def test_transform_request_derives_content_type_when_not_supplied() -> None:
    config = HostedVLLMAudioTranscriptionConfig()

    request_data = config.transform_audio_transcription_request(
        model="whisper-1",
        audio_file=("audio.mp3", b"audio"),
        optional_params={},
        litellm_params={},
    )

    assert request_data.files == {"file": ("audio.mp3", b"audio", "audio/mpeg")}
