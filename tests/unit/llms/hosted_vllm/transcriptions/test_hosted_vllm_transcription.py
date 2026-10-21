"""Hosted vLLM audio transcription (#33384).

The request goes through the shared http handler, so a per-request `ssl_verify` (a custom
CA bundle or False) configures the client instead of being ignored by the OpenAI SDK transport.
"""

import io
from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.hosted_vllm.transcriptions.transformation import HostedVLLMAudioTranscriptionConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL: Final = "openai/whisper-large-v3"
CUSTOM_CA: Final = "/etc/ssl/custom-ca.pem"
API_BASE: Final = "https://vllm.invalid:8443/v1"
WAV: Final = ("speech.wav", b"RIFF0000WAVEfmt ", "audio/wav")


@pytest.fixture(autouse=True)
def _clear_hosted_vllm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)


def test_provider_config_uses_the_native_transcription_endpoint():
    config: Final = ProviderConfigManager.get_provider_audio_transcription_config(
        model=f"hosted_vllm/{MODEL}",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert isinstance(config, HostedVLLMAudioTranscriptionConfig)
    assert config.has_native_transcription_endpoint is True


@pytest.mark.parametrize(
    "api_base",
    [
        "https://vllm.invalid:8443",
        "https://vllm.invalid:8443/",
        "https://vllm.invalid:8443/v1",
        "https://vllm.invalid:8443/v1/",
        "https://vllm.invalid:8443/v1/audio/transcriptions",
    ],
)
def test_get_complete_url_appends_audio_transcriptions_once(api_base: str):
    url: Final = HostedVLLMAudioTranscriptionConfig().get_complete_url(
        api_base=api_base, api_key=None, model=MODEL, optional_params={}, litellm_params={}
    )

    assert url == "https://vllm.invalid:8443/v1/audio/transcriptions"


def test_get_complete_url_requires_api_base():
    with pytest.raises(ValueError, match="api_base"):
        HostedVLLMAudioTranscriptionConfig().get_complete_url(
            api_base=None, api_key=None, model=MODEL, optional_params={}, litellm_params={}
        )


def test_request_is_multipart_with_extra_body_flattened():
    request: Final = HostedVLLMAudioTranscriptionConfig().transform_audio_transcription_request(
        model=MODEL,
        audio_file=WAV,
        optional_params={
            "language": "en",
            "temperature": 0.2,
            "timestamp_granularities": ["word", "segment"],
            "prompt": None,
            "extra_body": {"vad_filter": True, "language": "de"},
        },
        litellm_params={"ssl_verify": CUSTOM_CA},
    )

    assert request.files == {"file": WAV}
    assert request.data == {
        "model": MODEL,
        "vad_filter": "true",
        "language": "en",
        "temperature": "0.2",
        "timestamp_granularities": ["word", "segment"],
    }


class _SentRequest:
    """The one request the mock transport received."""

    def __init__(self) -> None:
        self.request: httpx.Request | None = None

    def respond(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.request = request
        return httpx.Response(200, json={"text": "transcribed"})


def _assert_sent_with_custom_ca(factory: Mock, sent: _SentRequest) -> None:
    assert factory.call_args.kwargs["params"] == {"ssl_verify": CUSTOM_CA}
    assert sent.request is not None
    assert str(sent.request.url) == "https://vllm.invalid:8443/v1/audio/transcriptions"
    assert sent.request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="file"' in sent.request.content
    assert b"ssl_verify" not in sent.request.content


def test_transcription_builds_its_client_with_the_request_ssl_verify():
    """Regression for #33384: the CA path from litellm_params must reach the http client."""
    sent: Final = _SentRequest()
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(sent.respond)))

    with patch("litellm.llms.custom_httpx.llm_http_handler.get_httpx_client", return_value=client) as factory:
        response: Final = litellm.transcription(
            model=f"hosted_vllm/{MODEL}",
            file=(WAV[0], io.BytesIO(WAV[1]), WAV[2]),
            api_base=API_BASE,
            ssl_verify=CUSTOM_CA,
        )

    _assert_sent_with_custom_ca(factory, sent)
    assert response.text == "transcribed"


@pytest.mark.asyncio
async def test_atranscription_builds_its_client_with_the_request_ssl_verify():
    sent: Final = _SentRequest()
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(sent.respond))

    with patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client", return_value=client) as factory:
        response: Final = await litellm.atranscription(
            model=f"hosted_vllm/{MODEL}",
            file=(WAV[0], io.BytesIO(WAV[1]), WAV[2]),
            api_base=API_BASE,
            ssl_verify=CUSTOM_CA,
        )

    _assert_sent_with_custom_ca(factory, sent)
    assert response.text == "transcribed"
