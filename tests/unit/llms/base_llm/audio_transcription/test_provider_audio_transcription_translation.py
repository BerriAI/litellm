import io
import json
from typing import Final, Mapping, Sequence, cast

import httpx
import pytest
import respx
from respx import MockRouter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import transcription
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.elevenlabs.audio_transcription.transformation import ElevenLabsAudioTranscriptionConfig
from litellm.llms.mistral.audio_transcription.transformation import MistralAudioTranscriptionConfig
from litellm.llms.ovhcloud.audio_transcription.transformation import OVHCloudAudioTranscriptionConfig
from litellm.utils import ProviderConfigManager


class _Kwargs(TypedDict, total=False):
    model: ReadOnly[str]
    api_key: ReadOnly[str]
    api_base: ReadOnly[str]
    timestamp_granularities: ReadOnly[Sequence[str]]


class _Case(TypedDict):
    id: ReadOnly[str]
    provider: ReadOnly[str]
    kwargs: ReadOnly[_Kwargs]
    url: ReadOnly[str]
    prefix_match: ReadOnly[bool]
    base_model: ReadOnly[str]
    request_markers: ReadOnly[tuple[bytes, ...]]
    marker_in_url: ReadOnly[bool]
    config_class: ReadOnly[type[BaseAudioTranscriptionConfig]]


_AUDIO_BYTES: Final = b"RIFFFAKEWAVDATA-gettysburg"

_CASES: Final[tuple[_Case, ...]] = (
    {
        "id": "openai_gpt4o",
        "provider": "openai",
        "kwargs": {
            "model": "openai/gpt-4o-transcribe",
            "api_key": "sk-offline",
            "timestamp_granularities": ["word"],
        },
        "url": "https://api.openai.com/v1/audio/transcriptions",
        "prefix_match": False,
        "base_model": "gpt-4o-transcribe",
        "request_markers": (b"gpt-4o-transcribe", b'name="timestamp_granularities[]"\r\n\r\nword\r\n'),
        "marker_in_url": False,
        "config_class": litellm.OpenAIGPTAudioTranscriptionConfig,
    },
    {
        "id": "elevenlabs_scribe",
        "provider": "elevenlabs",
        "kwargs": {"model": "elevenlabs/scribe_v1", "api_key": "xi-offline"},
        "url": "https://api.elevenlabs.io/v1/speech-to-text",
        "prefix_match": False,
        "base_model": "scribe_v1",
        "request_markers": (b"scribe_v1",),
        "marker_in_url": False,
        "config_class": ElevenLabsAudioTranscriptionConfig,
    },
    {
        "id": "deepgram_nova",
        "provider": "deepgram",
        "kwargs": {"model": "deepgram/nova-2", "api_key": "dg-offline"},
        "url": "https://api.deepgram.com/v1/listen",
        "prefix_match": True,
        "base_model": "nova-2",
        "request_markers": (b"model=nova-2",),
        "marker_in_url": True,
        "config_class": litellm.DeepgramAudioTranscriptionConfig,
    },
    {
        "id": "mistral_voxtral",
        "provider": "mistral",
        "kwargs": {"model": "mistral/voxtral-mini-latest", "api_key": "mistral-offline"},
        "url": "https://api.mistral.ai/v1/audio/transcriptions",
        "prefix_match": False,
        "base_model": "voxtral-mini-latest",
        "request_markers": (b"voxtral-mini-latest",),
        "marker_in_url": False,
        "config_class": MistralAudioTranscriptionConfig,
    },
    {
        "id": "ovhcloud_whisper",
        "provider": "ovhcloud",
        "kwargs": {"model": "ovhcloud/whisper-large-v3-turbo", "api_key": "ovh-offline"},
        "url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/audio/transcriptions",
        "prefix_match": False,
        "base_model": "whisper-large-v3-turbo",
        "request_markers": (b"whisper-large-v3-turbo",),
        "marker_in_url": False,
        "config_class": OVHCloudAudioTranscriptionConfig,
    },
)


def _canned_response(case: _Case) -> httpx.Response:
    if case["provider"] == "deepgram":
        return httpx.Response(
            200,
            json={
                "metadata": {"transaction_key": "offline", "duration": 1.5},
                "results": {
                    "channels": [
                        {"alternatives": [{"transcript": "four score and seven years ago", "confidence": 0.99}]}
                    ]
                },
            },
        )
    return httpx.Response(200, json={"text": "four score and seven years ago"})


@pytest.fixture(autouse=True)
def _httpx_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


def _register(case: _Case, respx_mock: MockRouter) -> respx.Route:
    if case["prefix_match"]:
        return respx_mock.post(url__startswith=case["url"]).mock(return_value=_canned_response(case))
    return respx_mock.post(case["url"]).mock(return_value=_canned_response(case))


def _assert_translated_request(case: _Case, request: httpx.Request) -> None:
    searched: Final = request.url.query if case["marker_in_url"] else request.content
    for marker in case["request_markers"]:
        assert marker in searched
    assert _AUDIO_BYTES in request.content


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_audio_transcription(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock)
    transcript: Final = transcription(**dict(case["kwargs"]), file=io.BytesIO(_AUDIO_BYTES))
    _assert_translated_request(case, route.calls.last.request)
    assert transcript.text == "four score and seven years ago"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
@pytest.mark.asyncio
async def test_audio_transcription_async(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock)
    transcript: Final = await litellm.atranscription(**dict(case["kwargs"]), file=io.BytesIO(_AUDIO_BYTES))
    _assert_translated_request(case, route.calls.last.request)
    assert transcript.text == "four score and seven years ago"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_audio_transcription_optional_params(case: _Case) -> None:
    optional_params: Final = get_supported_openai_params(
        model=case["kwargs"]["model"],
        custom_llm_provider=case["provider"],
        request_type="transcription",
    )
    assert isinstance(optional_params, list)
    assert optional_params == case["config_class"]().get_supported_openai_params(case["base_model"])
    assert "max_completion_tokens" not in optional_params


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_audio_transcription_config(case: _Case) -> None:
    config: Final = ProviderConfigManager.get_provider_audio_transcription_config(
        model=case["kwargs"]["model"],
        provider=litellm.LlmProviders(case["provider"]),
    )
    assert type(config) is case["config_class"]
    assert isinstance(config, BaseAudioTranscriptionConfig)
    if case["provider"] == "deepgram":
        complete_url: Final = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=case["base_model"],
            optional_params={},
            litellm_params={},
        )
        assert "api.deepgram.com" in complete_url
        assert "model=nova-2" in complete_url
    else:
        transformed: Final[AudioTranscriptionRequestData] = config.transform_audio_transcription_request(
            model=case["base_model"],
            audio_file=io.BytesIO(_AUDIO_BYTES),
            optional_params={},
            litellm_params={},
        )
        assert _AUDIO_BYTES in _transformed_payload(transformed)


def _transformed_payload(transformed: AudioTranscriptionRequestData) -> bytes:
    data: Final = transformed.data
    if isinstance(data, bytes):
        return data
    file_entry: Final = data.get("file") if isinstance(data, dict) else None
    if isinstance(file_entry, io.BytesIO):
        return file_entry.getvalue()
    if transformed.files is not None:
        first: Final = next(iter(transformed.files.values()))
        blob: Final = first[1] if isinstance(first, tuple) else first
        return blob.getvalue() if isinstance(blob, io.BytesIO) else cast(bytes, blob)
    return b""
