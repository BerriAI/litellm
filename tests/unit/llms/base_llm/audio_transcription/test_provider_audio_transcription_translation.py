import io
import json
from typing import Final, Mapping

import httpx
import pytest
import respx
from pydantic import JsonValue
from respx import MockRouter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import transcription


class _Kwargs(TypedDict, total=False):
    model: ReadOnly[str]
    api_key: ReadOnly[str]
    api_base: ReadOnly[str]


class _Case(TypedDict):
    id: ReadOnly[str]
    provider: ReadOnly[str]
    kwargs: ReadOnly[_Kwargs]
    url: ReadOnly[str]
    prefix_match: ReadOnly[bool]


_AUDIO_BYTES: Final = b"RIFFFAKEWAVDATA-gettysburg"

_CASES: Final[tuple[_Case, ...]] = (
    {
        "id": "openai_gpt4o",
        "provider": "openai",
        "kwargs": {"model": "openai/gpt-4o-transcribe", "api_key": "sk-offline", "timestamp_granularities": None},
        "url": "https://api.openai.com/v1/audio/transcriptions",
        "prefix_match": False,
    },
    {
        "id": "elevenlabs_scribe",
        "provider": "elevenlabs",
        "kwargs": {"model": "elevenlabs/scribe_v1", "api_key": "xi-offline"},
        "url": "https://api.elevenlabs.io/v1/speech-to-text",
        "prefix_match": False,
    },
    {
        "id": "deepgram_nova",
        "provider": "deepgram",
        "kwargs": {"model": "deepgram/nova-2", "api_key": "dg-offline"},
        "url": "https://api.deepgram.com/v1/listen",
        "prefix_match": True,
    },
    {
        "id": "mistral_voxtral",
        "provider": "mistral",
        "kwargs": {"model": "mistral/voxtral-mini-latest", "api_key": "mistral-offline"},
        "url": "https://api.mistral.ai/v1/audio/transcriptions",
        "prefix_match": False,
    },
    {
        "id": "ovhcloud_whisper",
        "provider": "ovhcloud",
        "kwargs": {"model": "ovhcloud/whisper-large-v3-turbo", "api_key": "ovh-offline"},
        "url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/audio/transcriptions",
        "prefix_match": False,
    },
)


def _canned_response(case: _Case) -> httpx.Response:
    if case["provider"] == "deepgram":
        return httpx.Response(
            200,
            json={
                "metadata": {"transaction_key": "offline", "duration": 1.5},
                "results": {
                    "channels": [{"alternatives": [{"transcript": "four score and seven years ago", "confidence": 0.99}]}]
                },
            },
        )
    return httpx.Response(200, json={"text": "four score and seven years ago"})


@pytest.fixture(autouse=True)
def _httpx_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


def _kwargs(case: _Case) -> Mapping[str, JsonValue]:
    out: Final = {k: v for k, v in dict(case["kwargs"]).items() if v is not None}
    return out


def _register(case: _Case, respx_mock: MockRouter) -> respx.Route:
    if case["prefix_match"]:
        return respx_mock.post(url__startswith=case["url"]).mock(
            return_value=_canned_response(case)
        )
    return respx_mock.post(case["url"]).mock(return_value=_canned_response(case))


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_audio_transcription(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock)
    transcript = transcription(**_kwargs(case), file=io.BytesIO(_AUDIO_BYTES))
    request: Final = route.calls.last.request
    assert _AUDIO_BYTES in request.content
    assert transcript.text == "four score and seven years ago"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
@pytest.mark.asyncio
async def test_audio_transcription_async(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock)
    transcript = await litellm.atranscription(**_kwargs(case), file=io.BytesIO(_AUDIO_BYTES))
    request: Final = route.calls.last.request
    assert _AUDIO_BYTES in request.content
    assert transcript.text == "four score and seven years ago"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_audio_transcription_optional_params(case: _Case) -> None:
    from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

    optional_params: Final = get_supported_openai_params(
        model=case["kwargs"]["model"],
        custom_llm_provider=case["provider"],
        request_type="transcription",
    )
    assert optional_params is not None
    assert "max_completion_tokens" not in optional_params


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_audio_transcription_config(case: _Case) -> None:
    from litellm.llms.base_llm.audio_transcription.transformation import BaseAudioTranscriptionConfig
    from litellm.utils import ProviderConfigManager

    config: Final = ProviderConfigManager.get_provider_audio_transcription_config(
        model=case["kwargs"]["model"],
        provider=litellm.LlmProviders(case["provider"]),
    )
    assert config is not None
    assert isinstance(config, BaseAudioTranscriptionConfig)
