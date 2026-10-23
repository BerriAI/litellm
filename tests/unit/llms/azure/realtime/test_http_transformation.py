import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from pydantic import JsonValue

_DATED_PREVIEW_API_VERSION: Final = "2025-04-01-preview"
_API_BASES: Final = (
    "https://my.openai.azure.com",
    "https://my.openai.azure.com/",
    "https://my.openai.azure.com/openai",
)
_GA_TRANSCRIPTION_CLIENT_SECRET: Final[dict[str, JsonValue]] = {
    "value": "ek_upstream",
    "expires_at": 1792775579,
    "session": {
        "type": "transcription",
        "object": "realtime.transcription_session",
        "id": "sess_upstream",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "whisper-deployment", "language": "en"},
            }
        },
    },
}


class _AzureThatRecordsTheRequest:
    def __init__(self, reply: Mapping[str, JsonValue] | None = None) -> None:
        self.requests: Final[list[httpx.Request]] = []
        self.reply: Final = reply or {"value": "ek_upstream"}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.reply)

    def client(self) -> AsyncHTTPHandler:
        return AsyncHTTPHandler(transport=httpx.MockTransport(self))


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", _API_BASES)
async def test_azure_client_secret_targets_the_ga_v1_surface_even_with_a_dated_api_version(api_base: str) -> None:
    azure: Final = _AzureThatRecordsTheRequest()
    await litellm.acreate_realtime_client_secret(
        model="azure/gpt-realtime-deployment",
        api_base=api_base,
        api_key="azure-key",
        api_version=_DATED_PREVIEW_API_VERSION,
        client=azure.client(),
    )
    (request,) = azure.requests
    assert str(request.url) == "https://my.openai.azure.com/openai/v1/realtime/client_secrets"
    assert request.headers.get("api-key") == "azure-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", _API_BASES)
async def test_azure_sdp_exchange_targets_the_ga_v1_surface_with_the_ephemeral_key_as_bearer(api_base: str) -> None:
    azure: Final = _AzureThatRecordsTheRequest()
    await litellm.arealtime_calls(
        openai_ephemeral_key="ek_upstream",
        sdp_body=b"v=0",
        model="azure/gpt-realtime-deployment",
        api_base=api_base,
        api_key="azure-key",
        api_version=_DATED_PREVIEW_API_VERSION,
        client=azure.client(),
    )
    (request,) = azure.requests
    assert str(request.url) == "https://my.openai.azure.com/openai/v1/realtime/calls"
    assert request.headers.get("authorization") == "Bearer ek_upstream"
    assert "api-key" not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", _API_BASES)
async def test_azure_transcription_session_is_minted_as_a_ga_transcription_client_secret(api_base: str) -> None:
    azure: Final = _AzureThatRecordsTheRequest(reply=_GA_TRANSCRIPTION_CLIENT_SECRET)
    response: Final = await litellm.acreate_realtime_transcription_session(
        model="azure/whisper-deployment",
        transcription_session={
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "smuggled-model", "language": "en"},
            "turn_detection": None,
        },
        api_base=api_base,
        api_key="azure-key",
        api_version=_DATED_PREVIEW_API_VERSION,
        client=azure.client(),
    )
    (request,) = azure.requests
    assert str(request.url) == "https://my.openai.azure.com/openai/v1/realtime/client_secrets"
    assert request.headers.get("api-key") == "azure-key"
    assert json.loads(request.content) == {
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "whisper-deployment", "language": "en"},
                    "turn_detection": None,
                }
            },
        }
    }
    assert response.status_code == 200
    assert response.json() == {
        "object": "realtime.transcription_session",
        "id": "sess_upstream",
        "input_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-deployment", "language": "en"},
        "client_secret": {"value": "ek_upstream", "expires_at": 1792775579},
    }
