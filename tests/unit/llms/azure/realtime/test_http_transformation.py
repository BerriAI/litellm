import gzip
import json
from typing import Final, Literal

import httpx
import pytest

import litellm
from litellm.llms.azure.realtime.transcription_sessions import azure_transcription_client_secret_response
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

_DATED_PREVIEW_API_VERSION: Final = "2025-04-01-preview"
_API_BASES: Final = (
    "https://my.openai.azure.com",
    "https://my.openai.azure.com/",
    "https://my.openai.azure.com/openai",
)
_AUDIO_FORMATS: Final = (
    ("pcm16", {"type": "audio/pcm", "rate": 24000}),
    ("g711_ulaw", {"type": "audio/pcmu"}),
    ("g711_alaw", {"type": "audio/pcma"}),
)


class _AzureThatRecordsTheRequest:
    def __init__(self, reply: dict[str, object] | None = None) -> None:
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
@pytest.mark.parametrize(("input_audio_format", "ga_audio_format"), _AUDIO_FORMATS)
async def test_azure_transcription_session_uses_ga_client_secrets(
    api_base: str, input_audio_format: str, ga_audio_format: dict[str, str | int]
) -> None:
    """Azure format and secret-expiry contracts, checked 2026-10-24.

    https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/realtime#components
    """
    azure: Final = _AzureThatRecordsTheRequest(
        reply={
            "value": "ek_upstream",
            "expires_at": 1792775579,
            "session": {
                "type": "transcription",
                "object": "realtime.transcription_session",
                "id": "sess_upstream",
                "expires_at": 0,
                "audio": {
                    "input": {
                        "format": ga_audio_format,
                        "transcription": {"model": "whisper-deployment", "language": "en"},
                        "noise_reduction": {"type": "near_field"},
                    }
                },
            },
        }
    )
    response: Final = await litellm.acreate_realtime_transcription_session(
        model="azure/whisper-deployment",
        transcription_session={
            "input_audio_format": input_audio_format,
            "input_audio_transcription": {"model": "untrusted-model", "language": "en"},
            "input_audio_noise_reduction": {"type": "near_field"},
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
                    "format": ga_audio_format,
                    "transcription": {"model": "whisper-deployment", "language": "en"},
                    "noise_reduction": {"type": "near_field"},
                }
            },
        }
    }
    assert response.status_code == 200
    assert response.json() == {
        "object": "realtime.transcription_session",
        "id": "sess_upstream",
        "input_audio_format": input_audio_format,
        "input_audio_transcription": {"model": "whisper-deployment", "language": "en"},
        "input_audio_noise_reduction": {"type": "near_field"},
        "client_secret": {"value": "ek_upstream", "expires_at": 1792775579},
    }


def test_azure_transcription_client_secret_response_preserves_errors() -> None:
    response: Final = httpx.Response(400, json={"error": "bad request"})

    assert azure_transcription_client_secret_response(response) is response


def test_azure_transcription_client_secret_response_preserves_non_object_json() -> None:
    response: Final = httpx.Response(200, json=[])

    assert azure_transcription_client_secret_response(response) is response


def test_azure_transcription_client_secret_response_preserves_upstream_headers_and_request() -> None:
    request: Final = httpx.Request("GET", "https://my.openai.azure.com/openai/v1/realtime/client_secrets")
    payload: Final = {
        "value": "ek_upstream",
        "expires_at": 1792775579,
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": "whisper-deployment", "language": "fr"},
                }
            },
        },
    }
    compressed_body: Final = gzip.compress(json.dumps(payload).encode())
    upstream_response: Final = httpx.Response(
        200,
        headers={
            "content-type": "application/json",
            "content-encoding": "gzip",
            "content-length": str(len(compressed_body)),
            "x-ratelimit-remaining-requests": "17",
        },
        content=compressed_body,
        request=request,
    )

    response: Final = azure_transcription_client_secret_response(upstream_response)

    assert response.request is request
    assert response.headers["x-ratelimit-remaining-requests"] == "17"
    assert "content-encoding" not in response.headers
    response.raise_for_status()
    assert response.json() == {
        "input_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-deployment", "language": "fr"},
        "client_secret": {"value": "ek_upstream", "expires_at": 1792775579},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_style", ("legacy", "ga"))
async def test_azure_transcription_session_preserves_requested_secret_expiry(
    expiry_style: Literal["legacy", "ga"],
) -> None:
    """Expiry contracts checked 2026-10-24 against the official SDK and Azure reference.

    https://github.com/openai/openai-python/blob/main/src/openai/types/beta/realtime/transcription_session_create_params.py
    https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/realtime#openairealtimecreateclientsecretrequest
    """
    azure: Final = _AzureThatRecordsTheRequest()
    expiry: Final = {"anchor": "created_at", "seconds": 600}
    await litellm.acreate_realtime_transcription_session(
        model="azure/whisper-deployment",
        transcription_session={"client_secret": {"expires_at": expiry}}
        if expiry_style == "legacy"
        else {"expires_after": expiry},
        api_base="https://my.openai.azure.com",
        api_key="azure-key",
        client=azure.client(),
    )

    (request,) = azure.requests
    assert json.loads(request.content) == {
        "expires_after": expiry,
        "session": {
            "type": "transcription",
            "audio": {"input": {"transcription": {"model": "whisper-deployment"}}},
        },
    }


@pytest.mark.asyncio
async def test_azure_transcription_session_preserves_disabled_audio_processing_with_legacy_modalities() -> None:
    """GA transcription has no modalities field and permits null audio settings, checked 2026-10-24.

    https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/realtime#openairealtimetranscriptionsessioncreaterequestga
    """
    azure: Final = _AzureThatRecordsTheRequest()
    include: Final = ["item.input_audio_transcription.logprobs"]
    await litellm.acreate_realtime_transcription_session(
        model="azure/whisper-deployment",
        transcription_session={
            "modalities": ["text"],
            "turn_detection": None,
            "input_audio_noise_reduction": None,
            "include": include,
        },
        api_base="https://my.openai.azure.com",
        api_key="azure-key",
        client=azure.client(),
    )

    (request,) = azure.requests
    assert json.loads(request.content) == {
        "session": {
            "type": "transcription",
            "include": include,
            "audio": {
                "input": {
                    "transcription": {"model": "whisper-deployment"},
                    "turn_detection": None,
                    "noise_reduction": None,
                }
            },
        }
    }


@pytest.mark.asyncio
async def test_azure_transcription_session_preserves_ga_audio_settings() -> None:
    azure: Final = _AzureThatRecordsTheRequest()
    await litellm.acreate_realtime_transcription_session(
        model="azure/whisper-routing-deployment",
        transcription_session={
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": "nested-model", "language": "fr"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            }
        },
        api_base="https://my.openai.azure.com",
        api_key="azure-key",
        client=azure.client(),
    )

    (request,) = azure.requests
    assert json.loads(request.content) == {
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": "whisper-routing-deployment", "language": "fr"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
        }
    }


@pytest.mark.asyncio
async def test_azure_transcription_session_legacy_audio_fields_override_ga_settings() -> None:
    azure: Final = _AzureThatRecordsTheRequest()
    await litellm.acreate_realtime_transcription_session(
        model="azure/whisper-routing-deployment",
        transcription_session={
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": "nested-model", "language": "fr"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
            "input_audio_format": "g711_alaw",
            "input_audio_transcription": {"language": "de"},
            "input_audio_noise_reduction": None,
            "turn_detection": None,
        },
        api_base="https://my.openai.azure.com",
        api_key="azure-key",
        client=azure.client(),
    )

    (request,) = azure.requests
    assert json.loads(request.content) == {
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcma"},
                    "transcription": {"model": "whisper-routing-deployment", "language": "de"},
                    "noise_reduction": None,
                    "turn_detection": None,
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
        }
    }
