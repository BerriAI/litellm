from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

_DATED_PREVIEW_API_VERSION: Final = "2025-04-01-preview"
_API_BASES: Final = (
    "https://my.openai.azure.com",
    "https://my.openai.azure.com/",
    "https://my.openai.azure.com/openai",
)


class _AzureThatRecordsTheRequest:
    def __init__(self) -> None:
        self.requests: Final[list[httpx.Request]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"value": "ek_upstream"})

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
@pytest.mark.parametrize(
    ("api_base", "url"),
    (
        ("https://gw.example.com/azure/v1", "https://gw.example.com/azure/v1/openai/realtime/transcription_sessions"),
        ("https://my.openai.azure.com/openai", "https://my.openai.azure.com/openai/realtime/transcription_sessions"),
        ("https://my.openai.azure.com/openai/v1", "https://my.openai.azure.com/openai/realtime/transcription_sessions"),
    ),
)
async def test_azure_transcription_session_strips_only_a_trailing_openai_path(api_base: str, url: str) -> None:
    azure: Final = _AzureThatRecordsTheRequest()
    await litellm.acreate_realtime_transcription_session(
        model="azure/gpt-realtime-whisper-deployment",
        api_base=api_base,
        api_key="azure-key",
        api_version=_DATED_PREVIEW_API_VERSION,
        client=azure.client(),
    )
    (request,) = azure.requests
    assert str(request.url) == f"{url}?api-version={_DATED_PREVIEW_API_VERSION}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller_headers", "sent_headers"),
    (
        pytest.param(
            {"Authorization": "Bearer caller-token"},
            {"authorization": "Bearer caller-token", "api-key": None},
            id="caller-authorization",
        ),
        pytest.param(
            {"authorization": "Bearer caller-token"},
            {"authorization": "Bearer caller-token", "api-key": None},
            id="caller-lowercase-authorization",
        ),
        pytest.param({"api-key": "caller-key"}, {"authorization": None, "api-key": "caller-key"}, id="caller-api-key"),
        pytest.param({}, {"authorization": "Bearer entra-token", "api-key": None}, id="no-caller-auth"),
    ),
)
async def test_azure_entra_fallback_never_overrides_the_callers_own_auth_header(
    caller_headers: Mapping[str, str],
    sent_headers: Mapping[str, str | None],
    monkeypatch: pytest.MonkeyPatch,
    no_ambient_azure_credentials: None,
) -> None:
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_AD_TOKEN", "entra-token")
    azure: Final = _AzureThatRecordsTheRequest()
    await litellm.acreate_realtime_client_secret(
        model="azure/gpt-realtime-deployment",
        api_base="https://my.openai.azure.com",
        extra_headers=caller_headers,
        client=azure.client(),
    )
    (request,) = azure.requests
    assert {name: request.headers.get(name) for name in sent_headers} == sent_headers
