from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

_DATED_PREVIEW_API_VERSION: Final = "2025-04-01-preview"
_API_BASES: Final = ("https://my.openai.azure.com", "https://my.openai.azure.com/", "https://my.openai.azure.com/openai")


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
