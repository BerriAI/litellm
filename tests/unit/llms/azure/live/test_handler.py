from typing import Final

import pytest

from litellm.llms.azure.live.handler import AzureLiveApiBaseMissing, LiveUpstream, azure_live_upstream


@pytest.mark.parametrize(
    "api_base",
    [
        "https://my-resource.openai.azure.com",
        "https://my-resource.openai.azure.com/",
        "https://my-resource.openai.azure.com/openai",
        "https://my-resource.openai.azure.com/openai/v1/",
        "https://my-resource.openai.azure.com/openai/deployments/gpt-live-1?api-version=2025-04-01-preview",
    ],
)
def test_azure_live_upstream_targets_the_v1_live_sessions_route_whatever_path_the_api_base_carries(
    api_base: str,
) -> None:
    upstream: Final = azure_live_upstream(api_base=api_base, api_key="azure-key", azure_ad_token=None)

    assert isinstance(upstream, LiveUpstream)
    assert upstream.url == "wss://my-resource.openai.azure.com/openai/v1/live/sessions"


def test_azure_live_upstream_keeps_a_plaintext_api_base_on_ws() -> None:
    upstream: Final = azure_live_upstream(api_base="http://localhost:8080", api_key="azure-key", azure_ad_token=None)

    assert isinstance(upstream, LiveUpstream)
    assert upstream.url == "ws://localhost:8080/openai/v1/live/sessions"


def test_azure_live_upstream_sends_a_static_key_as_api_key_header_only() -> None:
    upstream: Final = azure_live_upstream(
        api_base="https://my-resource.openai.azure.com", api_key="azure-key", azure_ad_token="entra-token"
    )

    assert isinstance(upstream, LiveUpstream)
    assert dict(upstream.headers) == {"api-key": "azure-key"}


def test_azure_live_upstream_sends_an_entra_token_as_bearer_when_no_key_is_set() -> None:
    upstream: Final = azure_live_upstream(
        api_base="https://my-resource.openai.azure.com", api_key=None, azure_ad_token="entra-token"
    )

    assert isinstance(upstream, LiveUpstream)
    assert dict(upstream.headers) == {"Authorization": "Bearer entra-token"}


def test_azure_live_upstream_refuses_to_connect_without_credentials() -> None:
    with pytest.raises(ValueError, match="Missing Azure credentials"):
        azure_live_upstream(api_base="https://my-resource.openai.azure.com", api_key=None, azure_ad_token=None)


@pytest.mark.parametrize("api_base", ["", "my-resource.openai.azure.com"])
def test_azure_live_upstream_refuses_an_api_base_without_a_host(api_base: str) -> None:
    assert azure_live_upstream(api_base=api_base, api_key="azure-key", azure_ad_token=None) == AzureLiveApiBaseMissing()
