"""Image generation must send the Entra ID token when the deployment has no API key.

Image generation builds its own httpx request instead of going through the
Azure SDK client, so a credential resolved by ``initialize_azure_sdk_client``
only reaches the wire if it is put on the headers. Without that, a key-less
deployment sends no credential at all and Azure answers
``401 Access denied due to invalid subscription key`` — on the same account
and identity that serve chat, embeddings and transcription key-less.
"""

from unittest.mock import MagicMock

import pytest

from litellm.llms.azure.azure import AzureChatCompletion


def _set_header(**kwargs):
    headers = kwargs.pop("headers")
    AzureChatCompletion._set_azure_ad_auth_header(headers=headers, **kwargs)
    return headers


class TestSetAzureADAuthHeader:
    def test_token_provider_from_client_params_becomes_bearer_header(self):
        """The managed-identity provider lands in azure_client_params, not in the
        caller's argument, when it comes from enable_azure_ad_token_refresh."""
        headers = _set_header(
            headers={"Content-Type": "application/json"},
            api_key=None,
            azure_ad_token=None,
            azure_ad_token_provider=None,
            azure_client_params={"azure_ad_token_provider": lambda: "mi-token"},
        )
        assert headers["Authorization"] == "Bearer mi-token"

    def test_token_from_client_params_becomes_bearer_header(self):
        headers = _set_header(
            headers={},
            api_key=None,
            azure_ad_token=None,
            azure_ad_token_provider=None,
            azure_client_params={"azure_ad_token": "static-token"},
        )
        assert headers["Authorization"] == "Bearer static-token"

    def test_explicit_provider_wins_over_client_params(self):
        headers = _set_header(
            headers={},
            api_key=None,
            azure_ad_token=None,
            azure_ad_token_provider=lambda: "explicit-token",
            azure_client_params={"azure_ad_token_provider": lambda: "fallback-token"},
        )
        assert headers["Authorization"] == "Bearer explicit-token"

    def test_stale_api_key_header_is_replaced_by_the_token(self):
        headers = _set_header(
            headers={"api-key": ""},
            api_key=None,
            azure_ad_token=None,
            azure_ad_token_provider=lambda: "mi-token",
            azure_client_params={},
        )
        assert "api-key" not in headers
        assert headers["Authorization"] == "Bearer mi-token"

    def test_api_key_deployment_is_left_alone(self):
        """A keyed deployment must keep its api-key header and gain no bearer."""
        headers = _set_header(
            headers={"api-key": "sk-azure"},
            api_key="sk-azure",
            azure_ad_token=None,
            azure_ad_token_provider=lambda: "mi-token",
            azure_client_params={"azure_ad_token_provider": lambda: "mi-token"},
        )
        assert headers["api-key"] == "sk-azure"
        assert "Authorization" not in headers

    def test_caller_supplied_authorization_is_not_overwritten(self):
        headers = _set_header(
            headers={"Authorization": "Bearer caller-token"},
            api_key=None,
            azure_ad_token=None,
            azure_ad_token_provider=lambda: "mi-token",
            azure_client_params={},
        )
        assert headers["Authorization"] == "Bearer caller-token"

    def test_no_credential_anywhere_leaves_headers_unchanged(self):
        headers = _set_header(
            headers={"Content-Type": "application/json"},
            api_key=None,
            azure_ad_token=None,
            azure_ad_token_provider=None,
            azure_client_params={},
        )
        assert headers == {"Content-Type": "application/json"}


@pytest.mark.parametrize("provider_key", ["azure_ad_token_provider", "azure_ad_token"])
def test_image_generation_request_carries_the_bearer_token(monkeypatch, provider_key):
    """End to end through image_generation: the outbound request is authenticated."""
    import httpx

    from litellm.types.utils import ImageResponse

    azure = AzureChatCompletion()
    captured: dict = {}

    value = (lambda: "mi-token") if provider_key == "azure_ad_token_provider" else "mi-token"

    monkeypatch.setattr(
        AzureChatCompletion,
        "initialize_azure_sdk_client",
        lambda self, **kwargs: {
            "azure_endpoint": "https://acct.openai.azure.com",
            "api_version": "2025-04-01-preview",
            provider_key: value,
        },
    )

    def fake_request(self, *, headers, **kwargs):
        captured["headers"] = headers
        return httpx.Response(
            status_code=200,
            json={"created": 1, "data": [{"b64_json": "aGk="}]},
            request=httpx.Request("POST", "https://acct.openai.azure.com"),
        )

    monkeypatch.setattr(AzureChatCompletion, "make_sync_azure_httpx_request", fake_request)

    azure.image_generation(
        prompt="a cat",
        timeout=60.0,
        optional_params={},
        logging_obj=MagicMock(),
        headers={"Content-Type": "application/json"},
        model="gpt-image-1",
        api_key=None,
        api_base="https://acct.openai.azure.com",
        api_version="2025-04-01-preview",
        model_response=ImageResponse(),
        litellm_params={},
    )

    assert captured["headers"]["Authorization"] == "Bearer mi-token"
    assert "api-key" not in captured["headers"]
