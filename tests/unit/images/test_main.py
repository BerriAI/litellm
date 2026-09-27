import json
from typing import Final

import httpx
import pytest
import respx

import litellm


def test_image_generation_keeps_an_internal_prefixed_kwarg_out_of_the_provider_request(
    respx_mock: respx.MockRouter,
) -> None:
    api_base: Final = "http://localhost:12346/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/images/generations.*").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})
    )

    litellm.image_generation(
        model="openai/gpt-image-1",
        prompt="a red circle",
        api_base=api_base,
        api_key="fake_openai_api_key",
        _litellm_undeclared_sentinel="internal",
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert "_litellm_undeclared_sentinel" not in sent, sent
    assert sent["prompt"] == "a red circle"


def test_azure_image_generation_sends_azure_ad_token_as_bearer_header_not_body_field(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    api_base: Final = "https://resource.services.ai.azure.com"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/openai/deployments/gpt-image-deployment/.*").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})
    )

    litellm.image_generation(
        model="azure/gpt-image-deployment",
        prompt="a red circle",
        api_base=api_base,
        api_version="2025-04-01-preview",
        azure_ad_token="entra-token",
        seed=42,
    )

    assert mock_route.called
    request: Final = respx_mock.calls[0].request
    sent: Final = json.loads(request.content)
    assert "entra-token" not in request.content.decode()
    assert sent["seed"] == 42
    assert request.headers["authorization"] == "Bearer entra-token"
