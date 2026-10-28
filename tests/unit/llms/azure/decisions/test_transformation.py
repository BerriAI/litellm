import json
import threading
import time
from collections.abc import Iterator, Mapping
from typing import Final
from unittest.mock import Mock

import pytest
import respx
from azure.core.credentials import AccessToken

import litellm
from litellm.llms.azure import common_utils
from litellm.llms.azure.decisions.transformation import AzureDecisionsConfig
from litellm.types.decisions import DecisionsResponse, OpenAIDecisionResponse

_BASE: Final = "https://resource.openai.azure.com"
_URL: Final = f"{_BASE}/openai/v1/decisions"
_RESPONSE: Final = {
    "model": "luna-deployment",
    "answers": [{"type": "predicate", "name": "defect", "probability": 0.9}],
    "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
}


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "AZURE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_BASE",
        "AZURE_AD_TOKEN",
        "AZURE_CLIENT_ID",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_SCOPE",
        "AZURE_USERNAME",
        "AZURE_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "azure_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    common_utils._cached_azure_ad_token_refresh_provider.cache_clear()
    yield
    common_utils._cached_azure_ad_token_refresh_provider.cache_clear()


async def call_decisions(
    is_async: bool,
    systemone: bool,
    credentials: Mapping[str, object],
    api_base: str = _BASE,
) -> DecisionsResponse | OpenAIDecisionResponse:
    params = {
        "model": "azure/luna-deployment",
        "api_base": api_base,
        **credentials,
    }
    if systemone:
        params["state"] = "The export job hangs."
        params["questions"] = {"defect": {"type": "noul", "instructions": "Is this a defect?"}}
        if is_async:
            return await litellm.asystemone(**params)
        return litellm.systemone(**params)
    params["input"] = "The export job hangs."
    params["questions"] = [{"type": "predicate", "name": "defect", "instructions": "Is this a defect?"}]
    if is_async:
        return await litellm.adecisions(**params)
    return litellm.decisions(**params)


@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.parametrize("systemone", (False, True))
@pytest.mark.parametrize("suffix", ("", "/", "/openai", "/openai/v1/"))
async def test_azure_decisions_uses_entra_token_and_openai_wire_format(
    respx_mock: respx.MockRouter, is_async: bool, systemone: bool, suffix: str
) -> None:
    route = respx_mock.post(_URL).respond(json=_RESPONSE)
    response = await call_decisions(is_async, systemone, {"azure_ad_token": "synthetic-entra"}, _BASE + suffix)
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer synthetic-entra"
    assert "api-key" not in request.headers
    assert json.loads(request.content) == {
        "model": "luna-deployment",
        "input": "The export job hangs.",
        "questions": [{"type": "predicate", "name": "defect", "instructions": "Is this a defect?"}],
    }
    if systemone:
        assert response.model_dump()["answers"]["defect"]["noul"] == 0.9
    else:
        assert response.model_dump()["answers"][0]["probability"] == 0.9
    assert response._hidden_params["custom_llm_provider"] == "azure"


@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.parametrize("systemone", (False, True))
async def test_default_credential_refreshes_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    is_async: bool,
    systemone: bool,
) -> None:
    credential = Mock(spec=["get_token"])
    credential.get_token.side_effect = [
        AccessToken("synthetic-first", int(time.time()) + 120),
        AccessToken("synthetic-refreshed", int(time.time()) + 3600),
    ]
    factory = Mock(return_value=credential)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unrelated-openai-key")
    route = respx_mock.post(_URL).respond(json=_RESPONSE)
    await call_decisions(is_async, systemone, {})
    await call_decisions(is_async, systemone, {})
    assert [call.request.headers["authorization"] for call in route.calls] == [
        "Bearer synthetic-first",
        "Bearer synthetic-refreshed",
    ]
    factory.assert_called_once_with()
    assert credential.get_token.call_count == 2
    assert all(
        call.args == ("https://cognitiveservices.azure.com/.default",) for call in credential.get_token.call_args_list
    )


@pytest.mark.parametrize("kind", ("service-principal", "oidc", "provider"))
async def test_azure_decisions_reuses_existing_azure_credential_resolution(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, kind: str
) -> None:
    provider = Mock(return_value="synthetic-entra")
    if kind == "service-principal":
        factory = Mock(return_value=provider)
        monkeypatch.setattr(common_utils, "get_azure_ad_token_from_entra_id", factory)
        credentials = {
            "tenant_id": "tenant",
            "client_id": "client",
            "client_secret": "synthetic-secret",
        }
    elif kind == "oidc":
        monkeypatch.setattr(common_utils, "get_azure_ad_token_from_oidc", provider)
        credentials = {
            "tenant_id": "tenant",
            "client_id": "client",
            "azure_ad_token": "oidc/azure/https://cognitiveservices.azure.com/.default",
        }
    else:
        credentials = {"azure_ad_token_provider": provider}
    route = respx_mock.post(_URL).respond(json=_RESPONSE)
    await call_decisions(True, False, credentials)
    assert route.calls.last.request.headers["authorization"] == "Bearer synthetic-entra"
    provider.assert_called_once()
    if kind == "service-principal":
        factory.assert_called_once_with(
            tenant_id="tenant",
            client_id="client",
            client_secret="synthetic-secret",
            scope="https://cognitiveservices.azure.com/.default",
        )
    if kind == "oidc":
        provider.assert_called_once_with(
            azure_ad_token=credentials["azure_ad_token"],
            azure_client_id="client",
            azure_tenant_id="tenant",
            scope="https://cognitiveservices.azure.com/.default",
        )


@pytest.mark.parametrize("key_source", ("api_key", "AZURE_API_KEY", "AZURE_OPENAI_API_KEY"))
async def test_azure_api_key_takes_precedence_over_entra(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, key_source: str
) -> None:
    provider = Mock(side_effect=AssertionError("API key must bypass Entra"))
    credentials = {"azure_ad_token_provider": provider}
    if key_source == "api_key":
        credentials["api_key"] = "synthetic-key"
    else:
        monkeypatch.setenv(key_source, "synthetic-key")
    route = respx_mock.post(_URL).respond(json=_RESPONSE)
    await call_decisions(True, False, credentials)
    assert route.calls.last.request.headers["api-key"] == "synthetic-key"
    assert "authorization" not in route.calls.last.request.headers
    provider.assert_not_called()


async def test_azure_credential_lookup_does_not_block_the_event_loop(
    respx_mock: respx.MockRouter,
) -> None:
    event_loop_thread = threading.get_ident()

    def provider() -> str:
        assert threading.get_ident() != event_loop_thread
        return "synthetic-entra"

    respx_mock.post(_URL).respond(json=_RESPONSE)
    await call_decisions(True, False, {"azure_ad_token_provider": provider})


def test_azure_decisions_requires_azure_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://unrelated.example")
    with pytest.raises(litellm.BadRequestError, match="api_base is required"):
        litellm.decisions(
            model="azure/luna-deployment",
            input="Text",
            questions=[
                {
                    "type": "predicate",
                    "name": "defect",
                    "instructions": "Is this a defect?",
                }
            ],
        )


def test_azure_decisions_does_not_echo_invalid_credentials() -> None:
    with pytest.raises(litellm.BadRequestError) as error:
        AzureDecisionsConfig().validate_environment(
            {},
            "luna-deployment",
            None,
            {"azure_ad_token": {"secret": "sensitive-value"}},
        )
    assert "sensitive-value" not in str(error.value)
