"""
Entra ID / OAuth auth for Azure AI Foundry routes.

Every azure_ai route must authenticate with an Entra ID token when no API key is configured,
instead of requiring an API key.
"""

import json
import queue
from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.azure_ai.common_utils import (
    get_azure_ai_agent_entra_token,
    get_azure_ai_auth_headers,
    has_azure_entra_params,
    resolve_azure_ai_agent_auth_header,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler

ENTRA_PARAMS = {"azure_ad_token": "entra-token"}


@pytest.fixture(autouse=True)
def clear_azure_env(monkeypatch):
    for env_var in (
        "AZURE_AI_API_KEY",
        "AZURE_API_KEY",
        "AZURE_AD_TOKEN",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_SCOPE",
        "OPENAI_API_KEY",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)


def test_api_key_wins_over_entra_credentials():
    headers = get_azure_ai_auth_headers(api_key="my-key", litellm_params=ENTRA_PARAMS, api_key_header="Api-Key")

    assert headers == {"Api-Key": "my-key"}


def test_entra_token_used_when_no_api_key():
    headers = get_azure_ai_auth_headers(api_key=None, litellm_params=ENTRA_PARAMS, api_key_header="Api-Key")

    assert headers == {"Authorization": "Bearer entra-token"}


def test_service_principal_token_is_requested_with_the_configured_scope():
    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to assert the SP credential+scope plumbing and the returned Bearer header; live SP path proven by the PR's Azure Foundry e2e QA
        mock_entra_id.return_value = lambda: "sp-token"

        headers = get_azure_ai_auth_headers(
            api_key=None,
            litellm_params={
                "tenant_id": "tenant",
                "client_id": "client",
                "client_secret": "secret",
                "azure_scope": "https://ai.azure.com/.default",
            },
        )

    mock_entra_id.assert_called_once_with(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret",
        scope="https://ai.azure.com/.default",
    )
    assert headers == {"Authorization": "Bearer sp-token"}


def test_error_mentions_both_credential_types_when_nothing_is_configured():
    with pytest.raises(ValueError, match="AZURE_AI_API_KEY") as exc_info:
        get_azure_ai_auth_headers(api_key=None, litellm_params={})

    message = str(exc_info.value)
    assert "AZURE_AI_API_KEY" in message
    assert "client_secret" in message


def test_embedding_falls_back_to_entra_token_instead_of_openai_key(monkeypatch):  # test-quality-ok: asserts the embedding handler is authed with the Entra token, not the OpenAI key fallback; live path proven by the PR's Azure Foundry e2e QA
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")

    with patch.object(litellm.main.azure_ai_embedding, "embedding") as mock_embedding:  # test-quality-ok: no injection seam for the embedding handler through the public embedding() API; live path proven by the PR's Azure Foundry e2e QA
        mock_embedding.return_value = litellm.EmbeddingResponse()

        litellm.embedding(
            model="azure_ai/cohere-embed-v3-english",
            input=["hello"],
            api_base="https://my-resource.services.ai.azure.com",
            azure_ad_token="entra-token",
        )

    assert mock_embedding.call_args.kwargs["api_key"] == "entra-token"


def test_image_generation_authenticates_with_entra_token():
    with patch.object(litellm.images.main.azure_chat_completions, "image_generation") as mock_image_generation:  # test-quality-ok: asserts image_generation forwards the computed Entra bearer header; no injection seam through the public API; live path proven by the PR's Azure Foundry e2e QA
        mock_image_generation.return_value = litellm.ImageResponse()

        litellm.image_generation(
            model="azure_ai/FLUX-1.1-pro",
            prompt="a red circle",
            api_base="https://my-resource.services.ai.azure.com",
            azure_ad_token="entra-token",
        )

    headers = mock_image_generation.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer entra-token"
    assert "api-key" not in headers


@pytest.mark.parametrize("header_name", ["Authorization", "authorization", "api-key", "API-KEY"])
def test_image_generation_keeps_caller_supplied_auth_header(header_name):
    with patch.object(litellm.images.main.azure_chat_completions, "image_generation") as mock_image_generation:  # test-quality-ok: asserts a caller-supplied auth header is preserved over Entra; no injection seam through the public API; live path proven by the PR's Azure Foundry e2e QA
        mock_image_generation.return_value = litellm.ImageResponse()

        litellm.image_generation(
            model="azure_ai/FLUX-1.1-pro",
            prompt="a red circle",
            api_base="https://my-resource.services.ai.azure.com",
            headers={header_name: "caller-credential"},
        )

    headers = mock_image_generation.call_args.kwargs["headers"]
    assert headers[header_name] == "caller-credential"
    assert len(headers) == 2


def test_image_generation_still_uses_api_key_header():
    with patch.object(litellm.images.main.azure_chat_completions, "image_generation") as mock_image_generation:  # test-quality-ok: asserts the api-key header path still works alongside Entra; no injection seam through the public API; live path proven by the PR's Azure Foundry e2e QA
        mock_image_generation.return_value = litellm.ImageResponse()

        litellm.image_generation(
            model="azure_ai/FLUX-1.1-pro",
            prompt="a red circle",
            api_base="https://my-resource.services.ai.azure.com",
            api_key="my-key",
        )

    headers = mock_image_generation.call_args.kwargs["headers"]
    assert headers["api-key"] == "my-key"
    assert "Authorization" not in headers


def test_agents_without_entra_credentials_are_not_treated_as_entra_agents():
    """Only a credential-bearing field opts an agent into Entra auth: scope or identity fields alone
    must never make the proxy mint a bearer for that agent's URL."""
    assert has_azure_entra_params({"api_key": "static", "headers": {"x": "y"}}) is False
    assert has_azure_entra_params(None) is False
    assert has_azure_entra_params({"azure_scope": "https://ai.azure.com/.default"}) is False
    assert has_azure_entra_params({"tenant_id": "t", "client_id": "c"}) is False
    assert has_azure_entra_params({"azure_ad_token": "entra-token"}) is True
    assert has_azure_entra_params({"tenant_id": "t", "client_id": "c", "client_secret": "s"}) is True
    assert has_azure_entra_params({"client_id": "c", "azure_username": "u", "azure_password": "p"}) is True


def test_agent_entra_token_ignores_the_process_wide_azure_credentials(monkeypatch):
    """The azure provider's token helper falls back to AZURE_* env vars. An agent's bearer must come
    from that agent's own litellm_params only, or the host's service principal would authenticate to
    whatever URL an agent registers."""
    monkeypatch.setenv("AZURE_TENANT_ID", "host-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "host-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "host-secret")
    monkeypatch.setenv("AZURE_AD_TOKEN", "host-token")

    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch so a host-credential leak would show up as a call instead of a network round trip
        mock_entra_id.return_value = lambda: "host-sp-token"

        with pytest.raises(ValueError, match="client_secret"):
            get_azure_ai_agent_entra_token({"azure_scope": "https://ai.azure.com/.default"})
        assert get_azure_ai_agent_entra_token({"azure_ad_token": "agent-token"}) == "agent-token"

    mock_entra_id.assert_not_called()


def test_agent_service_principal_fields_resolve_os_environ_references(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_TENANT_ID", "tenant-from-env")
    monkeypatch.setenv("FOUNDRY_AGENT_CLIENT_ID", "client-from-env")
    monkeypatch.setenv("FOUNDRY_AGENT_CLIENT_SECRET", "secret-from-env")

    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to assert the resolved secret values reach the credential; live SP path proven by the PR's Azure Foundry e2e QA
        mock_entra_id.return_value = lambda: "sp-token"

        token = get_azure_ai_agent_entra_token(
            {
                "tenant_id": "os.environ/FOUNDRY_AGENT_TENANT_ID",
                "client_id": "os.environ/FOUNDRY_AGENT_CLIENT_ID",
                "client_secret": "os.environ/FOUNDRY_AGENT_CLIENT_SECRET",
            }
        )

    mock_entra_id.assert_called_once_with(
        tenant_id="tenant-from-env",
        client_id="client-from-env",
        client_secret="secret-from-env",
        scope="https://ai.azure.com/.default",
    )
    assert token == "sp-token"


def test_agent_service_principal_wins_over_a_static_token_on_the_same_agent():
    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to pin the precedence between a refreshing credential and a static token
        mock_entra_id.return_value = lambda: "sp-token"

        token = get_azure_ai_agent_entra_token(
            {"tenant_id": "tenant", "client_id": "client", "client_secret": "secret", "azure_ad_token": "stale-token"}
        )

    assert token == "sp-token"


def test_agent_service_principal_token_defaults_to_the_foundry_agents_scope():
    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to assert the scope Foundry agents require reaches the credential; live SP path proven by the PR's Azure Foundry e2e QA
        mock_entra_id.return_value = lambda: "sp-token"

        token = get_azure_ai_agent_entra_token({"tenant_id": "tenant", "client_id": "client", "client_secret": "secret"})

    mock_entra_id.assert_called_once_with(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret",
        scope="https://ai.azure.com/.default",
    )
    assert token == "sp-token"


def test_agent_azure_scope_overrides_the_foundry_agents_default():
    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to assert an explicit azure_scope wins over the agents default; live SP path proven by the PR's Azure Foundry e2e QA
        mock_entra_id.return_value = lambda: "sp-token"

        get_azure_ai_agent_entra_token(
            {"tenant_id": "tenant", "client_id": "client", "client_secret": "secret", "azure_scope": "custom/.default"}
        )

    assert mock_entra_id.call_args.kwargs["scope"] == "custom/.default"


def test_agent_entra_values_resolve_os_environ_references(monkeypatch):
    monkeypatch.setenv("FOUNDRY_AGENT_AD_TOKEN", "token-from-env")

    assert get_azure_ai_agent_entra_token({"azure_ad_token": "os.environ/FOUNDRY_AGENT_AD_TOKEN"}) == "token-from-env"


_AGENT_APP_CLIENT_ID: Final = "0d08e61f-6184-4980-9cef-a2e9128f4cb2"
_AGENT_CREDENTIALS: Final = {
    "service_principal": (
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id",
        {"tenant_id": "tenant", "client_id": _AGENT_APP_CLIENT_ID, "client_secret": "wrong-secret"},
    ),
    "username_password": (
        "litellm.llms.azure.common_utils.get_azure_ad_token_from_username_password",
        {"client_id": _AGENT_APP_CLIENT_ID, "azure_username": "agent@example.com", "azure_password": "wrong"},
    ),
}


def _raising(error: Exception):
    def token() -> str:
        raise error

    return token


@pytest.mark.parametrize(("token_fetch", "agent_params"), _AGENT_CREDENTIALS.values(), ids=_AGENT_CREDENTIALS.keys())
def test_agent_entra_rejection_raises_401_with_only_the_aadsts_reason(token_fetch, agent_params):
    from azure.core.exceptions import ClientAuthenticationError

    rejection: Final = ClientAuthenticationError(
        message="Authentication failed: AADSTS7000215: Invalid client secret provided. Ensure the secret being sent "
        f"in the request is the client secret value, for a secret added to app '{_AGENT_APP_CLIENT_ID}'."
    )

    with patch(token_fetch, return_value=_raising(rejection)):  # test-quality-ok: stands in for the Entra token endpoint answering a wrong credential, the shape azure-identity raises
        with pytest.raises(litellm.AuthenticationError) as excinfo:
            get_azure_ai_agent_entra_token(agent_params)

    assert excinfo.value.status_code == 401
    assert "AADSTS7000215" in str(excinfo.value)
    assert _AGENT_APP_CLIENT_ID not in str(excinfo.value)


def test_agent_entra_endpoint_unreachable_is_not_an_auth_error():
    from azure.core.exceptions import ClientAuthenticationError

    unreachable: Final = ClientAuthenticationError(message="Authentication failed: Connection refused")
    token_fetch, agent_params = _AGENT_CREDENTIALS["service_principal"]

    with patch(token_fetch, return_value=_raising(unreachable)):  # test-quality-ok: stands in for an unreachable Entra token endpoint, the shape azure-identity raises
        with pytest.raises(ClientAuthenticationError) as excinfo:
            get_azure_ai_agent_entra_token(agent_params)

    assert not isinstance(excinfo.value, litellm.AuthenticationError)


def test_agent_entra_token_failure_names_the_credential_fields():
    with pytest.raises(ValueError, match="client_secret"):
        get_azure_ai_agent_entra_token({"azure_scope": "https://ai.azure.com/.default"})


def test_agent_oidc_token_without_agent_ids_never_borrows_the_host_identity(monkeypatch):
    """The shared OIDC helper fills a missing client and tenant id from AZURE_CLIENT_ID and AZURE_TENANT_ID,
    which would exchange the host's federated token for the host's identity at that agent's URL."""
    monkeypatch.setenv("AZURE_TENANT_ID", "host-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "host-client")

    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_oidc") as mock_oidc:  # test-quality-ok: stubs the OIDC exchange so a host-identity leak would show up as a call instead of a network round trip
        mock_oidc.return_value = "host-minted-token"

        with pytest.raises(ValueError, match="oidc/"):
            get_azure_ai_agent_entra_token({"azure_ad_token": "oidc/github"})
        with pytest.raises(ValueError, match="oidc/"):
            get_azure_ai_agent_entra_token({"azure_ad_token": "oidc/github", "tenant_id": "agent-tenant"})

    mock_oidc.assert_not_called()


def test_agent_oidc_token_exchanges_with_the_agent_ids_and_scope():
    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_oidc") as mock_oidc:  # test-quality-ok: stubs the OIDC exchange to assert the agent's own ids and the Foundry scope reach it
        mock_oidc.return_value = "agent-minted-token"

        token = get_azure_ai_agent_entra_token(
            {"azure_ad_token": "oidc/github", "tenant_id": "agent-tenant", "client_id": "agent-client"}
        )

    assert token == "agent-minted-token"
    mock_oidc.assert_called_once_with(
        azure_ad_token="oidc/github",
        azure_client_id="agent-client",
        azure_tenant_id="agent-tenant",
        scope="https://ai.azure.com/.default",
    )


@pytest.mark.asyncio
async def test_agent_auth_header_is_the_entra_bearer():
    headers = await resolve_azure_ai_agent_auth_header({"azure_ad_token": "entra-token"})

    assert headers == {"Authorization": "Bearer entra-token"}


OPENAI_CHAT_COMPLETION: Final = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-6.1-sol",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
OPENAI_EMBEDDING: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "cohere-embed-v3-english",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}
ANTHROPIC_MESSAGE: Final = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


class _PreCallOptionalParamsCapture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.optional_params: tuple[dict[str, object], ...] = ()

    def log_pre_api_call(self, model: str, messages: list[object], kwargs: dict[str, dict[str, object]]) -> None:
        self.optional_params = (*self.optional_params, kwargs["optional_params"])


@pytest.mark.parametrize(
    ("model", "provider_response"),
    [
        pytest.param("azure_ai/gpt-6.1-sol", OPENAI_CHAT_COMPLETION, id="openai-shaped"),
        pytest.param("azure_ai/claude-sonnet-5-5", ANTHROPIC_MESSAGE, id="claude"),
    ],
)
def test_completion_sends_the_entra_token_only_as_the_bearer_header(
    model: str, provider_response: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
):
    capture: Final = _PreCallOptionalParamsCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    sent: Final[queue.Queue[httpx.Request]] = queue.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        sent.put_nowait(request)
        return httpx.Response(200, json=provider_response)

    response: Final = litellm.completion(
        model=model,
        api_base="https://example-resource.services.ai.azure.com",
        azure_ad_token="entra-token",
        messages=[{"role": "user", "content": "hi"}],
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
    )

    request: Final = sent.get_nowait()
    assert request.headers["Authorization"] == "Bearer entra-token"
    assert "azure_ad_token" not in json.loads(request.content)
    assert response.choices[0].message.content == "OK"
    assert capture.optional_params
    assert "entra-token" not in json.dumps(capture.optional_params, default=str)


@pytest.mark.respx(assert_all_called=True)
def test_embedding_sends_the_entra_token_only_as_the_bearer_header(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(url__startswith="https://example-resource.services.ai.azure.com").mock(
        return_value=httpx.Response(200, json=OPENAI_EMBEDDING)
    )

    litellm.embedding(
        model="azure_ai/cohere-embed-v3-english",
        api_base="https://example-resource.services.ai.azure.com",
        azure_ad_token="entra-token",
        input=["hello"],
    )

    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer entra-token"
    assert "azure_ad_token" not in json.loads(request.content)


IMAGE_RESPONSE: Final = {"created": 1, "data": [{"b64_json": "aGk="}]}
PNG_BYTES: Final = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize(
    "extra_kwargs",
    [{}, {"extra_body": {"azure_ad_token": "entra-token", "client_secret": "entra-token"}}],
    ids=["kwarg", "extra_body"],
)
def test_image_edit_sends_the_entra_token_only_as_the_bearer_header(
    respx_mock: respx.MockRouter, extra_kwargs: dict[str, dict[str, str]]
):
    route: Final = respx_mock.post(url__startswith="https://example-resource.services.ai.azure.com").mock(
        return_value=httpx.Response(200, json=IMAGE_RESPONSE)
    )

    litellm.image_edit(
        model="azure_ai/FLUX.1-Kontext-pro",
        api_base="https://example-resource.services.ai.azure.com",
        api_version="2025-04-01-preview",
        azure_ad_token="entra-token",
        image=PNG_BYTES,
        prompt="add a hat",
        **extra_kwargs,
    )

    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer entra-token"
    assert b"entra-token" not in request.content


@pytest.mark.respx(assert_all_called=True)
def test_image_generation_sends_the_entra_token_only_as_the_bearer_header(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(url__startswith="https://example-resource.services.ai.azure.com").mock(
        return_value=httpx.Response(200, json=IMAGE_RESPONSE)
    )

    litellm.image_generation(
        model="azure_ai/FLUX-1.1-pro",
        api_base="https://example-resource.services.ai.azure.com",
        azure_ad_token="entra-token",
        prompt="a red circle",
    )

    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer entra-token"
    assert "azure_ad_token" not in json.loads(request.content)


@pytest.mark.respx(assert_all_called=True)
def test_transcription_sends_the_entra_token_only_as_the_bearer_header(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(url__startswith="https://example-resource.services.ai.azure.com").mock(
        return_value=httpx.Response(200, json={"text": "hi"})
    )

    litellm.transcription(
        model="azure_ai/whisper",
        api_base="https://example-resource.services.ai.azure.com",
        api_version="2024-06-01",
        azure_ad_token="entra-token",
        file=("hello.wav", b"RIFF" + b"\x00" * 32),
    )

    request: Final = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer entra-token"
    assert b"entra-token" not in request.content
