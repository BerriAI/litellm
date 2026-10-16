"""
Entra ID / OAuth auth for Azure AI Foundry routes.

Every azure_ai route must authenticate with an Entra ID token when no API key is configured,
instead of requiring an API key.
"""

from unittest.mock import patch

import pytest

import litellm
from litellm.llms.azure_ai.common_utils import (
    get_azure_ai_agent_entra_token,
    get_azure_ai_auth_headers,
    has_azure_entra_params,
    resolve_azure_ai_agent_auth_header,
)

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
