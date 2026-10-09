"""
Tests for the credential every Anthropic count-tokens request carries.

The count-tokens handler receives the auth header that ``AnthropicModelInfo.get_auth_header``
resolved, so a static key, an OAuth token (sk-ant-oat*), ``ANTHROPIC_AUTH_TOKEN`` and a minted
workload-identity token all reach Anthropic exactly the way chat on the same deployment does.

Regression tests for https://github.com/BerriAI/litellm/issues/22040 and for the
``ANTHROPIC_AUTH_TOKEN`` gap where count-tokens skipped minting but forwarded no credential.
"""

import os
import sys
from typing import Final

import httpx
import pytest
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import litellm
from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)
from litellm.llms.azure_ai.anthropic.count_tokens.token_counter import (
    AzureAIAnthropicTokenCounter,
)
from litellm.types.llms.anthropic import ANTHROPIC_OAUTH_BETA_HEADER
from litellm.types.utils import TokenCountResponse

# Fake tokens for testing (not real secrets)
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"

FEDERATED_DEPLOYMENT = {
    "litellm_params": {
        "model": "anthropic/claude-sonnet-4-5",
        "anthropic_federation_rule_id": "fdrl_x",
        "anthropic_organization_id": "org-x",
    }
}


def count_tokens_headers_for(api_key: str) -> dict[str, str]:
    auth_header = AnthropicModelInfo.get_auth_header(api_key=api_key)
    assert auth_header is not None
    return AnthropicCountTokensConfig().get_count_tokens_headers(auth_header)


@pytest.fixture
def httpx_transport_clients(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    client_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if client_cache is not None:
        client_cache.flush_cache()
    yield
    if client_cache is not None:
        client_cache.flush_cache()


class TestCountTokensOAuthHeaders:
    """Tests that count_tokens headers are correct for both regular and OAuth keys."""

    def test_regular_api_key_uses_x_api_key(self):
        """Regular API keys should be sent via x-api-key header."""
        headers = count_tokens_headers_for(FAKE_REGULAR_KEY)

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_oauth_key_uses_bearer_authorization(self):
        """OAuth tokens (sk-ant-oat*) should be sent via Authorization: Bearer."""
        headers = count_tokens_headers_for(FAKE_OAUTH_TOKEN)

        assert headers.get("authorization") == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in headers

    def test_oauth_key_sets_oauth_beta_header(self):
        """OAuth tokens should trigger the anthropic-beta oauth header."""
        headers = count_tokens_headers_for(FAKE_OAUTH_TOKEN)

        assert ANTHROPIC_OAUTH_BETA_HEADER in headers.get("anthropic-beta", "").split(",")

    def test_regular_key_preserves_token_counting_beta(self):
        """Regular keys should keep the token-counting beta header."""
        headers = count_tokens_headers_for(FAKE_REGULAR_KEY)

        assert headers.get("anthropic-beta") == ANTHROPIC_TOKEN_COUNTING_BETA_VERSION

    def test_headers_always_have_content_type(self):
        """Both regular and OAuth paths should have Content-Type."""
        for key in [FAKE_REGULAR_KEY, FAKE_OAUTH_TOKEN]:
            headers = count_tokens_headers_for(key)
            assert headers["Content-Type"] == "application/json"

    def test_headers_always_have_anthropic_version(self):
        """Both paths should have anthropic-version."""
        for key in [FAKE_REGULAR_KEY, FAKE_OAUTH_TOKEN]:
            headers = count_tokens_headers_for(key)
            assert headers["anthropic-version"] == "2023-06-01"

    def test_oauth_key_preserves_token_counting_beta(self):
        """OAuth tokens must preserve the token-counting beta alongside the OAuth beta."""
        headers = count_tokens_headers_for(FAKE_OAUTH_TOKEN)

        betas = headers.get("anthropic-beta", "").split(",")
        assert ANTHROPIC_TOKEN_COUNTING_BETA_VERSION in betas, f"token-counting beta missing: {betas}"
        assert ANTHROPIC_OAUTH_BETA_HEADER in betas, f"oauth beta missing: {betas}"


class TestCountTokensUsesWorkloadIdentity:
    """A federated deployment holds no static key. Without minting one, count_tokens returns None
    and the caller silently falls back to the local tokenizer, so the number a federated
    deployment reports would never come from Anthropic."""

    @pytest.mark.asyncio
    async def test_a_federated_deployment_mints_and_counts(self, monkeypatch):
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        minted = "sk-ant-oat01-minted-for-count"

        async def fake_mint(_params, _api_base, _model):
            return minted

        monkeypatch.setattr("litellm.llms.anthropic.common_utils.aget_anthropic_wif_token", fake_mint)

        seen: dict[str, object] = {}

        async def fake_request(**kwargs):
            seen.update(kwargs)
            return {"input_tokens": 42}

        monkeypatch.setattr(
            token_counter_module.anthropic_count_tokens_handler,
            "handle_count_tokens_request",
            fake_request,
            raising=False,
        )

        result = await token_counter_module.AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment=FEDERATED_DEPLOYMENT,
            request_model="claude-sonnet-4-5",
        )

        assert result is not None
        assert result.total_tokens == 42
        assert seen["auth_header"] == {
            "authorization": f"Bearer {minted}",
            "anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER,
        }

    @pytest.mark.asyncio
    async def test_an_auth_token_deployment_counts_with_a_bearer_and_never_mints(
        self, monkeypatch, httpx_transport_clients
    ):
        """With only ``ANTHROPIC_AUTH_TOKEN`` set, chat on a federated deployment authenticates with
        that token, so count-tokens must send the same Bearer instead of silently returning None."""
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token-for-testing")

        async def fake_mint(_params, _api_base, _model):
            raise AssertionError("an auth-token deployment must never mint a federated token")

        monkeypatch.setattr("litellm.llms.anthropic.common_utils.aget_anthropic_wif_token", fake_mint)

        with respx.mock(assert_all_called=True) as router:
            route = router.post("https://api.anthropic.com/v1/messages/count_tokens").mock(
                return_value=httpx.Response(200, json={"input_tokens": 11})
            )
            result = await token_counter_module.AnthropicTokenCounter().count_tokens(
                model_to_use="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "hi"}],
                contents=None,
                deployment=FEDERATED_DEPLOYMENT,
                request_model="claude-sonnet-4-5",
            )

        assert result is not None
        assert result.total_tokens == 11
        assert result.tokenizer_type == "anthropic_api"
        sent = route.calls.last.request.headers
        assert sent["authorization"] == "Bearer bearer-token-for-testing"
        assert "x-api-key" not in sent
        betas = sent["anthropic-beta"].split(",")
        assert ANTHROPIC_TOKEN_COUNTING_BETA_VERSION in betas
        assert ANTHROPIC_OAUTH_BETA_HEADER not in betas

    @pytest.mark.asyncio
    async def test_a_failed_mint_degrades_like_an_anthropic_error(self, monkeypatch):
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        async def failing_mint(_params, _api_base, model):
            raise litellm.AuthenticationError(
                message="federation_rule_id is not a well-formed fdrl_ tagged ID",
                llm_provider="anthropic",
                model=model,
            )

        monkeypatch.setattr("litellm.llms.anthropic.common_utils.aget_anthropic_wif_token", failing_mint)

        result = await token_counter_module.AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "anthropic_federation_rule_id": "not-a-rule",
                    "anthropic_organization_id": "org-x",
                }
            },
            request_model="claude-sonnet-4-5",
        )

        assert result is not None
        assert result.error is True
        assert result.status_code == 401
        assert result.total_tokens == 0
        assert "fdrl_" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_a_vault_backed_static_key_never_mints(self, monkeypatch):
        from litellm.llms.anthropic.count_tokens import token_counter as token_counter_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        vault_key = "sk-ant-api03-only-in-the-vault"

        def vault_only(secret_name, default_value=None):
            return vault_key if secret_name == "ANTHROPIC_API_KEY" else None

        monkeypatch.setattr("litellm.secret_managers.main.get_secret_str", vault_only, raising=False)

        async def fake_mint(_params, _api_base, _model):
            raise AssertionError("a static key must never mint a federated token")

        monkeypatch.setattr("litellm.llms.anthropic.common_utils.aget_anthropic_wif_token", fake_mint)

        seen: dict[str, object] = {}

        async def fake_request(**kwargs):
            seen.update(kwargs)
            return {"input_tokens": 7}

        monkeypatch.setattr(
            token_counter_module.anthropic_count_tokens_handler,
            "handle_count_tokens_request",
            fake_request,
            raising=False,
        )

        result = await token_counter_module.AnthropicTokenCounter().count_tokens(
            model_to_use="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment=FEDERATED_DEPLOYMENT,
            request_model="claude-sonnet-4-5",
        )

        assert result is not None
        assert result.total_tokens == 7
        assert seen["auth_header"] == {"x-api-key": vault_key}


@pytest.mark.parametrize(
    ("counter_type", "api_base", "endpoint", "tokenizer_type"),
    (
        (
            AnthropicTokenCounter,
            "https://gateway.example",
            "https://gateway.example/v1/messages/count_tokens",
            "anthropic_api",
        ),
        (
            AzureAIAnthropicTokenCounter,
            "https://resource.example",
            "https://resource.example/anthropic/v1/messages/count_tokens",
            "azure_ai_anthropic_api",
        ),
    ),
)
@pytest.mark.parametrize(("status_code", "expected_error"), ((200, False), (401, True)))
@pytest.mark.asyncio
async def test_count_token_counters_return_typed_success_and_error_responses(
    counter_type: type[AnthropicTokenCounter] | type[AzureAIAnthropicTokenCounter],
    api_base: str,
    endpoint: str,
    tokenizer_type: str,
    status_code: int,
    expected_error: bool,
    httpx_transport_clients: None,
) -> None:
    response_body: Final = {"input_tokens": 17} if status_code == 200 else {"error": {"message": "invalid key"}}
    router: Final = respx.mock

    with router:
        count_route: Final = router.post(endpoint).mock(return_value=httpx.Response(status_code, json=response_body))
        result: Final = await counter_type().count_tokens(
            model_to_use="claude-test",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "sk-ant-api03-test-key", "api_base": api_base}},
            request_model="claude-test",
        )
        requests: Final = tuple(router.calls)

    assert count_route.called
    assert len(requests) == 1
    assert requests[0].request.url == httpx.URL(endpoint)
    assert isinstance(result, TokenCountResponse)
    assert result.request_model == "claude-test"
    assert result.model_used == "claude-test"
    assert result.tokenizer_type == tokenizer_type

    if expected_error:
        assert result.error is True
        assert result.status_code == status_code
        assert result.total_tokens == 0
        return

    assert result.error is not True
    assert result.total_tokens == 17


@pytest.mark.parametrize(
    ("counter_type", "provider"),
    (
        (AnthropicTokenCounter, "anthropic"),
        (AzureAIAnthropicTokenCounter, "azure_ai"),
    ),
)
def test_count_token_counters_select_their_own_provider(
    counter_type: type[AnthropicTokenCounter] | type[AzureAIAnthropicTokenCounter],
    provider: str,
) -> None:
    counter: Final = counter_type()

    assert counter.should_use_token_counting_api(custom_llm_provider=provider) is True
    assert counter.should_use_token_counting_api(custom_llm_provider="unknown") is False
