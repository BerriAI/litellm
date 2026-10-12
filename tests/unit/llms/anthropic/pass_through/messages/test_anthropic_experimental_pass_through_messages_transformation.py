from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.anthropic.wif import get_anthropic_wif_token
from litellm.llms.minimax.messages.transformation import MinimaxMessagesConfig
from litellm.llms.tencent.messages.transformation import TencentAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams
from tests.unit.llms.anthropic.test_anthropic_wif import (
    ScriptedPoster,
    make_engine,
    token_response,
    write_token_file,
)

_WIF_PARAMS: Final[dict] = {
    "anthropic_federation_rule_id": "fdrl_abc123",
    "anthropic_organization_id": "org-uuid-1",
    "anthropic_identity_token_file": "/var/run/secrets/identity-token",
}


def test_workload_identity_allowed_for_anthropic() -> None:
    assert AnthropicMessagesConfig()._allows_workload_identity is True


def test_workload_identity_blocked_for_minimax() -> None:
    assert MinimaxMessagesConfig()._allows_workload_identity is False


def test_workload_identity_blocked_for_tencent() -> None:
    assert TencentAnthropicMessagesConfig()._allows_workload_identity is False


def test_minimax_validate_environment_never_attaches_anthropic_wif_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: before the fix, an Anthropic-WIF-configured proxy would mint a real
    Anthropic federation token inside MiniMax's inherited validate_anthropic_messages_environment
    and send it as the Authorization header on the MiniMax-routed request. With no MiniMax
    credential of its own the deployment must fail closed on the missing key instead."""
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_prod")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-prod-uuid")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    token_file = write_token_file(tmp_path, "jwt-assertion-value")
    litellm_params = {"anthropic_identity_token_file": str(token_file)}

    with pytest.raises(litellm.AuthenticationError, match="Missing Anthropic API Key"):
        MinimaxMessagesConfig().validate_anthropic_messages_environment(
            headers={},
            model="MiniMax-M2.1",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_key=None,
            api_base="https://api.minimax.io/anthropic",
        )


def test_tencent_validate_environment_never_attaches_anthropic_wif_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_prod")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-prod-uuid")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TENCENT_API_KEY", raising=False)
    token_file = write_token_file(tmp_path, "jwt-assertion-value")
    litellm_params = {"anthropic_identity_token_file": str(token_file)}

    monkeypatch.setattr(litellm, "api_key", None)
    with pytest.raises(litellm.AuthenticationError, match="Missing Anthropic API Key"):
        TencentAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers={},
            model="deepseek-v4-pro",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_key=None,
            api_base="https://tokenhub-intl.tencentcloudmaas.com",
        )


def test_wif_token_exchange_reaches_only_anthropic_not_minimax_or_tencent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_anthropic_wif_token's engine parameter is the only DI seam in the WIF minting chain;
    validate_anthropic_messages_environment always uses the module's default engine, so this
    drives that seam directly with the exact litellm_params AnthropicModelInfo.get_auth_header
    would receive from each config, proving MiniMax/Tencent never reach the token endpoint even
    when a mint would otherwise succeed."""
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_prod")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org-prod-uuid")
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token_file = write_token_file(tmp_path, "jwt-assertion-value")
    litellm_params = {"anthropic_identity_token_file": str(token_file)}
    poster = ScriptedPoster([token_response("sk-ant-oat01-canary")])
    engine = make_engine(poster)

    minted: Final = get_anthropic_wif_token(
        litellm_params,
        "https://api.anthropic.com",
        "claude-sonnet-4-5",
        engine,
    )
    assert minted == "sk-ant-oat01-canary"
    assert len(poster.requests) == 1

    for config in (MinimaxMessagesConfig(), TencentAnthropicMessagesConfig()):
        assert config._allows_workload_identity is False

    assert len(poster.requests) == 1


DANGEROUS_TOOL_USE_BETA: Final = "dangerous-tool-use-2026-09-03"
AUTO_MODE_SAFEGUARDS: Final = [
    {"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}
]


def _classifier_configs() -> tuple[AnthropicMessagesConfig, ...]:
    from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig
    from litellm.llms.databricks.messages.transformation import DatabricksAnthropicMessagesConfig

    return (AnthropicMessagesConfig(), AzureAnthropicMessagesConfig(), DatabricksAnthropicMessagesConfig())


def _no_classifier_configs() -> tuple[AnthropicMessagesConfig, ...]:
    from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig
    from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig

    return (DeepSeekAnthropicMessagesConfig(), OpenAILikeAnthropicMessagesConfig(), MinimaxMessagesConfig())


def _safeguards_request(config: AnthropicMessagesConfig) -> dict:
    return config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Use the Bash tool to run: echo hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 64, "safeguards": list(AUTO_MODE_SAFEGUARDS)},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def _beta_header(config: AnthropicMessagesConfig, client_beta: str) -> dict:
    return config._update_headers_with_anthropic_beta(
        headers={"anthropic-beta": client_beta}, optional_params={}, messages=[]
    )


@pytest.mark.parametrize("config", _classifier_configs(), ids=type)
def test_routes_whose_platform_runs_the_classifier_keep_safeguards_and_the_beta(
    config: AnthropicMessagesConfig, local_beta_headers_config: None
) -> None:
    """Claude Code's auto mode sends `safeguards` with dangerous-tool-use-2026-09-03 and reads
    `safeguard_results` back. Anthropic, Azure AI Foundry, and Databricks serving endpoints
    honour the pair (probed live 2026-10-12), so both halves must reach them unchanged."""
    assert _safeguards_request(config)["safeguards"] == AUTO_MODE_SAFEGUARDS
    assert _beta_header(config, DANGEROUS_TOOL_USE_BETA)["anthropic-beta"] == DANGEROUS_TOOL_USE_BETA


@pytest.mark.parametrize("config", _no_classifier_configs(), ids=type)
def test_routes_without_the_classifier_drop_safeguards_together_with_the_beta(
    config: AnthropicMessagesConfig, local_beta_headers_config: None
) -> None:
    """A platform without the classifier answers 400 "safeguards: Extra inputs are not permitted"
    to the body field whatever the header says, so the pair is dropped together and the turn
    stays a 200 without a verdict, which Claude Code's gateway contract treats as "feature off"."""
    assert "safeguards" not in _safeguards_request(config)
    assert "anthropic-beta" not in _beta_header(config, DANGEROUS_TOOL_USE_BETA)
    assert _beta_header(config, f"context-1m-2025-08-07,{DANGEROUS_TOOL_USE_BETA}") == {
        "anthropic-beta": "context-1m-2025-08-07"
    }


@pytest.mark.asyncio
async def test_no_classifier_route_sends_neither_safeguards_nor_the_beta_on_the_wire(
    local_beta_headers_config: None,
) -> None:
    """JSON-configured OpenAI-compatible providers forward every beta header untouched, so the
    pair has to be dropped by the request transform itself, not by the beta header filter."""
    import json

    import httpx

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake platform records what litellm sent

    def platform_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            json={
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
            request=request,
        )

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(platform_answers))
    try:
        response = await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=[{"role": "user", "content": "Use the Bash tool to run: echo hello"}],
            model="sail/claude-sonnet-4-6",
            api_key="test-api-key",
            api_base="https://sail.example.test/v1",
            safeguards=list(AUTO_MODE_SAFEGUARDS),
            extra_headers={"anthropic-beta": f"context-1m-2025-08-07,{DANGEROUS_TOOL_USE_BETA}"},
            client=upstream,
        )
    finally:
        await upstream.close()

    assert len(received) == 1
    sent: Final = received[0]
    assert "safeguards" not in json.loads(sent.content)
    assert sent.headers["anthropic-beta"] == "context-1m-2025-08-07"
    assert "safeguard_results" not in response
    assert response["content"][0]["text"] == "hello"
