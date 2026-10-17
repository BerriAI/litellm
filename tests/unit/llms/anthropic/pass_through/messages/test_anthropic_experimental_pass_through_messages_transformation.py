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
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)
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


@pytest.mark.parametrize(
    "config_class",
    [AnthropicMessagesConfig, VertexAIPartnerModelsAnthropicMessagesConfig],
)
def test_messages_request_removes_tool_custom_fields_without_mutating_tools(
    config_class: type[AnthropicMessagesConfig],
) -> None:
    tools: Final[list[dict[str, object]]] = [
        {
            "name": "A",
            "input_schema": {"type": "object", "properties": {}},
            "custom": {"eager_input_streaming": True},
        },
        {
            "name": "B",
            "input_schema": {"type": "object", "properties": {}},
            "custom": {"defer_loading": True},
        },
        {"name": "C", "input_schema": {"type": "object", "properties": {}}},
    ]
    optional_params: Final[dict[str, object]] = {"max_tokens": 50, "tools": tools}

    request: Final = config_class().transform_anthropic_messages_request(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "Hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert request["tools"] == [
        {"name": "A", "input_schema": {"type": "object", "properties": {}}},
        {
            "name": "B",
            "input_schema": {"type": "object", "properties": {}},
            "defer_loading": True,
        },
        {"name": "C", "input_schema": {"type": "object", "properties": {}}},
    ]
    assert tools[0]["custom"] == {"eager_input_streaming": True}
    assert optional_params["tools"] is tools


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
