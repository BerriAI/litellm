from collections.abc import Mapping
from pathlib import Path
from typing import Final
from unittest.mock import patch

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


class _BetaMapRejectsTheClassifierConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock_converse"


def _no_classifier_configs() -> tuple[AnthropicMessagesConfig, ...]:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig

    sail: Final = JSONProviderRegistry.get("sail")
    assert sail is not None
    return (JSONProviderAnthropicMessagesConfig(sail), _BetaMapRejectsTheClassifierConfig())


def _configs_that_forward_safeguards_as_sent() -> tuple[AnthropicMessagesConfig, ...]:
    from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig
    from litellm.llms.github_copilot.messages.transformation import GithubCopilotAnthropicMessagesConfig
    from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig

    return (
        DeepSeekAnthropicMessagesConfig(),
        GithubCopilotAnthropicMessagesConfig(),
        MinimaxMessagesConfig(),
        OpenAILikeAnthropicMessagesConfig(),
    )


def _safeguards_request(config: AnthropicMessagesConfig, litellm_params: GenericLiteLLMParams | None = None) -> dict:
    return config.transform_anthropic_messages_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Use the Bash tool to run: echo hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 64, "safeguards": list(AUTO_MODE_SAFEGUARDS)},
        litellm_params=litellm_params or GenericLiteLLMParams(),
        headers={},
    )


def _beta_header(config: AnthropicMessagesConfig, client_beta: str) -> dict:
    return config._update_headers_with_anthropic_beta(
        headers={"anthropic-beta": client_beta}, optional_params={}, messages=[]
    )


def _derived_beta_header(config: AnthropicMessagesConfig) -> dict:
    return config._update_headers_with_anthropic_beta(
        headers={}, optional_params={"safeguards": list(AUTO_MODE_SAFEGUARDS)}, messages=[]
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
    assert _derived_beta_header(config)["anthropic-beta"] == DANGEROUS_TOOL_USE_BETA


@pytest.mark.parametrize("config", _configs_that_forward_safeguards_as_sent(), ids=type)
def test_unjudged_routes_forward_safeguards_and_the_beta_as_sent(
    config: AnthropicMessagesConfig, local_beta_headers_config: None
) -> None:
    """A platform the beta header map has no column for, or a deployment the operator opted into
    /v1/messages passthrough, was never judged, so the request reaches it the way the caller
    sent it, as it did before the pair could be dropped."""
    assert _safeguards_request(config)["safeguards"] == AUTO_MODE_SAFEGUARDS
    assert _beta_header(config, DANGEROUS_TOOL_USE_BETA)["anthropic-beta"] == DANGEROUS_TOOL_USE_BETA
    assert _derived_beta_header(config)["anthropic-beta"] == DANGEROUS_TOOL_USE_BETA


_CLASSIFIER_SETTINGS: Final = {"safeguards_classifier_model": "classifier"}


@pytest.mark.parametrize("config", _no_classifier_configs(), ids=type)
@pytest.mark.parametrize(
    ("litellm_params", "global_drop_params", "general_settings", "forwarded"),
    (
        pytest.param(GenericLiteLLMParams(), False, {}, AUTO_MODE_SAFEGUARDS, id="as_sent"),
        pytest.param(GenericLiteLLMParams(drop_params=True), False, {}, None, id="deployment_drop_params"),
        pytest.param(GenericLiteLLMParams(), True, {}, None, id="global_drop_params"),
        pytest.param(GenericLiteLLMParams(), False, _CLASSIFIER_SETTINGS, None, id="proxy_classifier"),
    ),
)
def test_a_route_without_the_classifier_drops_safeguards_only_under_drop_params_or_the_proxy_classifier(
    config: AnthropicMessagesConfig,
    litellm_params: GenericLiteLLMParams,
    global_drop_params: bool,
    general_settings: Mapping[str, str],
    forwarded: list[dict] | None,
    monkeypatch: pytest.MonkeyPatch,
    local_beta_headers_config: None,
) -> None:
    """A platform without the classifier answers 400 to `safeguards`, which is how litellm treats any
    unsupported param: dropped under drop_params, forwarded as sent without it. The proxy's own
    classifier answers in the platform's place, so the field is dropped then too."""
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    with patch("litellm.proxy.proxy_server.general_settings", general_settings):
        request: Final = _safeguards_request(config, litellm_params)
    assert request.get("safeguards") == forwarded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("drop_params", "forwarded", "forwarded_beta"),
    (
        pytest.param(False, AUTO_MODE_SAFEGUARDS, f"context-1m-2025-08-07,{DANGEROUS_TOOL_USE_BETA}", id="as_sent"),
        pytest.param(True, None, "context-1m-2025-08-07", id="drop_params"),
    ),
)
async def test_a_json_provider_gets_safeguards_and_the_beta_together_unless_drop_params_drops_both(
    drop_params: bool, forwarded: list[dict] | None, forwarded_beta: str, local_beta_headers_config: None
) -> None:
    """JSON-configured OpenAI-compatible providers forward every beta header untouched, so under
    drop_params the request transform itself drops the beta along with the field."""
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
            drop_params=drop_params,
            client=upstream,
        )
    finally:
        await upstream.close()

    assert len(received) == 1
    sent: Final = received[0]
    assert json.loads(sent.content).get("safeguards") == forwarded
    assert sent.headers["anthropic-beta"] == forwarded_beta
    assert "safeguard_results" not in response
    assert response["content"][0]["text"] == "hello"


@pytest.mark.parametrize(
    ("drop_params", "expected_tool_result_content"),
    [
        (True, [{"type": "text", "text": "sunny"}]),
        (None, [{"type": "text", "text": 5}, {"type": "text", "text": "sunny"}]),
    ],
    ids=["drop_params", "no_drop_params"],
)
def test_drop_params_drops_a_non_string_tool_result_text_on_the_native_request(
    drop_params, expected_tool_result_content
):
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [{"type": "text", "text": 5}, {"type": "text", "text": "sunny"}],
                }
            ],
        },
    ]

    request = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 64},
        litellm_params=GenericLiteLLMParams(drop_params=drop_params),
        headers={},
    )

    assert request["messages"][2]["content"][0]["content"] == expected_tool_result_content


def test_drop_params_forwards_a_non_object_message_item_as_sent_on_the_native_request():
    tool_result_turn = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": [{"type": "text", "text": None}, {"type": "text", "text": "sunny"}],
            }
        ],
    }

    request = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "What is the weather?"}, 7, tool_result_turn],
        anthropic_messages_optional_request_params={"max_tokens": 64},
        litellm_params=GenericLiteLLMParams(drop_params=True),
        headers={},
    )

    assert request["messages"][1] == 7
    assert request["messages"][2]["content"][0]["content"] == [{"type": "text", "text": "sunny"}]
