from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.anthropic.wif import get_anthropic_wif_token
from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig
from litellm.llms.bedrock.claude_platform.messages_transformation import BedrockClaudePlatformMessagesConfig
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.llms.bedrock.messages.mantle_transformation import AmazonMantleMessagesConfig
from litellm.llms.minimax.messages.transformation import MinimaxMessagesConfig
from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig
from litellm.llms.tencent.messages.transformation import TencentAnthropicMessagesConfig
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)
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


_SAMPLING_PARAMS: Final[dict[str, object]] = {"temperature": 1, "top_p": 0.9, "top_k": 40}
_CLAUDE_ROUTES_ON_ANTHROPIC_API_VERSIONS: Final = (
    pytest.param(AnthropicMessagesConfig, "claude-sonnet-4-6", id="anthropic"),
    pytest.param(AzureAnthropicMessagesConfig, "claude-sonnet-4-6", id="azure_ai"),
    pytest.param(AmazonMantleMessagesConfig, "anthropic.claude-sonnet-4-6", id="bedrock_mantle"),
    pytest.param(BedrockClaudePlatformMessagesConfig, "claude-sonnet-4-6", id="bedrock_claude_platform"),
    pytest.param(VertexAIPartnerModelsAnthropicMessagesConfig, "claude-sonnet-4-6", id="vertex_ai"),
)


@pytest.fixture(autouse=True)
def _anthropic_base_url_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


def _request_with_sampling_params(
    config: AnthropicMessagesConfig, model: str, version: str, drop_params: bool, api_base: str | None = None
) -> dict[str, object]:
    return config.transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        anthropic_messages_optional_request_params={"max_tokens": 16, **_SAMPLING_PARAMS},
        litellm_params=GenericLiteLLMParams(drop_params=drop_params, api_base=api_base),
        headers={"Anthropic-Version": version},
    )


def _sampling_params_in(request: dict[str, object]) -> dict[str, object]:
    return {name: request.get(name) for name in _SAMPLING_PARAMS}


@pytest.mark.parametrize(("config_class", "model"), _CLAUDE_ROUTES_ON_ANTHROPIC_API_VERSIONS)
def test_a_claude_route_drops_sampling_params_under_drop_params_from_anthropic_version_2026_10_21(
    config_class: type[AnthropicMessagesConfig], model: str
) -> None:
    request: Final = _request_with_sampling_params(config_class(), model, "2026-10-21", drop_params=True)

    assert request.keys().isdisjoint(_SAMPLING_PARAMS)
    assert request["max_tokens"] == 16


@pytest.mark.parametrize(("config_class", "model"), _CLAUDE_ROUTES_ON_ANTHROPIC_API_VERSIONS)
def test_a_claude_route_answers_400_naming_the_sampling_params_under_anthropic_version_2026_10_21(
    config_class: type[AnthropicMessagesConfig], model: str
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError) as raised:
        _request_with_sampling_params(config_class(), model, "2026-10-21", drop_params=False)

    assert raised.value.status_code == 400
    assert raised.value.message.startswith(
        f"litellm.UnsupportedParamsError: {model} does not support temperature=1. anthropic-version 2026-10-21 "
        "removed temperature, top_p and top_k."
    )


def test_a_later_dated_anthropic_version_also_drops_sampling_params() -> None:
    request: Final = _request_with_sampling_params(
        AnthropicMessagesConfig(), "claude-sonnet-4-6", "2027-01-15", drop_params=True
    )

    assert request.keys().isdisjoint(_SAMPLING_PARAMS)


@pytest.mark.parametrize(
    ("config_class", "model", "version"),
    [
        pytest.param(AnthropicMessagesConfig, "claude-sonnet-4-6", "2023-06-01", id="anthropic-2023-06-01"),
        pytest.param(AnthropicMessagesConfig, "claude-sonnet-4-6", "2026-10-20", id="anthropic-day-before"),
        pytest.param(AnthropicMessagesConfig, "claude-sonnet-4-6", "latest", id="anthropic-undated"),
        pytest.param(AzureAnthropicMessagesConfig, "claude-sonnet-4-6", "2023-06-01", id="azure_ai-2023-06-01"),
        pytest.param(
            VertexAIPartnerModelsAnthropicMessagesConfig, "claude-sonnet-4-6", "2023-06-01", id="vertex_ai-2023-06-01"
        ),
        pytest.param(
            AmazonAnthropicClaudeMessagesConfig,
            "anthropic.claude-sonnet-4-6",
            "2026-10-21",
            id="bedrock_invoke-sends-its-own-version",
        ),
        pytest.param(MinimaxMessagesConfig, "MiniMax-M2", "2026-10-21", id="non-anthropic-platform"),
    ],
)
def test_sampling_params_reach_an_upstream_whose_api_version_still_takes_them(
    config_class: type[AnthropicMessagesConfig], model: str, version: str
) -> None:
    request: Final = _request_with_sampling_params(config_class(), model, version, drop_params=False)

    assert _sampling_params_in(request) == _SAMPLING_PARAMS


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize(
    ("config_class", "model", "api_base"),
    [
        pytest.param(
            AnthropicMessagesConfig, "deepseek-v4-flash", "https://api.deepseek.com/anthropic", id="anthropic-deepseek"
        ),
        pytest.param(AnthropicMessagesConfig, "MiniMax-M3", "https://api.minimax.io/anthropic", id="anthropic-minimax"),
        pytest.param(
            AnthropicMessagesConfig,
            "claude-sonnet-4-6",
            "https://notservices.ai.azure.com.example/anthropic",
            id="anthropic-foundry-lookalike",
        ),
        pytest.param(
            OpenAILikeAnthropicMessagesConfig,
            "qwen3-coder",
            "http://vllm.internal:8000/v1",
            id="openai_like-passthrough",
        ),
    ],
)
def test_a_third_party_api_base_keeps_sampling_params_under_anthropic_version_2026_10_21(
    config_class: type[AnthropicMessagesConfig], model: str, api_base: str, drop_params: bool
) -> None:
    request: Final = _request_with_sampling_params(
        config_class(), model, "2026-10-21", drop_params=drop_params, api_base=api_base
    )

    assert _sampling_params_in(request) == _SAMPLING_PARAMS


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize("base_url_env", ["ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"])
def test_a_third_party_env_base_url_keeps_sampling_params_under_anthropic_version_2026_10_21(
    monkeypatch: pytest.MonkeyPatch, base_url_env: str, drop_params: bool
) -> None:
    monkeypatch.setenv(base_url_env, "https://api.deepseek.com/anthropic")

    request: Final = _request_with_sampling_params(
        AnthropicMessagesConfig(), "deepseek-v4-flash", "2026-10-21", drop_params=drop_params
    )

    assert _sampling_params_in(request) == _SAMPLING_PARAMS


@pytest.mark.parametrize(
    ("api_base", "base_url_env"),
    [
        pytest.param("https://api.anthropic.com", None, id="api_base"),
        pytest.param("https://api.anthropic.com/v1/messages", None, id="api_base-full-url"),
        pytest.param(None, "https://api.anthropic.com/", id="env-base-url"),
        pytest.param("https://api.anthropic.com", "https://api.deepseek.com/anthropic", id="api_base-over-env"),
        pytest.param("https://team.services.ai.azure.com/anthropic", None, id="foundry"),
        pytest.param("https://team.cognitiveservices.azure.com/anthropic", None, id="foundry-cognitive-services"),
        pytest.param("team.openai.azure.com/anthropic", None, id="foundry-openai-host-without-scheme"),
    ],
)
def test_anthropic_and_foundry_api_bases_gate_sampling_params_under_anthropic_version_2026_10_21(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None, base_url_env: str | None
) -> None:
    if base_url_env is not None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url_env)

    dropped: Final = _request_with_sampling_params(
        AnthropicMessagesConfig(), "claude-sonnet-4-6", "2026-10-21", drop_params=True, api_base=api_base
    )
    assert dropped.keys().isdisjoint(_SAMPLING_PARAMS)
    with pytest.raises(litellm.UnsupportedParamsError):
        _request_with_sampling_params(
            AnthropicMessagesConfig(), "claude-sonnet-4-6", "2026-10-21", drop_params=False, api_base=api_base
        )


_LIST_ID_ADVISOR_TURN: Final = (
    {"type": "server_tool_use", "id": "srvtoolu_01", "name": "advisor", "input": {}},
    {"type": "advisor_tool_result", "tool_use_id": ["srvtoolu_01"], "content": "ok"},
    {"type": "text", "text": "done"},
)


@pytest.mark.parametrize(
    ("drop_params", "expected"),
    [
        pytest.param(None, list(_LIST_ID_ADVISOR_TURN[1:]), id="kept-for-the-provider"),
        pytest.param(True, [{"type": "text", "text": "done"}], id="dropped-under-drop-params"),
    ],
)
def test_an_advisor_block_with_a_list_id_is_dropped_only_under_drop_params(
    drop_params: bool | None, expected: list[dict[str, object]]
) -> None:
    request: Final = AnthropicMessagesConfig().transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType]  # bare-dict signature
        model="claude-opus-4-8",
        messages=[
            {"role": "assistant", "content": list(_LIST_ID_ADVISOR_TURN)},
            {"role": "user", "content": "and now?"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 64},
        litellm_params=GenericLiteLLMParams(drop_params=drop_params),
        headers={},
    )
    assert request["messages"][0]["content"] == expected
