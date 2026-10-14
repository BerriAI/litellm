import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

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


_BREAKPOINT: Final = {"mode": "explicit", "ttl": "30m"}
_EPHEMERAL: Final = {"type": "ephemeral"}
_REQUEST: Final = TypeAdapter(dict[str, object])


def _messages_with_breakpoints() -> Sequence[Mapping[str, object]]:
    return [
        {
            "role": "user",
            "prompt_cache_breakpoint": _BREAKPOINT,
            "content": [
                {"type": "text", "text": "cached prefix", "prompt_cache_breakpoint": _BREAKPOINT},
                {"type": "text", "text": "native mark", "cache_control": _EPHEMERAL},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"prompt_cache_breakpoint": "a tool argument, not a mark"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "result", "prompt_cache_breakpoint": _BREAKPOINT}],
                }
            ],
        },
    ]


def _optional_params_with_breakpoints() -> Mapping[str, object]:
    return {
        "max_tokens": 64,
        "system": [{"type": "text", "text": "system prompt", "prompt_cache_breakpoint": _BREAKPOINT}],
        "tools": [
            {
                "name": "lookup",
                "input_schema": {"type": "object", "properties": {"prompt_cache_breakpoint": {"type": "string"}}},
                "prompt_cache_breakpoint": _BREAKPOINT,
            }
        ],
    }


def _transform(
    messages: Sequence[Mapping[str, object]],
    optional_params: Mapping[str, object],
    litellm_params: GenericLiteLLMParams,
) -> Mapping[str, object]:
    return _REQUEST.validate_python(
        AnthropicMessagesConfig().transform_anthropic_messages_request(
            model="claude-haiku-5-5",
            messages=[dict(message) for message in messages],
            anthropic_messages_optional_request_params=dict(optional_params),
            litellm_params=litellm_params,
            headers={},
        )
    )


def test_prompt_cache_breakpoint_is_forwarded_verbatim_without_drop_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", None)

    result: Final = _transform(
        _messages_with_breakpoints(), _optional_params_with_breakpoints(), GenericLiteLLMParams()
    )

    assert result["messages"] == _messages_with_breakpoints()
    assert result["system"] == _optional_params_with_breakpoints()["system"]
    assert result["tools"] == _optional_params_with_breakpoints()["tools"]


@pytest.mark.parametrize(
    ("global_drop_params", "litellm_params"),
    [
        (True, GenericLiteLLMParams()),
        (None, GenericLiteLLMParams(drop_params=True)),
        (None, GenericLiteLLMParams(drop_params="true")),
    ],
    ids=["litellm.drop_params", "request_or_deployment_drop_params", "drop_params_string_flag"],
)
def test_drop_params_strips_prompt_cache_breakpoint_from_every_block(
    monkeypatch: pytest.MonkeyPatch, global_drop_params: bool | None, litellm_params: GenericLiteLLMParams
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)

    result: Final = _transform(_messages_with_breakpoints(), _optional_params_with_breakpoints(), litellm_params)

    assert result["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "cached prefix"},
                {"type": "text", "text": "native mark", "cache_control": _EPHEMERAL},
            ],
        },
        _messages_with_breakpoints()[1],
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "result"}]}
            ],
        },
    ]
    assert result["system"] == [{"type": "text", "text": "system prompt"}]
    assert result["tools"] == [
        {
            "name": "lookup",
            "input_schema": {"type": "object", "properties": {"prompt_cache_breakpoint": {"type": "string"}}},
        }
    ]


def test_drop_params_leaves_the_callers_blocks_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)
    messages: Final = _messages_with_breakpoints()
    optional_params: Final = _optional_params_with_breakpoints()
    system: Final = optional_params["system"]
    tools: Final = optional_params["tools"]

    _transform(messages, optional_params, GenericLiteLLMParams())

    assert messages == _messages_with_breakpoints()
    assert system == _optional_params_with_breakpoints()["system"]
    assert tools == _optional_params_with_breakpoints()["tools"]


def _cache_control_only_request() -> tuple[Sequence[Mapping[str, object]], Mapping[str, object]]:
    return (
        [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": _EPHEMERAL}]}],
        {
            "max_tokens": 64,
            "system": [{"type": "text", "text": "sys", "cache_control": _EPHEMERAL}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}, "cache_control": _EPHEMERAL}],
        },
    )


def test_drop_params_keeps_a_cache_control_only_request_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", None)

    without_drop: Final = _transform(*_cache_control_only_request(), GenericLiteLLMParams())
    with_drop: Final = _transform(*_cache_control_only_request(), GenericLiteLLMParams(drop_params=True))

    assert json.dumps(with_drop) == json.dumps(without_drop)
