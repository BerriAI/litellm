"""
Unit tests for DashScope's Anthropic-compatible Messages endpoint.
"""

import pytest

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.dashscope.messages.transformation import (
    DashScopeAnthropicMessagesConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

WORKSPACE = "https://llm-abc123.cn-beijing.maas.aliyuncs.com"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            None,
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
            "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages",
        ),
        (
            WORKSPACE,
            f"{WORKSPACE}/apps/anthropic/v1/messages",
        ),
        (
            f"{WORKSPACE}/compatible-mode/v1",
            f"{WORKSPACE}/apps/anthropic/v1/messages",
        ),
        (
            "https://gateway.example.com/anthropic",
            "https://gateway.example.com/anthropic/v1/messages",
        ),
    ],
)
def test_get_complete_url(api_base: str | None, expected: str):
    config = DashScopeAnthropicMessagesConfig()

    url = config.get_complete_url(
        api_base=api_base,
        api_key="sk-test",
        model="qwen3.7-plus",
        optional_params={},
        litellm_params={},
    )

    assert url == expected


def test_missing_dashscope_key_does_not_fall_back_to_anthropic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setattr(litellm, "api_key", None)
    headers: dict[str, str] = {}

    with pytest.raises(AuthenticationError) as error:
        DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers=headers,
            model="qwen3.7-plus",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=WORKSPACE,
        )

    assert "DashScope" in str(error.value)
    assert "sk-ant-secret" not in str(error.value)
    assert {name.lower() for name in headers}.isdisjoint({"x-api-key", "authorization"})


def test_caller_auth_header_is_honored_without_a_dashscope_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    headers = {"x-api-key": "caller-key"}

    resolved_headers, _ = DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers=headers,
        model="qwen3.7-plus",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=WORKSPACE,
    )

    assert resolved_headers["x-api-key"] == "caller-key"


def test_dashscope_key_wins_over_an_anthropic_environment_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    resolved_headers, _ = DashScopeAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="qwen3.7-plus",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=WORKSPACE,
    )

    header_values = " ".join(resolved_headers.values())
    assert "sk-dashscope" in header_values
    assert "sk-ant-secret" not in header_values


def test_dashscope_resolves_to_the_native_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="qwen3.7-plus",
        provider=LlmProviders.DASHSCOPE,
    )

    assert isinstance(config, DashScopeAnthropicMessagesConfig)
    assert config.custom_llm_provider == "dashscope"


def test_dashscope_prefix_reaches_the_provider():
    model, provider, _, _ = litellm.get_llm_provider(
        "dashscope/qwen3.7-plus",
        api_base=WORKSPACE,
    )

    assert provider == "dashscope"
    assert model == "qwen3.7-plus"
