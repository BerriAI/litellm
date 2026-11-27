"""
Test MiniMax Anthropic-compatible API support
"""

from typing import Final

import pytest

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.minimax.messages.transformation import MinimaxMessagesConfig


def test_minimax_anthropic_config():
    """Test that MinimaxMessagesConfig is properly configured"""
    config = MinimaxMessagesConfig()

    # Test custom_llm_provider
    assert config.custom_llm_provider == "minimax"

    # Test get_api_base default
    api_base = config.get_api_base()
    assert api_base == "https://api.minimax.io/anthropic/v1/messages"

    # Test get_api_base with custom value
    custom_base = config.get_api_base(
        api_base="https://api.minimaxi.com/anthropic/v1/messages"
    )
    assert custom_base == "https://api.minimaxi.com/anthropic/v1/messages"


def test_minimax_provider_routing():
    """Test that minimax provider is properly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with minimax/ prefix
    model, provider, api_key, api_base = get_llm_provider(
        model="minimax/MiniMax-M2.1",
        api_base="https://api.minimax.io/anthropic/v1/messages",
    )
    assert provider == "minimax"
    assert model == "MiniMax-M2.1"


def test_minimax_provider_config_manager():
    """Test that ProviderConfigManager returns MinimaxMessagesConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="MiniMax-M2.1", provider=LlmProviders.MINIMAX
    )

    assert config is not None
    assert isinstance(config, MinimaxMessagesConfig)
    assert config.custom_llm_provider == "minimax"


if __name__ == "__main__":
    # Run basic tests that don't require API key
    print("Testing MiniMax Anthropic Config...")
    test_minimax_anthropic_config()
    print("✓ Config test passed")

    print("\nTesting MiniMax Provider Routing...")
    test_minimax_provider_routing()
    print("✓ Routing test passed")

    print("\nTesting MiniMax Provider Config Manager...")
    test_minimax_provider_config_manager()
    print("✓ Provider config manager test passed")

    print("\n✅ All basic tests passed!")


def test_minimax_messages_env_key_attached(monkeypatch):
    """Regression: an env-only MINIMAX_API_KEY must be attached on /v1/messages validation"""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-env-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    config = MinimaxMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="MiniMax-M2.1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
    )
    assert headers["x-api-key"] == "test-minimax-env-key"


def test_minimax_messages_explicit_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    config = MinimaxMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="MiniMax-M2.1",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key="param-key",
    )
    assert headers["x-api-key"] == "param-key"


@pytest.mark.parametrize("minimax_api_key", [None, "   "])
@pytest.mark.parametrize(
    ("anthropic_env", "anthropic_secret"),
    [
        ("ANTHROPIC_API_KEY", "sk-ant-secret"),
        ("ANTHROPIC_AUTH_TOKEN", "oauth-ant-secret"),
    ],
)
def test_missing_minimax_key_does_not_fall_back_to_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    minimax_api_key: str | None,
    anthropic_env: str,
    anthropic_secret: str,
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    if minimax_api_key is not None:
        monkeypatch.setenv("MINIMAX_API_KEY", minimax_api_key)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv(anthropic_env, anthropic_secret)
    monkeypatch.setattr(litellm, "api_key", None)
    headers: Final[dict[str, object]] = {}

    with pytest.raises(AuthenticationError) as error:
        MinimaxMessagesConfig().validate_anthropic_messages_environment(
            headers=headers,
            model="MiniMax-M2.1",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://api.minimax.io/anthropic",
        )

    assert "MiniMax" in str(error.value)
    assert anthropic_secret not in str(error.value)
    assert {name.lower() for name in headers}.isdisjoint({"x-api-key", "authorization"})


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    [
        ("x-api-key", "caller-key"),
        ("Authorization", "Bearer caller-token"),
    ],
)
def test_caller_auth_header_is_honored_without_minimax_key(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
    header_value: str,
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setattr(litellm, "api_key", None)
    headers: Final[dict[str, object]] = {header_name: header_value}

    resolved_headers, _ = MinimaxMessagesConfig().validate_anthropic_messages_environment(
        headers=headers,
        model="MiniMax-M2.1",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://api.minimax.io/anthropic",
    )

    assert resolved_headers[header_name] == header_value
    assert "sk-ant-secret" not in resolved_headers.values()
