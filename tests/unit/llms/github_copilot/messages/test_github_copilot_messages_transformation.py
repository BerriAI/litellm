from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE, GetAPIKeyError
from litellm.llms.github_copilot.messages.transformation import (
    GithubCopilotAnthropicMessagesConfig,
)


@pytest.mark.parametrize(
    ("resolved_base", "environment_base", "expected_base"),
    [
        (None, None, DEFAULT_GITHUB_COPILOT_API_BASE),
        (None, "https://copilot.example.test/", "https://copilot.example.test"),
        (None, "http://localhost:8080/", "http://localhost:8080"),
        ("https://resolved.example.test", "https://env.example.test", "https://resolved.example.test"),
        ("https://resolved.example.test/", "https://env.example.test", "https://resolved.example.test"),
        ("https://resolved.example.test/v1/messages", None, "https://resolved.example.test"),
    ],
)
def test_github_copilot_anthropic_messages_get_complete_url(
    monkeypatch: pytest.MonkeyPatch,
    resolved_base: str | None,
    environment_base: str | None,
    expected_base: str,
) -> None:
    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)
    config: Final = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()

    url: Final = config.get_complete_url(
        api_base=resolved_base,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )

    assert url == f"{expected_base}/v1/messages"
    assert config.authenticator.mock_calls == []


def test_github_copilot_anthropic_messages_validate_environment(monkeypatch: pytest.MonkeyPatch):
    """Test environment validation and header injection."""
    monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"

    headers = {}
    # Pass a hostile api_base to confirm it is ignored.
    validated_headers, api_base = config.validate_anthropic_messages_environment(
        headers=headers,
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://attacker.example.com",
    )

    assert "copilot-integration-id" in validated_headers
    assert validated_headers["copilot-integration-id"] == "vscode-chat"
    assert validated_headers["Authorization"] == "Bearer gh.test-key-123"
    assert "anthropic-version" in validated_headers
    assert validated_headers["anthropic-version"] == "2023-06-01"
    # /v1/messages must use the messages-proxy intent so the Copilot backend
    # enables Anthropic-native features (context_management, thinking, etc.).
    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert validated_headers["x-github-api-version"] == "2026-06-01"
    assert api_base == DEFAULT_GITHUB_COPILOT_API_BASE


def test_github_copilot_anthropic_messages_validate_environment_injects_beta_headers():
    """Anthropic-beta headers must be auto-injected for advanced features
    (context_management, output_format, etc.) — matches the parent
    AnthropicMessagesConfig contract."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"output_format": {"type": "json_object"}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "anthropic-beta" in validated_headers
    assert "structured-outputs-2025-11-13" in validated_headers["anthropic-beta"]


def test_github_copilot_anthropic_messages_validate_environment_preserves_caller_anthropic_version():
    """Caller-supplied anthropic-version must be forwarded verbatim."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={"anthropic-version": "2024-10-22"},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["anthropic-version"] == "2024-10-22"


def test_github_copilot_anthropic_messages_validate_environment_injects_context_management_beta():
    """context_management in optional_params must trigger the corresponding
    anthropic-beta header so the Copilot backend accepts the field."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert "anthropic-beta" in validated_headers
    assert "context-management-2025-06-27" in validated_headers["anthropic-beta"]


def test_github_copilot_anthropic_messages_validate_environment_auth_error():
    """Test error handling when authentication fails."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator to raise an error
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(status_code=401, message="No valid API key found")

    with pytest.raises(AuthenticationError):
        config.validate_anthropic_messages_environment(
            headers={},
            model="github_copilot/claude-haiku-4.5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )


def test_github_copilot_anthropic_messages_supported_params():
    """Test supported parameters list."""
    config = GithubCopilotAnthropicMessagesConfig()
    params = config.get_supported_anthropic_messages_params("github_copilot/claude-haiku-4.5")

    # Should inherit from AnthropicMessagesConfig
    assert "messages" in params
    assert "model" in params
    assert "max_tokens" in params
    assert "thinking" in params


def test_provider_config_manager_dispatches_claude_to_copilot_messages_config():
    """ProviderConfigManager must return the Copilot Anthropic Messages config
    for Claude models served via github_copilot."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="github_copilot/claude-haiku-4.5",
        provider=LlmProviders.GITHUB_COPILOT,
    )

    assert isinstance(config, GithubCopilotAnthropicMessagesConfig)


def test_provider_config_manager_skips_non_claude_copilot_models():
    """Non-Claude github_copilot models (e.g. gpt-*) must not be routed through
    the Anthropic Messages dispatch."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="github_copilot/gpt-5-mini",
        provider=LlmProviders.GITHUB_COPILOT,
    )

    assert config is None


@pytest.mark.parametrize(
    ("environment_base", "expected_base"),
    [
        (None, DEFAULT_GITHUB_COPILOT_API_BASE),
        ("https://copilot.example.test/", "https://copilot.example.test"),
        ("http://localhost:8080/", "http://localhost:8080"),
        ("https://user:password@copilot.example.test", DEFAULT_GITHUB_COPILOT_API_BASE),
    ],
)
def test_validate_environment_ignores_deployment_base_and_resolves_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch, environment_base: str | None, expected_base: str
) -> None:
    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)
    config: Final = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://attacker.example.com",
    )

    assert api_base == expected_base
    assert headers["Authorization"] == "Bearer gh.test-key"
    assert (
        config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="github_copilot/claude-haiku-4.5",
            optional_params={},
            litellm_params={},
        )
        == f"{expected_base}/v1/messages"
    )


def test_github_copilot_config_disables_anthropic_beta_filtering():
    """Copilot's /v1/messages is a native Anthropic passthrough, so injected
    anthropic-beta values (context_management, structured outputs, ...) must be
    forwarded verbatim. The default provider-scoped filter would drop them
    because github_copilot has no entry in the beta headers config; a regression
    here would silently disable header-gated Anthropic features for Copilot."""
    from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
    from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig

    config = GithubCopilotAnthropicMessagesConfig()
    assert config.should_filter_anthropic_beta_headers() is False
    assert AzureAnthropicMessagesConfig().should_filter_anthropic_beta_headers() is True

    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"

    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "context-management-2025-06-27" in headers["anthropic-beta"]

    # The override is load-bearing: had the config opted into the provider-scoped
    # filter, the handler would have run it and dropped every value, since
    # github_copilot has no mapping. Prove that here so a regression that flips
    # should_filter back on is caught as the silent feature breakage it causes.
    stripped = update_headers_with_filtered_beta(headers=dict(headers), provider="github_copilot")
    assert "anthropic-beta" not in stripped


def test_github_copilot_config_does_not_handle_web_search_natively():
    """Copilot's /v1/messages does not run web_search, so its config must report
    handles_web_search_natively() == False. This is what keeps the web-search
    interception handler short-circuiting Copilot instead of routing to it, even
    though Copilot now has a BaseAnthropicMessagesConfig. The base Anthropic
    config (bedrock/vertex/anthropic path) must report True."""
    from litellm.llms.anthropic.pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    assert GithubCopilotAnthropicMessagesConfig().handles_web_search_natively() is False
    assert AnthropicMessagesConfig().handles_web_search_natively() is True


def test_github_copilot_messages_config_probes_capabilities_under_copilot_namespace():
    """Capability probes in the shared pass-through helpers read
    ``self.custom_llm_provider``; without this override they probed the
    ``anthropic`` namespace and ignored the exact ``github_copilot/claude-*``
    cost-map entries."""
    assert GithubCopilotAnthropicMessagesConfig().custom_llm_provider == "github_copilot"


@pytest.mark.parametrize(
    ("environment_base", "expected_base"),
    [(None, DEFAULT_GITHUB_COPILOT_API_BASE), ("http://localhost:8080/", "http://localhost:8080")],
)
def test_validate_environment_uses_per_user_session_and_skips_authenticator(
    monkeypatch: pytest.MonkeyPatch, environment_base: str | None, expected_base: str
) -> None:
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    if environment_base is None:
        monkeypatch.delenv("GITHUB_COPILOT_API_BASE", raising=False)
    else:
        monkeypatch.setenv("GITHUB_COPILOT_API_BASE", environment_base)

    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = AssertionError("shared authenticator must not run")

    session = GithubCopilotUserSession(token="user-copilot-token")
    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-sonnet-5.5",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
        api_base="https://attacker.example.test",
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    assert api_base == expected_base
    assert config.authenticator.mock_calls == []
    assert (
        config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="github_copilot/claude-sonnet-5.5",
            optional_params={},
            litellm_params={"github_copilot_user_session": session},
        )
        == f"{expected_base}/v1/messages"
    )


@pytest.mark.parametrize("authorization_header", ["Authorization", "authorization", "AUTHORIZATION"])
def test_per_user_session_token_wins_over_caller_authorization(authorization_header: str):
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    session = GithubCopilotUserSession(token="user-copilot-token")
    headers, _ = config.validate_anthropic_messages_environment(
        headers={authorization_header: "Bearer caller-token", "x-custom": "preserved"},
        model="github_copilot/claude-sonnet-4.5",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    assert {key: value for key, value in headers.items() if key.lower() == "authorization"} == {
        "Authorization": "Bearer user-copilot-token"
    }
    assert headers["x-custom"] == "preserved"
    assert config.authenticator.mock_calls == []


def test_transform_response_carries_upstream_usage():
    config = GithubCopilotAnthropicMessagesConfig()
    raw = httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "github_copilot/claude-sonnet-5.5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 14, "output_tokens": 3},
        },
    )
    result = config.transform_anthropic_messages_response(
        model="github_copilot/claude-sonnet-5.5",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert result["usage"]["input_tokens"] == 14
    assert result["usage"]["output_tokens"] == 3
