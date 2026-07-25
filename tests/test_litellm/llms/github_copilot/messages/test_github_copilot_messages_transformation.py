import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.common_utils import GetAPIKeyError
from litellm.llms.github_copilot.messages.transformation import (
    GithubCopilotAnthropicMessagesConfig,
)


def test_github_copilot_anthropic_messages_config_init():
    """Test GithubCopilotAnthropicMessagesConfig initialization."""
    config = GithubCopilotAnthropicMessagesConfig()
    assert config is not None
    assert hasattr(config, "authenticator")


def test_github_copilot_anthropic_messages_get_complete_url():
    """get_complete_url builds the /v1/messages URL from the base it is handed.

    In the request flow that ``api_base`` is the value already resolved by
    validate_anthropic_messages_environment (the authenticated Copilot host); the
    caller-supplied base is discarded there, not here (see the validate tests).
    """
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_base.return_value = None

    # No api_base supplied and no authenticator base -> default Copilot endpoint.
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/v1/messages"
    # Falls back to a single authenticator read, not a hard-coded second one.
    config.authenticator.get_api_base.assert_called()

    # The resolved (validated) base passed in is reused verbatim; no extra read.
    config.authenticator.get_api_base.reset_mock()
    url = config.get_complete_url(
        api_base="https://api.business.githubcopilot.com",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.business.githubcopilot.com/v1/messages"
    config.authenticator.get_api_base.assert_not_called()

    # A trailing slash on the base must not produce a double-slash URL.
    url = config.get_complete_url(
        api_base="https://api.business.githubcopilot.com/",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.business.githubcopilot.com/v1/messages"

    # An already-complete /v1/messages base is left untouched.
    url = config.get_complete_url(
        api_base="https://api.githubcopilot.com/v1/messages",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/v1/messages"


def test_github_copilot_anthropic_messages_get_complete_url_normalizes_authenticator_trailing_slash():
    """A tenant base with a trailing slash from the authenticator fallback must
    not yield a double-slash URL."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_base.return_value = "https://api.business.githubcopilot.com/"

    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.business.githubcopilot.com/v1/messages"


def test_github_copilot_anthropic_messages_validate_environment():
    """Test environment validation and header injection."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

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
    assert "Authorization" in validated_headers
    assert "anthropic-version" in validated_headers
    assert validated_headers["anthropic-version"] == "2023-06-01"
    # /v1/messages must use the messages-proxy intent so the Copilot backend
    # enables Anthropic-native features (context_management, thinking, etc.).
    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert validated_headers["x-github-api-version"] == "2026-06-01"
    assert api_base == "https://api.githubcopilot.com"


def test_github_copilot_anthropic_messages_validate_environment_injects_beta_headers():
    """Anthropic-beta headers must be auto-injected for advanced features
    (context_management, output_format, etc.) — matches the parent
    AnthropicMessagesConfig contract."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = None

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
    config.authenticator.get_api_base.return_value = None

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
    config.authenticator.get_api_base.return_value = None

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


def test_github_copilot_anthropic_messages_validate_environment_normalizes_trailing_slash():
    """A tenant base with a trailing slash from the authenticator must be
    normalized so the URL built downstream has no double slash."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = "https://api.business.githubcopilot.com/"

    _, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://attacker.example.com",
    )

    assert api_base == "https://api.business.githubcopilot.com"


def test_github_copilot_config_disables_anthropic_beta_filtering():
    """Copilot's /v1/messages is a native Anthropic passthrough, so injected
    anthropic-beta values (context_management, structured outputs, ...) must be
    forwarded verbatim. The default provider-scoped filter would drop them
    because github_copilot has no entry in the beta headers config; a regression
    here would silently disable header-gated Anthropic features for Copilot."""
    from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )

    config = GithubCopilotAnthropicMessagesConfig()
    assert config.should_filter_anthropic_beta_headers() is False
    assert AnthropicMessagesConfig().should_filter_anthropic_beta_headers() is True

    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = None

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
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
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
