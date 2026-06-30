import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.github_copilot.messages.transformation import (
    GithubCopilotMessagesConfig,
)
from litellm.utils import ProviderConfigManager


def test_github_copilot_provider_uses_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude-opus-4.8",
        provider=litellm.LlmProviders.GITHUB_COPILOT,
    )

    assert isinstance(config, GithubCopilotMessagesConfig)
    assert config.custom_llm_provider == "github_copilot"


def test_anthropic_provider_keeps_default_config_for_claude_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude-opus-4.8",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, GithubCopilotMessagesConfig)


def test_github_copilot_messages_url_targets_native_endpoint():
    config = GithubCopilotMessagesConfig()

    for api_base in (
        None,
        "https://api.githubcopilot.com",
        "https://api.githubcopilot.com/",
        "https://api.githubcopilot.com/v1/messages",
    ):
        assert (
            config.get_complete_url(
                api_base=api_base,
                api_key=None,
                model="claude-opus-4.8",
                optional_params={},
                litellm_params={},
            )
            == "https://api.githubcopilot.com/v1/messages"
        )


def test_github_copilot_messages_headers_use_bearer_not_x_api_key():
    config = GithubCopilotMessagesConfig()

    # Pass api_key so the shared Authenticator is not invoked.
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="claude-opus-4.8",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="copilot-token",
    )

    # Copilot authenticates with a bearer token + editor identity headers.
    assert headers["Authorization"] == "Bearer copilot-token"
    assert headers["editor-version"]
    assert headers["copilot-integration-id"] == "vscode-chat"
    assert headers["anthropic-version"] == "2023-06-01"
    # The bearer header is authoritative; no redundant x-api-key.
    assert "x-api-key" not in headers


def test_github_copilot_messages_injects_context_management_beta_header():
    config = GithubCopilotMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="claude-opus-4.8",
        messages=[],
        optional_params={
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}
        },
        litellm_params={},
        api_key="copilot-token",
    )

    # context_management requires this beta header or the Copilot backend 400s.
    assert "context-management-2025-06-27" in headers.get("anthropic-beta", "")


def test_github_copilot_messages_preserves_caller_authorization():
    config = GithubCopilotMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"authorization": "Bearer caller-supplied"},
        model="claude-opus-4.8",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="copilot-token",
    )

    assert headers["authorization"] == "Bearer caller-supplied"
    assert "Authorization" not in headers
