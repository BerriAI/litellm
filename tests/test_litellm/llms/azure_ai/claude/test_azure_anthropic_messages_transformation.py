import copy
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from unittest.mock import patch

import pytest

from litellm.llms.azure_ai.anthropic.messages_transformation import (
    AzureAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


class TestAzureAnthropicMessagesConfig:
    def test_inherits_from_anthropic_messages_config(self):
        """Test that AzureAnthropicMessagesConfig inherits from AnthropicMessagesConfig"""
        config = AzureAnthropicMessagesConfig()
        assert isinstance(config, AzureAnthropicMessagesConfig)
        # Check that it has methods from parent class
        assert hasattr(config, "get_supported_anthropic_messages_params")
        assert hasattr(config, "get_complete_url")
        assert hasattr(config, "validate_anthropic_messages_environment")
        assert hasattr(config, "transform_anthropic_messages_request")
        assert hasattr(config, "transform_anthropic_messages_response")

    def test_validate_anthropic_messages_environment_with_dict_litellm_params(self):
        """Test validate_anthropic_messages_environment with dict litellm_params"""
        config = AzureAnthropicMessagesConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}
        api_key = "test-api-key"

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            result, api_base = config.validate_anthropic_messages_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
            )

            # Verify that dict was converted to GenericLiteLLMParams
            call_args = mock_validate.call_args
            assert isinstance(call_args[1]["litellm_params"], GenericLiteLLMParams)
            assert call_args[1]["litellm_params"].api_key == "test-api-key"
            assert "anthropic-version" in result
            assert "x-api-key" in result
            assert result["x-api-key"] == "test-api-key"
            assert "api-key" not in result

    def test_validate_anthropic_messages_environment_converts_api_key_to_x_api_key(
        self,
    ):
        """Test that api-key header is converted to x-api-key"""
        config = AzureAnthropicMessagesConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            result, api_base = config.validate_anthropic_messages_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )

            # Verify api-key was converted to x-api-key
            assert "x-api-key" in result
            assert result["x-api-key"] == "test-api-key"
            assert "api-key" not in result

    def test_validate_anthropic_messages_environment_sets_headers(self):
        """Test that required headers are set"""
        config = AzureAnthropicMessagesConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            result, api_base = config.validate_anthropic_messages_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )

            assert "anthropic-version" in result
            assert result["anthropic-version"] == "2023-06-01"
            assert "content-type" in result
            assert result["content-type"] == "application/json"
            assert "x-api-key" in result

    def test_get_complete_url_with_base_url(self):
        """Test get_complete_url with base URL"""
        config = AzureAnthropicMessagesConfig()
        api_base = "https://test.services.ai.azure.com/anthropic"
        api_key = "test-api-key"
        model = "claude-sonnet-4-5"
        optional_params = {}
        litellm_params = {}

        url = config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        assert url == "https://test.services.ai.azure.com/anthropic/v1/messages"

    def test_get_complete_url_with_base_url_ending_with_slash(self):
        """Test get_complete_url with base URL ending with slash"""
        config = AzureAnthropicMessagesConfig()
        api_base = "https://test.services.ai.azure.com/anthropic/"
        api_key = "test-api-key"
        model = "claude-sonnet-4-5"
        optional_params = {}
        litellm_params = {}

        url = config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        assert url == "https://test.services.ai.azure.com/anthropic/v1/messages"

    def test_get_complete_url_with_base_url_already_containing_v1_messages(self):
        """Test get_complete_url with base URL already containing /v1/messages"""
        config = AzureAnthropicMessagesConfig()
        api_base = "https://test.services.ai.azure.com/anthropic/v1/messages"
        api_key = "test-api-key"
        model = "claude-sonnet-4-5"
        optional_params = {}
        litellm_params = {}

        url = config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        assert url == "https://test.services.ai.azure.com/anthropic/v1/messages"

    def test_get_complete_url_with_base_url_containing_anthropic(self):
        """Test get_complete_url with base URL already containing /anthropic"""
        config = AzureAnthropicMessagesConfig()
        api_base = "https://test.services.ai.azure.com/anthropic"
        api_key = "test-api-key"
        model = "claude-sonnet-4-5"
        optional_params = {}
        litellm_params = {}

        url = config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        assert url == "https://test.services.ai.azure.com/anthropic/v1/messages"

    def test_get_complete_url_with_base_url_without_anthropic(self):
        """Test get_complete_url with base URL without /anthropic"""
        config = AzureAnthropicMessagesConfig()
        api_base = "https://test.services.ai.azure.com"
        api_key = "test-api-key"
        model = "claude-sonnet-4-5"
        optional_params = {}
        litellm_params = {}

        url = config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        assert url == "https://test.services.ai.azure.com/anthropic/v1/messages"

    def test_get_complete_url_raises_error_when_api_base_missing(self):
        """Test get_complete_url raises error when api_base is None"""
        config = AzureAnthropicMessagesConfig()
        api_base = None
        api_key = "test-api-key"
        model = "claude-sonnet-4-5"
        optional_params = {}
        litellm_params = {}

        with patch("litellm.secret_managers.main.get_secret_str", return_value=None):
            with pytest.raises(ValueError, match="Missing Azure API Base"):
                config.get_complete_url(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                )

    def test_get_supported_anthropic_messages_params(self):
        """Test get_supported_anthropic_messages_params returns correct params"""
        config = AzureAnthropicMessagesConfig()
        model = "claude-sonnet-4-5"
        params = config.get_supported_anthropic_messages_params(model)

        assert "messages" in params
        assert "model" in params
        assert "max_tokens" in params
        assert "temperature" in params
        assert "tools" in params
        assert "tool_choice" in params

    def test_transform_anthropic_messages_request_removes_scope_from_cache_control(
        self,
    ):
        """Test that scope is removed from cache_control (Azure AI Foundry doesn't support it)"""
        config = AzureAnthropicMessagesConfig()
        model = "claude-sonnet-4-5"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral", "scope": "global"},
                    }
                ],
            }
        ]
        anthropic_messages_optional_request_params = {
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "You are an AI assistant.",
                    "cache_control": {"type": "ephemeral", "scope": "global"},
                }
            ],
        }
        litellm_params = GenericLiteLLMParams()
        headers = {}

        result = config.transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert "scope" not in result["system"][0]["cache_control"]
        assert result["system"][0]["cache_control"]["type"] == "ephemeral"
        assert "scope" not in result["messages"][0]["content"][0]["cache_control"]
        assert (
            result["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"
        )


class TestProviderConfigManagerAzureAnthropicMessages:
    """Test ProviderConfigManager returns correct config for Azure AI Anthropic Messages API"""

    def test_get_provider_anthropic_messages_config_returns_azure_config(self):
        """Test that ProviderConfigManager returns AzureAnthropicMessagesConfig for azure_ai provider with claude model"""
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="claude-sonnet-4-5_gb_20250929",
            provider=litellm.LlmProviders.AZURE_AI,
        )

        assert config is not None
        assert isinstance(config, AzureAnthropicMessagesConfig)

    def test_get_provider_anthropic_messages_config_case_insensitive_model_name(self):
        """Test that model name check is case insensitive"""
        import litellm
        from litellm.utils import ProviderConfigManager

        # Test with uppercase CLAUDE
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="CLAUDE-SONNET-4-5",
            provider=litellm.LlmProviders.AZURE_AI,
        )

        assert config is not None
        assert isinstance(config, AzureAnthropicMessagesConfig)

    def test_get_provider_anthropic_messages_config_returns_none_for_non_claude_model(
        self,
    ):
        """Test that ProviderConfigManager returns None for non-claude model on azure_ai"""
        import litellm
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="gpt-4o",
            provider=litellm.LlmProviders.AZURE_AI,
        )

        assert config is None


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled backup cost map so capability flags match this branch."""
    import litellm

    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


def test_messages_thinking_shape_follows_exact_azure_entry_flag(local_model_cost_map, monkeypatch):
    """The Azure messages config must probe capabilities under ``azure_ai`` so an
    operator setting ``supports_adaptive_thinking: false`` on the exact
    ``azure_ai/claude-opus-4-8`` entry beats the unmodified ``anthropic`` entry.
    With the inherited ``"anthropic"`` provider default the flip was ignored and
    the transform kept emitting ``thinking.type='adaptive'``."""
    import litellm

    config = AzureAnthropicMessagesConfig()

    def transform():
        return config.transform_anthropic_messages_request(
            model="claude-opus-4-8",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_optional_request_params={
                "max_tokens": 4096,
                "reasoning_effort": "medium",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    result = transform()
    assert result.get("thinking") == {"type": "adaptive"}
    assert result.get("output_config") == {"effort": "medium"}

    monkeypatch.setitem(
        litellm.model_cost["azure_ai/claude-opus-4-8"], "supports_adaptive_thinking", False
    )
    litellm.get_model_info.cache_clear()
    assert litellm.model_cost["claude-opus-4-8"]["supports_adaptive_thinking"] is True

    flipped = transform()
    thinking = flipped.get("thinking")
    assert isinstance(thinking, dict)
    assert thinking.get("type") == "enabled"
    assert isinstance(thinking.get("budget_tokens"), int)
    assert "output_config" not in flipped


def _azure_transform(model, messages, system=None):
    config = AzureAnthropicMessagesConfig()
    params = {"max_tokens": 256}
    if system is not None:
        params["system"] = system
    return config.transform_anthropic_messages_request(
        model=model,
        messages=copy.deepcopy(messages),
        anthropic_messages_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


class TestAzureAnthropicMidConversationSystem:
    """Azure AI Foundry serves Claude on the first-party Anthropic /v1/messages
    contract: a mid-conversation ``role: "system"`` reminder is accepted in place
    on Claude 4.8+/5 but 400s ("role 'system' is not supported on this model") on
    older Claude, and a *leading* system entry 400s on every model ("messages.0:
    use the top-level 'system' parameter"). These tests pin the model-aware hoist
    the config applies so Claude Code sessions neither collapse the prompt cache
    on 4.8+ nor hard-fail on 4.7 and older (RCA: Kraken Tech high-spend)."""

    def test_supported_model_keeps_mid_conversation_system_in_place(self, local_model_cost_map):
        messages = [
            {"role": "user", "content": "read the file"},
            {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        result = _azure_transform("claude-opus-4-8", messages)
        assert result["messages"] == messages

    def test_supported_model_hoists_only_leading_system_run(self, local_model_cost_map):
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": "Cite sources."},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "mid-conversation reminder"},
            {"role": "user", "content": "continue"},
        ]
        result = _azure_transform("claude-opus-4-8", messages)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "mid-conversation reminder"},
            {"role": "user", "content": "continue"},
        ]
        assert result["system"] == [
            {"type": "text", "text": "You are terse."},
            {"type": "text", "text": "Cite sources."},
        ]

    def test_unsupported_model_hoists_mid_conversation_system(self, local_model_cost_map):
        messages = [
            {"role": "user", "content": "read the file"},
            {"role": "system", "content": "[Truncated: PARTIAL view of big1.txt]"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        result = _azure_transform(
            "claude-opus-4-7", messages, system=[{"type": "text", "text": "Base."}]
        )
        assert result["messages"] == [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": "reading"},
            {"role": "user", "content": "continue"},
        ]
        assert result["system"] == [
            {"type": "text", "text": "Base."},
            {"type": "text", "text": "[Truncated: PARTIAL view of big1.txt]"},
        ]


def test_azure_claude_4_8_plus_cost_map_entries_carry_mid_conversation_system_flag():
    """Exact cost-map hits win over the ``claude-mid-conversation-system``
    fallback rule, so an ``azure_ai`` Claude 4.8+/5 entry missing the flag would
    be treated as unsupported and hoist every reminder, collapsing the prompt
    cache. Every mapped azure_ai entry the rule matches must carry the flag."""
    import re

    import litellm

    cost_map_path = os.path.join(
        os.path.dirname(litellm.__file__), "model_prices_and_context_window_backup.json"
    )
    with open(cost_map_path) as f:
        cost_map = json.load(f)
    rules = cost_map["fallback_generalizations"]["rules"]
    rule_pattern = next(
        (r["pattern"] for r in rules if r["name"] == "claude-mid-conversation-system"),
        None,
    )
    assert rule_pattern is not None, "claude-mid-conversation-system rule not found in fallback_generalizations"
    pattern = re.compile(rule_pattern, re.IGNORECASE)
    missing = [
        key
        for key, info in cost_map.items()
        if isinstance(info, dict)
        and info.get("litellm_provider") == "azure_ai"
        and pattern.search(key)
        and info.get("supports_mid_conversation_system") is not True
    ]
    assert missing == []
