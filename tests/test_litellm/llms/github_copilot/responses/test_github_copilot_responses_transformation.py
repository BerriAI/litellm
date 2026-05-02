"""
Tests for GitHub Copilot Responses API transformation

Tests the GithubCopilotResponsesAPIConfig class that handles GitHub Copilot-specific
transformations for the Responses API.

Source: litellm/llms/github_copilot/responses/transformation.py
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest
import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    """Pin litellm.model_cost to the bundled local backup so tests don't depend
    on remote catalog fetches (and don't change behavior across remote refreshes)."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url)
    )
    litellm.add_known_models(model_cost_map=litellm.model_cost)


class TestGithubCopilotResponsesAPITransformation:
    """Test GitHub Copilot Responses API configuration and transformations"""

    def test_github_copilot_provider_config_registration(self):
        """Test that GitHub Copilot provider returns the native Responses API
        config for a Responses-capable catalog model. Exercises the full stack:
        catalog lookup -> github_copilot_supports_responses_api -> native config."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/gpt-5.3-codex",
            provider=LlmProviders.GITHUB_COPILOT,
        )

        assert (
            config is not None
        ), "Config should not be None for GitHub Copilot provider"
        assert isinstance(
            config, GithubCopilotResponsesAPIConfig
        ), f"Expected GithubCopilotResponsesAPIConfig, got {type(config)}"
        assert (
            config.custom_llm_provider == LlmProviders.GITHUB_COPILOT
        ), "custom_llm_provider should be GITHUB_COPILOT"

    @patch("litellm.llms.github_copilot.responses.transformation.Authenticator")
    def test_github_copilot_responses_endpoint_url(self, mock_authenticator_class):
        """Test that get_complete_url returns correct GitHub Copilot endpoint"""
        # Mock authenticator to return default base
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = (
            "https://api.individual.githubcopilot.com"
        )
        mock_authenticator_class.return_value = mock_auth_instance

        config = GithubCopilotResponsesAPIConfig()

        # Test with default GitHub Copilot API base (from authenticator)
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert (
            url == "https://api.individual.githubcopilot.com/responses"
        ), f"Expected GitHub Copilot responses endpoint, got {url}"

        # Test with custom api_base (overrides authenticator)
        custom_url = config.get_complete_url(
            api_base="https://custom.githubcopilot.com", litellm_params={}
        )
        assert (
            custom_url == "https://custom.githubcopilot.com/responses"
        ), f"Expected custom endpoint, got {custom_url}"

        # Test with trailing slash
        url_with_slash = config.get_complete_url(
            api_base="https://api.githubcopilot.com/", litellm_params={}
        )
        assert (
            url_with_slash == "https://api.githubcopilot.com/responses"
        ), "Should handle trailing slash"

    @patch("litellm.llms.github_copilot.responses.transformation.Authenticator")
    def test_validate_environment_default_headers(self, mock_authenticator_class):
        """Test that validate_environment generates correct default headers"""
        # Mock the authenticator
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_key.return_value = "test-api-key-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = GithubCopilotResponsesAPIConfig()

        headers = config.validate_environment(
            headers={}, model="gpt-5.1-codex", litellm_params={}
        )

        # Check required headers
        assert headers["Authorization"] == "Bearer test-api-key-123"
        assert headers["content-type"] == "application/json"
        assert headers["copilot-integration-id"] == "vscode-chat"
        assert headers["editor-version"] == "vscode/1.95.0"
        assert headers["editor-plugin-version"] == "copilot-chat/0.26.7"
        assert headers["user-agent"] == "GitHubCopilotChat/0.26.7"
        assert headers["openai-intent"] == "conversation-panel"
        assert headers["x-github-api-version"] == "2025-04-01"
        assert "x-request-id" in headers

    @patch("litellm.llms.github_copilot.responses.transformation.Authenticator")
    def test_validate_environment_user_headers_override(self, mock_authenticator_class):
        """Test that user-provided headers override default headers"""
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_key.return_value = "test-api-key-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = GithubCopilotResponsesAPIConfig()

        custom_headers = {
            "editor-version": "custom/2.0.0",
            "custom-header": "custom-value",
        }

        headers = config.validate_environment(
            headers=custom_headers, model="gpt-5.1-codex", litellm_params={}
        )

        # User header should override default
        assert headers["editor-version"] == "custom/2.0.0"
        # Custom header should be preserved
        assert headers["custom-header"] == "custom-value"
        # Default headers should still be present
        assert headers["Authorization"] == "Bearer test-api-key-123"

    def test_get_initiator_with_assistant_role(self):
        """Test _get_initiator returns 'agent' for assistant role"""
        config = GithubCopilotResponsesAPIConfig()

        input_with_assistant = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        initiator = config._get_initiator(input_with_assistant)
        assert initiator == "agent", "Should return 'agent' for assistant role"

    def test_get_initiator_with_no_role(self):
        """Test _get_initiator returns 'agent' for items without role"""
        config = GithubCopilotResponsesAPIConfig()

        input_without_role = [
            {"role": "user", "content": "Hello"},
            {"type": "reasoning", "content": "thinking..."},  # No role field
        ]

        initiator = config._get_initiator(input_without_role)
        assert initiator == "agent", "Should return 'agent' for items without role"

    def test_get_initiator_with_user_only(self):
        """Test _get_initiator returns 'user' for user-only messages"""
        config = GithubCopilotResponsesAPIConfig()

        input_user_only = [{"role": "user", "content": "Hello"}]

        initiator = config._get_initiator(input_user_only)
        assert initiator == "user", "Should return 'user' for user-only messages"

    def test_get_initiator_with_string_input(self):
        """Test _get_initiator returns 'user' for string input"""
        config = GithubCopilotResponsesAPIConfig()

        initiator = config._get_initiator("Hello, how are you?")
        assert initiator == "user", "Should return 'user' for string input"

    def test_has_vision_input_with_input_image(self):
        """Test _has_vision_input detects input_image type"""
        config = GithubCopilotResponsesAPIConfig()

        input_with_vision = [
            {"role": "user", "content": [{"type": "input_image", "data": "base64..."}]}
        ]

        has_vision = config._has_vision_input(input_with_vision)
        assert has_vision is True, "Should detect input_image type"

    def test_has_vision_input_nested(self):
        """Test _has_vision_input detects nested input_image"""
        config = GithubCopilotResponsesAPIConfig()

        input_nested_vision = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "multipart",
                        "content": [{"type": "input_image", "data": "base64..."}],
                    },
                ],
            }
        ]

        has_vision = config._has_vision_input(input_nested_vision)
        assert has_vision is True, "Should detect nested input_image"

    def test_has_vision_input_without_vision(self):
        """Test _has_vision_input returns False for text-only input"""
        config = GithubCopilotResponsesAPIConfig()

        input_text_only = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        has_vision = config._has_vision_input(input_text_only)
        assert has_vision is False, "Should return False for text-only input"

    def test_has_vision_input_with_string(self):
        """Test _has_vision_input returns False for string input"""
        config = GithubCopilotResponsesAPIConfig()

        has_vision = config._has_vision_input("Just a text message")
        assert has_vision is False, "Should return False for string input"

    @patch("litellm.llms.github_copilot.responses.transformation.Authenticator")
    def test_validate_environment_with_vision_header(self, mock_authenticator_class):
        """Test that copilot-vision-request header is added for vision input"""
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_key.return_value = "test-api-key"
        mock_authenticator_class.return_value = mock_auth_instance

        config = GithubCopilotResponsesAPIConfig()

        # Create mock litellm_params with input attribute
        mock_litellm_params = MagicMock()
        mock_litellm_params.input = [
            {
                "role": "user",
                "content": [{"type": "input_image", "data": "base64..."}],
            }
        ]

        headers = config.validate_environment(
            headers={}, model="gpt-5.1-codex", litellm_params=mock_litellm_params
        )

        assert (
            headers.get("copilot-vision-request") == "true"
        ), "Should add copilot-vision-request header for vision input"

    @patch("litellm.llms.github_copilot.responses.transformation.Authenticator")
    def test_validate_environment_with_x_initiator(self, mock_authenticator_class):
        """Test that X-Initiator header is set based on input"""
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_key.return_value = "test-api-key"
        mock_authenticator_class.return_value = mock_auth_instance

        config = GithubCopilotResponsesAPIConfig()

        # Create mock litellm_params with input attribute
        mock_litellm_params = MagicMock()
        mock_litellm_params.input = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        headers = config.validate_environment(
            headers={}, model="gpt-5.1-codex", litellm_params=mock_litellm_params
        )

        assert (
            headers.get("X-Initiator") == "agent"
        ), "Should set X-Initiator to 'agent' for assistant role"

    def test_map_openai_params_no_transformation(self):
        """Test that map_openai_params passes through parameters unchanged"""
        config = GithubCopilotResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            temperature=0.7, max_output_tokens=1000, stream=False
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="gpt-5.1-codex",
            drop_params=False,
        )

        assert result.get("temperature") == 0.7
        assert result.get("max_output_tokens") == 1000
        assert result.get("stream") is False

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns expected parameters"""
        config = GithubCopilotResponsesAPIConfig()

        supported = config.get_supported_openai_params("gpt-5.1-codex")

        # Should include standard OpenAI Responses API parameters
        expected_params = [
            "model",
            "input",
            "instructions",
            "temperature",
            "max_output_tokens",
            "stream",
            "tools",
            "tool_choice",
        ]

        for param in expected_params:
            assert param in supported, f"{param} should be in supported params"

    def test_handle_reasoning_item_preserves_encrypted_content(self):
        """Test that _handle_reasoning_item preserves encrypted_content for GitHub Copilot.

        GitHub Copilot uses encrypted_content in reasoning items to maintain
        conversation state across turns. This field must be preserved for
        multi-turn conversations to work.
        """
        config = GithubCopilotResponsesAPIConfig()

        reasoning_item = {
            "type": "reasoning",
            "id": "reasoning-123",
            "summary": ["Step 1", "Step 2"],
            "encrypted_content": "encrypted-blob-abc123",
            "status": None,  # Should be filtered out
            "content": None,  # Should be filtered out
        }

        result = config._handle_reasoning_item(reasoning_item)

        # encrypted_content should be preserved
        assert (
            result.get("encrypted_content") == "encrypted-blob-abc123"
        ), "encrypted_content must be preserved for GitHub Copilot multi-turn conversations"
        # status=None should be filtered out
        assert "status" not in result, "status=None should be filtered out"
        # content=None should be filtered out
        assert "content" not in result, "content=None should be filtered out"
        # Other fields should be preserved
        assert result.get("type") == "reasoning"
        assert result.get("id") == "reasoning-123"
        assert result.get("summary") == ["Step 1", "Step 2"]

    def test_handle_reasoning_item_without_encrypted_content(self):
        """Test _handle_reasoning_item when encrypted_content is not present"""
        config = GithubCopilotResponsesAPIConfig()

        reasoning_item = {
            "type": "reasoning",
            "id": "reasoning-456",
            "summary": ["Thinking..."],
            "status": None,
        }

        result = config._handle_reasoning_item(reasoning_item)

        # Should not have encrypted_content key at all
        assert "encrypted_content" not in result
        # status=None should be filtered out
        assert "status" not in result
        # Other fields preserved
        assert result.get("type") == "reasoning"
        assert result.get("id") == "reasoning-456"

    def test_handle_reasoning_item_non_reasoning_passthrough(self):
        """Test _handle_reasoning_item passes through non-reasoning items unchanged"""
        config = GithubCopilotResponsesAPIConfig()

        message_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello"}],
        }

        result = config._handle_reasoning_item(message_item)

        # Non-reasoning items should pass through unchanged
        assert result == message_item


class TestGithubCopilotResponsesAPIRouting:
    """``ProviderConfigManager.get_provider_responses_api_config`` for github_copilot
    returns the native Responses config only when the model has ``mode=responses``
    in the (already-merged) model info; otherwise returns None so the dispatcher
    routes through the chat-completions translation bridge."""

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_returns_config_when_mode_is_responses(self, mock_get_info):
        """``mode=responses`` returns native config."""
        mock_get_info.return_value = {"mode": "responses"}
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-responses-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert isinstance(config, GithubCopilotResponsesAPIConfig)

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_returns_none_when_mode_is_chat(self, mock_get_info):
        """``mode=chat`` returns None so dispatcher uses bridge."""
        mock_get_info.return_value = {"mode": "chat"}
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-chat-only-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert config is None

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_returns_none_when_mode_is_unset_and_no_endpoints(self, mock_get_info):
        """Entry without ``mode`` and without ``supported_endpoints`` returns None
        (conservative default)."""
        mock_get_info.return_value = {}
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert config is None

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_returns_config_when_mode_unset_but_endpoints_have_responses(
        self, mock_get_info
    ):
        """``mode`` unset but ``supported_endpoints`` declaring /v1/responses
        returns native config (endpoint-list fallback for stale-but-correct
        catalog entries that lack ``mode``)."""
        mock_get_info.return_value = {
            "supported_endpoints": ["/v1/chat/completions", "/v1/responses"]
        }
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert isinstance(config, GithubCopilotResponsesAPIConfig)

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_mode_chat_overrides_endpoints_with_responses(self, mock_get_info):
        """``mode=chat`` is a hard opt-out: forces bridge even when
        ``supported_endpoints`` includes /v1/responses. Lets users force the
        bridge for dual-endpoint models without clearing endpoint metadata."""
        mock_get_info.return_value = {
            "mode": "chat",
            "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
        }
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert config is None

    def test_returns_config_when_model_is_none(self):
        """Follow-up GET/DELETE operations pass model=None and keep the native
        config path (no per-model lookup is possible)."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=None,
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert isinstance(config, GithubCopilotResponsesAPIConfig)

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_returns_none_when_get_model_info_raises(self, mock_get_info):
        """Catalog lookup failure (model not registered) returns None
        (conservative default; bridge handles unknown models safely)."""
        mock_get_info.side_effect = Exception("model not in catalog")
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/never-seen-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert config is None

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_user_override_via_register_model(self, mock_get_info):
        """User-supplied per-deployment ``model_info`` flows through
        ``litellm.register_model`` (called by the router) into the merged
        catalog read by ``_get_model_info_helper``. Setting ``mode=responses``
        for a model whose catalog entry says ``mode=chat`` therefore opts in
        to native dispatch without any per-call argument plumbing."""
        mock_get_info.return_value = {"mode": "responses"}
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-chat-only-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert isinstance(config, GithubCopilotResponsesAPIConfig)

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_realistic_chat_only_entry_returns_none(self, mock_get_info):
        """Realistic ``model_prices_and_context_window.json`` shape for a
        chat-only Copilot model (e.g. github_copilot/gemini-3.1-pro-preview)
        returns None so /v1/responses calls fall back to the bridge."""
        mock_get_info.return_value = {
            "litellm_provider": "github_copilot",
            "max_input_tokens": 136000,
            "max_output_tokens": 64000,
            "max_tokens": 64000,
            "mode": "chat",
            "supported_endpoints": ["/v1/chat/completions"],
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
            "supports_reasoning": True,
        }
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-chat-only-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert config is None

    @patch(
        "litellm.llms.github_copilot.responses.transformation._get_model_info_helper"
    )
    def test_realistic_responses_only_entry_returns_config(self, mock_get_info):
        """Realistic catalog entry for a Responses-only Copilot model
        (e.g. github_copilot/gpt-5.5) returns the native config."""
        mock_get_info.return_value = {
            "litellm_provider": "github_copilot",
            "max_input_tokens": 272000,
            "max_output_tokens": 128000,
            "max_tokens": 128000,
            "mode": "responses",
            "supported_endpoints": ["/v1/responses"],
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_parallel_function_calling": True,
            "supports_response_schema": True,
            "supports_vision": True,
            "supports_reasoning": True,
            "supports_none_reasoning_effort": True,
            "supports_xhigh_reasoning_effort": True,
        }
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/some-responses-only-model",
            provider=LlmProviders.GITHUB_COPILOT,
        )
        assert isinstance(config, GithubCopilotResponsesAPIConfig)
