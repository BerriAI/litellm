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
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


class TestGithubCopilotResponsesAPITransformation:
    """Test GitHub Copilot Responses API configuration and transformations"""

    def test_github_copilot_provider_config_registration(self):
        """Test that GitHub Copilot provider returns GithubCopilotResponsesAPIConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="github_copilot/gpt-5.1-codex",
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
        assert url == "https://api.individual.githubcopilot.com/responses", (
            f"Expected GitHub Copilot responses endpoint, got {url}"
        )

        # Test with custom api_base (overrides authenticator)
        custom_url = config.get_complete_url(
            api_base="https://custom.githubcopilot.com", litellm_params={}
        )
        assert custom_url == "https://custom.githubcopilot.com/responses", (
            f"Expected custom endpoint, got {custom_url}"
        )

        # Test with trailing slash
        url_with_slash = config.get_complete_url(
            api_base="https://api.githubcopilot.com/", litellm_params={}
        )
        assert url_with_slash == "https://api.githubcopilot.com/responses", (
            "Should handle trailing slash"
        )

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

        assert headers.get("copilot-vision-request") == "true", (
            "Should add copilot-vision-request header for vision input"
        )

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

        assert headers.get("X-Initiator") == "agent", (
            "Should set X-Initiator to 'agent' for assistant role"
        )

    def test_map_openai_params_no_transformation(self):
        """Test that map_openai_params passes through parameters unchanged"""
        config = GithubCopilotResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            temperature=0.7, max_output_tokens=1000, stream=False
        )

        result = config.map_openai_params(
            response_api_optional_params=params, model="gpt-5.1-codex", drop_params=False
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
        assert result.get("encrypted_content") == "encrypted-blob-abc123", (
            "encrypted_content must be preserved for GitHub Copilot multi-turn conversations"
        )
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
