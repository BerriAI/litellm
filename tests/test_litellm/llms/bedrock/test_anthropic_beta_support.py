"""
Test anthropic_beta header support for AWS Bedrock.

Tests that anthropic-beta headers are correctly processed and passed to AWS Bedrock
for enabling beta features like 1M context window, computer use tools, etc.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.common_utils import get_anthropic_beta_from_headers
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)


class TestAnthropicBetaHeaderSupport:
    """Test anthropic_beta header functionality across Bedrock APIs."""

    def test_get_anthropic_beta_from_headers_empty(self):
        """Test header extraction with no headers."""
        headers = {}
        result = get_anthropic_beta_from_headers(headers)
        assert result == []

    def test_get_anthropic_beta_from_headers_single(self):
        """Test header extraction with single beta header."""
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        result = get_anthropic_beta_from_headers(headers)
        assert result == ["context-1m-2025-08-07"]

    def test_get_anthropic_beta_from_headers_multiple(self):
        """Test header extraction with multiple comma-separated beta headers."""
        headers = {"anthropic-beta": "context-1m-2025-08-07,computer-use-2024-10-22"}
        result = get_anthropic_beta_from_headers(headers)
        assert result == ["context-1m-2025-08-07", "computer-use-2024-10-22"]

    def test_get_anthropic_beta_from_headers_whitespace(self):
        """Test header extraction handles whitespace correctly."""
        headers = {"anthropic-beta": " context-1m-2025-08-07 , computer-use-2024-10-22 "}
        result = get_anthropic_beta_from_headers(headers)
        assert result == ["context-1m-2025-08-07", "computer-use-2024-10-22"]

    def test_invoke_transformation_anthropic_beta(self):
        """Test that Invoke API transformation includes anthropic_beta in request."""
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07,computer-use-2024-10-22"}
        
        result = config.transform_request(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers
        )
        
        assert "anthropic_beta" in result
        # Beta flags are stored as sets, so order may vary
        assert set(result["anthropic_beta"]) == {"context-1m-2025-08-07", "computer-use-2024-10-22"}

    def test_converse_transformation_anthropic_beta(self):
        """Test that Converse API transformation includes anthropic_beta in additionalModelRequestFields."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14"}
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        assert "additionalModelRequestFields" in result
        additional_fields = result["additionalModelRequestFields"]
        assert "anthropic_beta" in additional_fields
        # Sort both arrays before comparing to avoid flakiness from ordering differences
        assert sorted(additional_fields["anthropic_beta"]) == sorted(["context-1m-2025-08-07", "interleaved-thinking-2025-05-14"])

    def test_messages_transformation_anthropic_beta(self):
        """Test that Messages API transformation includes anthropic_beta in request."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "output-128k-2025-02-19"}
        
        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )
        
        assert "anthropic_beta" in result
        # Sort both arrays before comparing to avoid flakiness from ordering differences
        assert sorted(result["anthropic_beta"]) == sorted(["output-128k-2025-02-19"])

    def test_converse_computer_use_compatibility(self):
        """Test that user anthropic_beta headers work with computer use tools."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        # Computer use tools should automatically add computer-use-2024-10-22
        tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768
            }
        ]
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result["additionalModelRequestFields"]
        betas = additional_fields["anthropic_beta"]
        
        # Should contain both user-provided and auto-added beta headers
        assert "context-1m-2025-08-07" in betas
        assert "computer-use-2024-10-22" in betas
        assert len(betas) == 2  # No duplicates

    def test_no_anthropic_beta_headers(self):
        """Test that transformations work correctly when no anthropic_beta headers are provided."""
        config = AmazonConverseConfig()
        headers = {}
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in additional_fields

    def test_anthropic_beta_all_supported_features(self):
        """Test that all documented beta features are properly handled."""
        supported_features = [
            "context-1m-2025-08-07",
            "computer-use-2025-01-24",
            "computer-use-2024-10-22",
            "token-efficient-tools-2025-02-19",
            "interleaved-thinking-2025-05-14",
            "output-128k-2025-02-19",
            "dev-full-thinking-2025-05-14"
        ]
        
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": ",".join(supported_features)}
        
        result = config.transform_request(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers
        )
        
        assert "anthropic_beta" in result
        # Beta flags are stored as sets, so order may vary
        assert set(result["anthropic_beta"]) == set(supported_features)

    def test_prompt_caching_no_beta_header_messages_api(self):
        """Test that prompt caching (cache_control) does NOT add prompt-caching-2024-07-31 beta header for Bedrock.
        
        Bedrock recognizes prompt caching via the request body (cache_control field),
        not through beta headers. This test verifies the fix.
        """
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {}
        
        # Messages with cache_control set (prompt caching enabled)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
        
        result = config.transform_anthropic_messages_request(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )
        
        # Verify prompt-caching-2024-07-31 is NOT in anthropic_beta
        if "anthropic_beta" in result:
            assert "prompt-caching-2024-07-31" not in result["anthropic_beta"], (
                "prompt-caching-2024-07-31 should not be added as a beta header for Bedrock. "
                "Bedrock recognizes prompt caching via cache_control in the request body, not beta headers."
            )
        else:
            # It's also valid if anthropic_beta is not present at all
            assert True

    def test_prompt_caching_no_beta_header_chat_api(self):
        """Test that prompt caching (cache_control) does NOT add prompt-caching-2024-07-31 beta header for Bedrock Chat API.
        
        Bedrock recognizes prompt caching via the request body (cache_control field),
        not through beta headers. This test verifies the fix.
        """
        config = AmazonAnthropicClaudeConfig()
        headers = {}
        
        # Messages with cache_control set (prompt caching enabled)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
        
        result = config.transform_request(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers=headers
        )
        
        # Verify prompt-caching-2024-07-31 is NOT in anthropic_beta
        if "anthropic_beta" in result:
            assert "prompt-caching-2024-07-31" not in result["anthropic_beta"], (
                "prompt-caching-2024-07-31 should not be added as a beta header for Bedrock. "
                "Bedrock recognizes prompt caching via cache_control in the request body, not beta headers."
            )
        else:
            # It's also valid if anthropic_beta is not present at all
            assert True

    def test_prompt_caching_with_other_beta_headers(self):
        """Test that prompt caching doesn't interfere with other valid beta headers."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        # Messages with cache_control set
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
        
        result = config.transform_anthropic_messages_request(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )
        
        # Should have the user-provided beta header but NOT prompt-caching
        if "anthropic_beta" in result:
            assert "context-1m-2025-08-07" in result["anthropic_beta"]
            assert "prompt-caching-2024-07-31" not in result["anthropic_beta"]
        else:
            # If no beta headers, that's also fine
            assert True

    def test_converse_non_anthropic_model_no_anthropic_beta(self):
        """Test that non-Anthropic models (e.g., Qwen) do NOT get anthropic_beta in additionalModelRequestFields.
        
        This is critical because non-Anthropic models on Bedrock will error with
        "unknown variant anthropic_beta" if this field is included.
        """
        config = AmazonConverseConfig()
        # Even if headers contain anthropic-beta, non-Anthropic models should NOT get it
        headers = {"anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14"}
        
        # Test with Qwen model (using ARN format like the user's config)
        result = config._transform_request_helper(
            model="qwen.qwen3-coder-480b-a35b-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in additional_fields, (
            "anthropic_beta should NOT be added for non-Anthropic models like Qwen. "
            "This field is only supported by Anthropic/Claude models on Bedrock."
        )

    def test_converse_llama_model_no_anthropic_beta(self):
        """Test that Llama models do NOT get anthropic_beta in additionalModelRequestFields."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        result = config._transform_request_helper(
            model="meta.llama3-2-11b-instruct-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in additional_fields, (
            "anthropic_beta should NOT be added for Llama models."
        )

    def test_converse_nova_model_no_anthropic_beta(self):
        """Test that Amazon Nova models do NOT get anthropic_beta in additionalModelRequestFields."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "computer-use-2024-10-22"}
        
        result = config._transform_request_helper(
            model="amazon.nova-pro-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in additional_fields, (
            "anthropic_beta should NOT be added for Amazon Nova models."
        )

    def test_converse_anthropic_model_gets_anthropic_beta(self):
        """Test that Anthropic models DO get anthropic_beta in additionalModelRequestFields."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" in additional_fields, (
            "anthropic_beta SHOULD be added for Anthropic models."
        )
        assert "context-1m-2025-08-07" in additional_fields["anthropic_beta"]

    def test_converse_anthropic_model_with_cross_region_prefix(self):
        """Test that Anthropic models with cross-region prefix still get anthropic_beta."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        # Model with 'us.' cross-region prefix
        result = config._transform_request_helper(
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" in additional_fields, (
            "anthropic_beta SHOULD be added for Anthropic models with cross-region prefix."
        )
        assert "context-1m-2025-08-07" in additional_fields["anthropic_beta"]
