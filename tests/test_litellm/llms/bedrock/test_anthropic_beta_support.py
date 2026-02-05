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
            model="anthropic.claude-opus-4-5-20250514-v1:0",
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
            model="anthropic.claude-opus-4-5-20250514-v1:0",
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
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )

        additional_fields = result["additionalModelRequestFields"]
        betas = additional_fields["anthropic_beta"]

        # Should contain both user-provided and auto-added beta headers
        assert "context-1m-2025-08-07" in betas
        # Opus 4.5 gets computer-use-2025-11-24 (not the older 2024-10-22)
        assert "computer-use-2025-11-24" in betas
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


class TestBedrockBetaHeaderFiltering:
    """Test centralized beta header filtering across all Bedrock APIs."""

    def test_invoke_chat_filters_unsupported_headers(self):
        """Test that Invoke Chat API filters out unsupported beta headers."""
        config = AmazonAnthropicClaudeConfig()
        headers = {
            "anthropic-beta": "computer-use-2025-01-24,unknown-beta-2099-01-01,context-1m-2025-08-07"
        }

        result = config.transform_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )

        assert "anthropic_beta" in result
        beta_set = set(result["anthropic_beta"])

        # Should keep supported headers
        assert "computer-use-2025-01-24" in beta_set
        assert "context-1m-2025-08-07" in beta_set

        # Should filter out unsupported header
        assert "unknown-beta-2099-01-01" not in beta_set

    def test_converse_filters_unsupported_headers(self):
        """Test that Converse API filters out unsupported beta headers."""
        config = AmazonConverseConfig()
        headers = {
            "anthropic-beta": "interleaved-thinking-2025-05-14,unknown-beta-2099-01-01"
        }

        result = config._transform_request_helper(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers,
        )

        additional_fields = result["additionalModelRequestFields"]
        beta_list = additional_fields["anthropic_beta"]

        # Should keep supported header
        assert "interleaved-thinking-2025-05-14" in beta_list

        # Should filter out unsupported header
        assert "unknown-beta-2099-01-01" not in beta_list

    def test_messages_filters_unsupported_headers(self):
        """Test that Messages API filters out unsupported beta headers."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {
            "anthropic-beta": "output-128k-2025-02-19,unknown-beta-2099-01-01"
        }

        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers,
        )

        assert "anthropic_beta" in result
        beta_list = result["anthropic_beta"]

        # Should keep supported header
        assert "output-128k-2025-02-19" in beta_list

        # Should filter out unsupported header
        assert "unknown-beta-2099-01-01" not in beta_list

    def test_version_based_filtering_thinking_headers(self):
        """Test that thinking headers are filtered based on model version."""
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}

        # Claude 4.5 should support thinking
        result_45 = config.transform_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        assert "anthropic_beta" in result_45
        assert "interleaved-thinking-2025-05-14" in result_45["anthropic_beta"]

        # Claude 3.5 should NOT support thinking
        result_35 = config.transform_request(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        # Should either not have anthropic_beta or not contain thinking header
        if "anthropic_beta" in result_35:
            assert "interleaved-thinking-2025-05-14" not in result_35["anthropic_beta"]

    def test_family_restriction_effort_opus_only(self):
        """Test that effort parameter only works on Opus 4.5+."""
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": "effort-2025-11-24"}

        # Opus 4.5 should support effort
        result_opus = config.transform_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        assert "anthropic_beta" in result_opus
        assert "effort-2025-11-24" in result_opus["anthropic_beta"]

        # Sonnet 4.5 should NOT support effort (wrong family)
        result_sonnet = config.transform_request(
            model="anthropic.claude-sonnet-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        # Should either not have anthropic_beta or not contain effort
        if "anthropic_beta" in result_sonnet:
            assert "effort-2025-11-24" not in result_sonnet["anthropic_beta"]

    def test_tool_search_family_restriction(self):
        """Test that tool search works on Opus and Sonnet 4.5+, but not Haiku."""
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": "tool-search-tool-2025-10-19"}

        # Opus 4.5 should support tool search
        result_opus = config.transform_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        assert "tool-search-tool-2025-10-19" in result_opus["anthropic_beta"]

        # Sonnet 4.5 should support tool search
        result_sonnet = config.transform_request(
            model="anthropic.claude-sonnet-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        assert "tool-search-tool-2025-10-19" in result_sonnet["anthropic_beta"]

        # Haiku 4.5 should NOT support tool search (wrong family)
        result_haiku = config.transform_request(
            model="anthropic.claude-haiku-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )
        # Should either not have anthropic_beta or not contain tool search
        if "anthropic_beta" in result_haiku:
            assert "tool-search-tool-2025-10-19" not in result_haiku["anthropic_beta"]

    def test_messages_advanced_tool_use_translation(self):
        """Test that Messages API translates advanced-tool-use to tool search headers."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

        # Opus 4.5 should translate advanced-tool-use
        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers,
        )

        assert "anthropic_beta" in result
        beta_list = result["anthropic_beta"]

        # Should translate to tool search headers
        assert "tool-search-tool-2025-10-19" in beta_list
        assert "tool-examples-2025-10-29" in beta_list

        # Should NOT contain original advanced-tool-use header
        assert "advanced-tool-use-2025-11-20" not in beta_list

    def test_messages_advanced_tool_use_no_translation_old_model(self):
        """Test that advanced-tool-use is NOT translated on older models."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

        # Claude 4.0 should NOT translate (too old)
        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-opus-4-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers,
        )

        # Should not have anthropic_beta or should be empty
        # (advanced-tool-use is not in whitelist and shouldn't translate)
        if "anthropic_beta" in result:
            assert len(result["anthropic_beta"]) == 0

    def test_messages_advanced_tool_use_no_translation_haiku(self):
        """Test that advanced-tool-use is NOT translated on Haiku (wrong family)."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

        # Haiku 4.5 should NOT translate (wrong family)
        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-haiku-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers,
        )

        # Should not have anthropic_beta or should be empty
        if "anthropic_beta" in result:
            assert len(result["anthropic_beta"]) == 0

    def test_cross_api_consistency(self):
        """Test that same headers work consistently across all three APIs."""
        headers = {"anthropic-beta": "computer-use-2025-01-24,context-1m-2025-08-07"}
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        # Invoke Chat
        config_invoke = AmazonAnthropicClaudeConfig()
        result_invoke = config_invoke.transform_request(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )

        # Converse
        config_converse = AmazonConverseConfig()
        result_converse = config_converse._transform_request_helper(
            model=model,
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers,
        )

        # Messages
        config_messages = AmazonAnthropicClaudeMessagesConfig()
        result_messages = config_messages.transform_anthropic_messages_request(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers,
        )

        # All should have the same beta headers
        invoke_betas = set(result_invoke["anthropic_beta"])
        converse_betas = set(
            result_converse["additionalModelRequestFields"]["anthropic_beta"]
        )
        messages_betas = set(result_messages["anthropic_beta"])

        assert invoke_betas == converse_betas == messages_betas
        assert "computer-use-2025-01-24" in invoke_betas
        assert "context-1m-2025-08-07" in invoke_betas

    def test_backward_compatibility_existing_headers(self):
        """Test that all previously supported headers still work after migration."""
        config = AmazonAnthropicClaudeConfig()

        # Test all 11 core supported beta headers
        all_headers = [
            "computer-use-2024-10-22",
            "computer-use-2025-01-24",
            "token-efficient-tools-2025-02-19",
            "interleaved-thinking-2025-05-14",
            "output-128k-2025-02-19",
            "dev-full-thinking-2025-05-14",
            "context-1m-2025-08-07",
            "context-management-2025-06-27",
            "effort-2025-11-24",
            "tool-search-tool-2025-10-19",
            "tool-examples-2025-10-29",
        ]

        headers = {"anthropic-beta": ",".join(all_headers)}

        result = config.transform_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers,
        )

        assert "anthropic_beta" in result
        result_betas = set(result["anthropic_beta"])

        # All headers should be present (Opus 4.5 supports all of them)
        for header in all_headers:
            assert header in result_betas, f"Header {header} was filtered out unexpectedly"

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

        # Use Claude 4.5+ model since several features require 4.0+
        result = config.transform_request(
            model="anthropic.claude-opus-4-5-20250514-v1:0",
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
        headers = {"anthropic-beta": "computer-use-2025-01-24"}

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
        assert "computer-use-2025-01-24" in additional_fields["anthropic_beta"]

    def test_converse_anthropic_model_with_cross_region_prefix(self):
        """Test that Anthropic models with cross-region prefix still get anthropic_beta."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "computer-use-2025-01-24"}

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
        assert "computer-use-2025-01-24" in additional_fields["anthropic_beta"]

    def test_messages_advanced_tool_use_translation_opus_4_5(self):
        """Test that advanced-tool-use header is translated to Bedrock-specific headers for Opus 4.5.

        Regression test for: Claude Code sends advanced-tool-use-2025-11-20 header which needs
        to be translated to tool-search-tool-2025-10-19 and tool-examples-2025-10-29 for
        Bedrock Invoke API on Claude Opus 4.5.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html
        """
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

        result = config.transform_anthropic_messages_request(
            model="us.anthropic.claude-opus-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )

        assert "anthropic_beta" in result
        beta_headers = result["anthropic_beta"]

        # advanced-tool-use should be removed
        assert "advanced-tool-use-2025-11-20" not in beta_headers, (
            "advanced-tool-use-2025-11-20 should be removed for Bedrock Invoke API"
        )

        # Bedrock-specific headers should be added for Opus 4.5
        assert "tool-search-tool-2025-10-19" in beta_headers, (
            "tool-search-tool-2025-10-19 should be added for Opus 4.5"
        )
        assert "tool-examples-2025-10-29" in beta_headers, (
            "tool-examples-2025-10-29 should be added for Opus 4.5"
        )

    def test_messages_advanced_tool_use_translation_sonnet_4_5(self):
        """Test that advanced-tool-use header is translated to Bedrock-specific headers for Sonnet 4.5.

        Regression test for: Claude Code sends advanced-tool-use-2025-11-20 header which needs
        to be translated to tool-search-tool-2025-10-19 and tool-examples-2025-10-29 for
        Bedrock Invoke API on Claude Sonnet 4.5.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
        """
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

        result = config.transform_anthropic_messages_request(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )

        assert "anthropic_beta" in result
        beta_headers = result["anthropic_beta"]

        # advanced-tool-use should be removed
        assert "advanced-tool-use-2025-11-20" not in beta_headers, (
            "advanced-tool-use-2025-11-20 should be removed for Bedrock Invoke API"
        )

        # Bedrock-specific headers should be added for Sonnet 4.5
        assert "tool-search-tool-2025-10-19" in beta_headers, (
            "tool-search-tool-2025-10-19 should be added for Sonnet 4.5"
        )
        assert "tool-examples-2025-10-29" in beta_headers, (
            "tool-examples-2025-10-29 should be added for Sonnet 4.5"
        )

    def test_messages_advanced_tool_use_filtered_unsupported_model(self):
        """Test that advanced-tool-use header is filtered out for models that don't support tool search.

        The translation to Bedrock-specific headers should only happen for models that
        support tool search on Bedrock (Opus 4.5, Sonnet 4.5).
        For other models, the advanced-tool-use header should just be removed.
        """
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "advanced-tool-use-2025-11-20"}

        # Test with Claude 3.5 Sonnet (does NOT support tool search on Bedrock)
        result = config.transform_anthropic_messages_request(
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )

        beta_headers = result.get("anthropic_beta", [])

        # advanced-tool-use should be removed
        assert "advanced-tool-use-2025-11-20" not in beta_headers

        # Bedrock-specific headers should NOT be added for unsupported models
        assert "tool-search-tool-2025-10-19" not in beta_headers
        assert "tool-examples-2025-10-29" not in beta_headers


class TestContextManagementBodyParamStripping:
    """Test that context_management is stripped from request body for Bedrock APIs.

    Bedrock doesn't support context_management as a request body parameter.
    The feature is enabled via the anthropic-beta header instead. If left in the body,
    Bedrock returns: 'context_management: Extra inputs are not permitted'.
    """

    def test_messages_api_strips_context_management(self):
        """Test that Messages API removes context_management from request body."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {}

        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-sonnet-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={
                "max_tokens": 100,
                "context_management": {"type": "automatic", "max_context_tokens": 50000},
            },
            litellm_params={},
            headers=headers,
        )

        # context_management must NOT be in the request body
        assert "context_management" not in result

    def test_invoke_chat_api_strips_context_management(self):
        """Test that Invoke Chat API removes context_management from request body."""
        config = AmazonAnthropicClaudeConfig()
        headers = {}

        result = config.transform_request(
            model="anthropic.claude-sonnet-4-5-20250514-v1:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={
                "context_management": {"type": "automatic", "max_context_tokens": 50000},
            },
            litellm_params={},
            headers=headers,
        )

        # context_management must NOT be in the request body
        assert "context_management" not in result

    def test_converse_api_strips_context_management(self):
        """Test that Converse API doesn't pass context_management in additionalModelRequestFields."""
        config = AmazonConverseConfig()
        headers = {}

        result = config._transform_request_helper(
            model="anthropic.claude-sonnet-4-5-20250514-v1:0",
            system_content_blocks=[],
            optional_params={
                "context_management": {"type": "automatic", "max_context_tokens": 50000},
            },
            messages=[{"role": "user", "content": "Test"}],
            headers=headers,
        )

        additional_fields = result.get("additionalModelRequestFields", {})
        # context_management must NOT leak into additionalModelRequestFields
        assert "context_management" not in additional_fields
