"""
Comprehensive tests for Bedrock beta headers configuration.

Tests the centralized whitelist-based filtering with version-based model support.
"""

import pytest

from litellm.llms.bedrock.beta_headers_config import (
    BEDROCK_CORE_SUPPORTED_BETAS,
    BedrockAPI,
    BedrockBetaHeaderFilter,
    get_bedrock_beta_filter,
)


class TestBedrockBetaHeaderFilter:
    """Test the BedrockBetaHeaderFilter class."""

    def test_factory_function(self):
        """Test factory function returns correct filter instance."""
        filter_chat = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        assert isinstance(filter_chat, BedrockBetaHeaderFilter)
        assert filter_chat.api_type == BedrockAPI.INVOKE_CHAT

        filter_converse = get_bedrock_beta_filter(BedrockAPI.CONVERSE)
        assert isinstance(filter_converse, BedrockBetaHeaderFilter)
        assert filter_converse.api_type == BedrockAPI.CONVERSE

    def test_whitelist_filtering_basic(self):
        """Test basic whitelist filtering keeps supported headers."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        # Supported header should pass through
        result = filter_obj.filter_beta_headers(
            ["computer-use-2025-01-24"], model, translate=False
        )
        assert result == ["computer-use-2025-01-24"]

    def test_whitelist_filtering_blocks_unsupported(self):
        """Test whitelist filtering blocks unsupported headers."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        # Unsupported header should be filtered out
        result = filter_obj.filter_beta_headers(
            ["unknown-beta-2099-01-01"], model, translate=False
        )
        assert result == []

    def test_whitelist_filtering_mixed_headers(self):
        """Test filtering with mix of supported and unsupported headers."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            [
                "computer-use-2025-01-24",  # Supported
                "unknown-beta-2099-01-01",  # Unsupported
                "effort-2025-11-24",  # Supported
            ],
            model,
            translate=False,
        )
        # Should only keep supported headers
        assert set(result) == {"computer-use-2025-01-24", "effort-2025-11-24"}

    def test_empty_headers_list(self):
        """Test filtering with empty headers list."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers([], model)
        assert result == []

    def test_all_supported_betas_in_whitelist(self):
        """Test that all core supported betas are in whitelist."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        assert len(filter_obj.supported_betas) == len(BEDROCK_CORE_SUPPORTED_BETAS)


class TestModelVersionExtraction:
    """Test model version extraction logic."""

    def test_extract_version_opus_4_5(self):
        """Test version extraction for Claude Opus 4.5."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version(
            "anthropic.claude-opus-4-5-20250514-v1:0"
        )
        assert version == 4.5

    def test_extract_version_sonnet_4(self):
        """Test version extraction for Claude Sonnet 4."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version(
            "anthropic.claude-sonnet-4-20250514-v1:0"
        )
        assert version == 4.0

    def test_extract_version_legacy_3_5_sonnet(self):
        """Test version extraction for legacy Claude 3.5 Sonnet."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version(
            "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        assert version == 3.5

    def test_extract_version_legacy_3_sonnet(self):
        """Test version extraction for legacy Claude 3 Sonnet."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version(
            "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        assert version == 3.0

    def test_extract_version_haiku_4_5(self):
        """Test version extraction for Claude Haiku 4.5."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version(
            "anthropic.claude-haiku-4-5-20250514-v1:0"
        )
        assert version == 4.5

    def test_extract_version_invalid_format(self):
        """Test version extraction with invalid model format."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version("invalid-model-format")
        assert version is None

    def test_extract_version_future_opus_5(self):
        """Test version extraction for future Claude Opus 5."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        version = filter_obj._extract_model_version(
            "anthropic.claude-opus-5-20270101-v1:0"
        )
        assert version == 5.0


class TestModelFamilyExtraction:
    """Test model family extraction logic."""

    def test_extract_family_opus(self):
        """Test family extraction for Opus models."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        family = filter_obj._extract_model_family(
            "anthropic.claude-opus-4-5-20250514-v1:0"
        )
        assert family == "opus"

    def test_extract_family_sonnet(self):
        """Test family extraction for Sonnet models."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        family = filter_obj._extract_model_family(
            "anthropic.claude-sonnet-4-20250514-v1:0"
        )
        assert family == "sonnet"

    def test_extract_family_haiku(self):
        """Test family extraction for Haiku models."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        family = filter_obj._extract_model_family(
            "anthropic.claude-haiku-4-5-20250514-v1:0"
        )
        assert family == "haiku"

    def test_extract_family_legacy_sonnet(self):
        """Test family extraction for legacy Sonnet naming."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        family = filter_obj._extract_model_family(
            "anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        assert family == "sonnet"

    def test_extract_family_invalid(self):
        """Test family extraction with invalid model format."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        family = filter_obj._extract_model_family("invalid-model-format")
        assert family is None


class TestVersionBasedFiltering:
    """Test version-based filtering for beta headers."""

    def test_thinking_headers_require_claude_4(self):
        """Test that thinking headers require Claude 4.0+."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)

        # Claude 4.5 should support thinking
        result = filter_obj.filter_beta_headers(
            ["interleaved-thinking-2025-05-14"],
            "anthropic.claude-opus-4-5-20250514-v1:0",
            translate=False,
        )
        assert "interleaved-thinking-2025-05-14" in result

        # Claude 3.5 should NOT support thinking
        result = filter_obj.filter_beta_headers(
            ["interleaved-thinking-2025-05-14"],
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            translate=False,
        )
        assert "interleaved-thinking-2025-05-14" not in result

    def test_context_management_requires_claude_4_5(self):
        """Test that context management requires Claude 4.5+."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)

        # Claude 4.5 should support context management
        result = filter_obj.filter_beta_headers(
            ["context-management-2025-06-27"],
            "anthropic.claude-sonnet-4-5-20250514-v1:0",
            translate=False,
        )
        assert "context-management-2025-06-27" in result

        # Claude 4.0 should NOT support context management
        result = filter_obj.filter_beta_headers(
            ["context-management-2025-06-27"],
            "anthropic.claude-sonnet-4-20250514-v1:0",
            translate=False,
        )
        assert "context-management-2025-06-27" not in result

    def test_computer_use_works_on_all_versions(self):
        """Test that computer-use works on all Claude versions (no version requirement)."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)

        # Claude 3.5 should support computer use
        result = filter_obj.filter_beta_headers(
            ["computer-use-2025-01-24"],
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            translate=False,
        )
        assert "computer-use-2025-01-24" in result

        # Claude 4.5 should also support computer use
        result = filter_obj.filter_beta_headers(
            ["computer-use-2025-01-24"],
            "anthropic.claude-opus-4-5-20250514-v1:0",
            translate=False,
        )
        assert "computer-use-2025-01-24" in result

    def test_future_model_supports_all_headers(self):
        """Test that future Claude 5.0 automatically supports all 4.0+ headers."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-5-20270101-v1:0"

        # Claude 5 should support all headers requiring 4.0+
        result = filter_obj.filter_beta_headers(
            [
                "interleaved-thinking-2025-05-14",  # Requires 4.0+
                "context-management-2025-06-27",  # Requires 4.5+
                "context-1m-2025-08-07",  # Requires 4.0+
            ],
            model,
            translate=False,
        )
        assert len(result) == 3
        assert "interleaved-thinking-2025-05-14" in result
        assert "context-management-2025-06-27" in result
        assert "context-1m-2025-08-07" in result


class TestFamilyRestrictions:
    """Test model family restrictions for beta headers."""

    def test_effort_only_on_opus(self):
        """Test that effort parameter only works on Opus 4.5+."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)

        # Opus 4.5 should support effort
        result = filter_obj.filter_beta_headers(
            ["effort-2025-11-24"],
            "anthropic.claude-opus-4-5-20250514-v1:0",
            translate=False,
        )
        assert "effort-2025-11-24" in result

        # Sonnet 4.5 should NOT support effort (wrong family)
        result = filter_obj.filter_beta_headers(
            ["effort-2025-11-24"],
            "anthropic.claude-sonnet-4-5-20250514-v1:0",
            translate=False,
        )
        assert "effort-2025-11-24" not in result

        # Haiku 4.5 should NOT support effort (wrong family)
        result = filter_obj.filter_beta_headers(
            ["effort-2025-11-24"],
            "anthropic.claude-haiku-4-5-20250514-v1:0",
            translate=False,
        )
        assert "effort-2025-11-24" not in result

    def test_tool_search_on_opus_and_sonnet(self):
        """Test that tool search works on Opus and Sonnet 4.5+, but not Haiku."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)

        # Opus 4.5 should support tool search
        result = filter_obj.filter_beta_headers(
            ["tool-search-tool-2025-10-19"],
            "anthropic.claude-opus-4-5-20250514-v1:0",
            translate=False,
        )
        assert "tool-search-tool-2025-10-19" in result

        # Sonnet 4.5 should support tool search
        result = filter_obj.filter_beta_headers(
            ["tool-search-tool-2025-10-19"],
            "anthropic.claude-sonnet-4-5-20250514-v1:0",
            translate=False,
        )
        assert "tool-search-tool-2025-10-19" in result

        # Haiku 4.5 should NOT support tool search (wrong family)
        result = filter_obj.filter_beta_headers(
            ["tool-search-tool-2025-10-19"],
            "anthropic.claude-haiku-4-5-20250514-v1:0",
            translate=False,
        )
        assert "tool-search-tool-2025-10-19" not in result


class TestBetaHeaderTranslation:
    """Test beta header translation for backward compatibility."""

    def test_advanced_tool_use_translation_opus_4_5(self):
        """Test advanced-tool-use translates to tool search headers on Opus 4.5."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            ["advanced-tool-use-2025-11-20"], model, translate=True
        )

        # Should translate to tool search headers
        assert "tool-search-tool-2025-10-19" in result
        assert "tool-examples-2025-10-29" in result
        assert "advanced-tool-use-2025-11-20" not in result

    def test_advanced_tool_use_translation_sonnet_4_5(self):
        """Test advanced-tool-use translates on Sonnet 4.5."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        model = "anthropic.claude-sonnet-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            ["advanced-tool-use-2025-11-20"], model, translate=True
        )

        # Should translate to tool search headers
        assert "tool-search-tool-2025-10-19" in result
        assert "tool-examples-2025-10-29" in result

    def test_advanced_tool_use_no_translation_claude_4(self):
        """Test advanced-tool-use does NOT translate on Claude 4.0 (too old)."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        model = "anthropic.claude-opus-4-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            ["advanced-tool-use-2025-11-20"], model, translate=True
        )

        # Should not translate (version too old)
        assert result == []

    def test_advanced_tool_use_no_translation_haiku(self):
        """Test advanced-tool-use does NOT translate on Haiku (wrong family)."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        model = "anthropic.claude-haiku-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            ["advanced-tool-use-2025-11-20"], model, translate=True
        )

        # Should not translate (wrong family)
        assert result == []

    def test_translation_disabled(self):
        """Test that translation can be disabled."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            ["advanced-tool-use-2025-11-20"], model, translate=False
        )

        # Should not translate when disabled
        # advanced-tool-use is not in whitelist, so should be filtered out
        assert result == []


class TestCrossAPIConsistency:
    """Test that filtering is consistent across all three APIs."""

    def test_same_headers_work_on_all_apis(self):
        """Test that supported headers work consistently across all APIs."""
        model = "anthropic.claude-opus-4-5-20250514-v1:0"
        headers = ["computer-use-2025-01-24", "effort-2025-11-24"]

        filter_chat = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        filter_messages = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        filter_converse = get_bedrock_beta_filter(BedrockAPI.CONVERSE)

        result_chat = set(filter_chat.filter_beta_headers(headers, model, translate=False))
        result_messages = set(
            filter_messages.filter_beta_headers(headers, model, translate=False)
        )
        result_converse = set(
            filter_converse.filter_beta_headers(headers, model, translate=False)
        )

        # All APIs should return the same results
        assert result_chat == result_messages == result_converse

    def test_unsupported_headers_filtered_on_all_apis(self):
        """Test that unsupported headers are filtered consistently."""
        model = "anthropic.claude-opus-4-5-20250514-v1:0"
        headers = ["unknown-beta-2099-01-01"]

        filter_chat = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        filter_messages = get_bedrock_beta_filter(BedrockAPI.INVOKE_MESSAGES)
        filter_converse = get_bedrock_beta_filter(BedrockAPI.CONVERSE)

        result_chat = filter_chat.filter_beta_headers(headers, model)
        result_messages = filter_messages.filter_beta_headers(headers, model)
        result_converse = filter_converse.filter_beta_headers(headers, model)

        # All should filter out unsupported headers
        assert result_chat == result_messages == result_converse == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_model_version_blocks_versioned_headers(self):
        """Test that unparseable model version blocks headers with version requirements."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "invalid-model-format"

        # Headers with version requirements should be blocked
        result = filter_obj.filter_beta_headers(
            ["interleaved-thinking-2025-05-14"], model, translate=False
        )
        assert result == []

        # Headers without version requirements should still work
        result = filter_obj.filter_beta_headers(
            ["computer-use-2025-01-24"], model, translate=False
        )
        assert "computer-use-2025-01-24" in result

    def test_duplicate_headers_deduplicated(self):
        """Test that duplicate headers are deduplicated."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            [
                "computer-use-2025-01-24",
                "computer-use-2025-01-24",
                "computer-use-2025-01-24",
            ],
            model,
            translate=False,
        )
        assert result == ["computer-use-2025-01-24"]

    def test_output_is_sorted(self):
        """Test that output is sorted for deterministic results."""
        filter_obj = get_bedrock_beta_filter(BedrockAPI.INVOKE_CHAT)
        model = "anthropic.claude-opus-4-5-20250514-v1:0"

        result = filter_obj.filter_beta_headers(
            ["effort-2025-11-24", "computer-use-2025-01-24", "context-1m-2025-08-07"],
            model,
            translate=False,
        )
        # Should be alphabetically sorted
        assert result == sorted(result)
