"""
Unit tests for litellm/llms/bedrock/common_utils.py

Tests the standalone model name utility functions and BedrockTokenCounter.
"""

import pytest

from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo,
    extract_model_name_from_bedrock_arn,
    get_bedrock_base_model,
    get_bedrock_cross_region_inference_regions,
    strip_bedrock_routing_prefix,
    strip_bedrock_throughput_suffix,
)
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter


class TestStripBedrockRoutingPrefix:
    """Tests for strip_bedrock_routing_prefix function."""

    def test_strips_bedrock_prefix(self):
        assert strip_bedrock_routing_prefix("bedrock/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_converse_prefix(self):
        assert strip_bedrock_routing_prefix("converse/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_invoke_prefix(self):
        assert strip_bedrock_routing_prefix("invoke/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_openai_prefix(self):
        assert strip_bedrock_routing_prefix("openai/gpt-4") == "gpt-4"

    def test_strips_all_known_prefixes(self):
        # Function strips all known prefixes iteratively
        # bedrock/converse/model -> converse/model -> model
        assert strip_bedrock_routing_prefix("bedrock/converse/claude-3") == "claude-3"

    def test_no_prefix_unchanged(self):
        assert strip_bedrock_routing_prefix("claude-3-sonnet") == "claude-3-sonnet"

    def test_model_with_dots_unchanged(self):
        assert (
            strip_bedrock_routing_prefix("anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )


class TestStripBedrockThroughputSuffix:
    """Tests for strip_bedrock_throughput_suffix function."""

    @pytest.mark.parametrize("input_model,expected", [
        ("anthropic.claude-3-5-sonnet-20241022-v2:0:51k", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0:18k", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("model:1:51k", "model:1"),
        ("model:123:18k", "model:123"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("anthropic.claude-3-sonnet", "anthropic.claude-3-sonnet"),
    ])
    def test_strip_throughput_suffix(self, input_model, expected):
        assert strip_bedrock_throughput_suffix(input_model) == expected


class TestExtractModelNameFromBedrockArn:
    """Tests for extract_model_name_from_bedrock_arn function."""

    def test_extracts_from_provisioned_model_arn(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model-id"
        assert extract_model_name_from_bedrock_arn(arn) == "my-model-id"

    def test_extracts_from_foundation_model_arn(self):
        arn = "arn:aws:bedrock:us-west-2:123456789012:foundation-model/anthropic.claude-v2"
        assert extract_model_name_from_bedrock_arn(arn) == "anthropic.claude-v2"

    def test_non_arn_unchanged(self):
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
        assert extract_model_name_from_bedrock_arn(model) == model

    def test_case_insensitive_arn_detection(self):
        arn = "ARN:aws:bedrock:us-east-1:123456789012:model/my-model"
        assert extract_model_name_from_bedrock_arn(arn) == "my-model"


class TestGetBedrockCrossRegionInferenceRegions:
    """Tests for get_bedrock_cross_region_inference_regions function."""

    def test_returns_expected_regions(self):
        regions = get_bedrock_cross_region_inference_regions()
        assert "us" in regions
        assert "eu" in regions
        assert "global" in regions
        assert "apac" in regions

    def test_returns_list(self):
        regions = get_bedrock_cross_region_inference_regions()
        assert isinstance(regions, list)


class TestGetBedrockBaseModel:
    """Tests for get_bedrock_base_model function."""

    def test_strips_bedrock_prefix(self):
        assert get_bedrock_base_model("bedrock/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_converse_prefix(self):
        assert get_bedrock_base_model("bedrock/converse/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_us_region_prefix(self):
        # us.anthropic.model -> anthropic.model
        assert (
            get_bedrock_base_model("us.anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

    def test_strips_eu_region_prefix(self):
        assert (
            get_bedrock_base_model("eu.anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

    def test_extracts_from_arn(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model"
        assert get_bedrock_base_model(arn) == "my-model"

    def test_model_without_prefix_unchanged(self):
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
        assert get_bedrock_base_model(model) == model

    def test_combined_bedrock_and_region_prefix(self):
        # bedrock/us.anthropic.model -> anthropic.model
        assert (
            get_bedrock_base_model("bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

    @pytest.mark.parametrize("input_model,expected", [
        ("anthropic.claude-3-5-sonnet-20241022-v2:0:51k", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0:18k", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0:51k", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0:51k", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ])
    def test_strips_throughput_suffix(self, input_model, expected):
        """Test that throughput tier suffixes like :51k are stripped. Issue #19113."""
        assert get_bedrock_base_model(input_model) == expected


class TestBedrockModelInfoWrappers:
    """Tests that BedrockModelInfo methods correctly wrap standalone functions."""

    def test_get_base_model_matches_standalone(self):
        test_cases = [
            "bedrock/claude-3-sonnet",
            "us.anthropic.claude-3-sonnet-20240229-v1:0",
            "arn:aws:bedrock:us-east-1:123:model/my-model",
        ]
        for model in test_cases:
            assert BedrockModelInfo.get_base_model(model) == get_bedrock_base_model(model)

    def test_extract_model_name_from_arn_matches_standalone(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model"
        assert (
            BedrockModelInfo.extract_model_name_from_arn(arn)
            == extract_model_name_from_bedrock_arn(arn)
        )

    def test_get_non_litellm_routing_model_name_matches_standalone(self):
        model = "bedrock/converse/claude-3"
        assert (
            BedrockModelInfo.get_non_litellm_routing_model_name(model)
            == strip_bedrock_routing_prefix(model)
        )


class TestBedrockTokenCounter:
    """Tests for BedrockTokenCounter class."""

    def test_should_use_token_counting_api_for_bedrock(self):
        counter = BedrockTokenCounter()
        assert counter.should_use_token_counting_api("bedrock") is True

    def test_should_not_use_token_counting_api_for_other_providers(self):
        counter = BedrockTokenCounter()
        assert counter.should_use_token_counting_api("openai") is False
        assert counter.should_use_token_counting_api("anthropic") is False
        assert counter.should_use_token_counting_api(None) is False

    def test_get_token_counter_returns_bedrock_token_counter(self):
        model_info = BedrockModelInfo()
        token_counter = model_info.get_token_counter()
        assert isinstance(token_counter, BedrockTokenCounter)

    @pytest.mark.asyncio
    async def test_count_tokens_returns_none_for_empty_messages(self):
        counter = BedrockTokenCounter()
        result = await counter.count_tokens(
            model_to_use="anthropic.claude-3-sonnet",
            messages=None,
            contents=None,
        )
        assert result is None

        result = await counter.count_tokens(
            model_to_use="anthropic.claude-3-sonnet",
            messages=[],
            contents=None,
        )
        assert result is None
