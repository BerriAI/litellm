"""
Tests for Bedrock Converse API serviceTier support.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.types.llms.bedrock import ServiceTierBlock


def test_service_tier_block_type():
    """Test that ServiceTierBlock is properly defined."""
    # Test valid service tier values
    priority_tier: ServiceTierBlock = {"type": "priority"}
    default_tier: ServiceTierBlock = {"type": "default"}
    flex_tier: ServiceTierBlock = {"type": "flex"}

    assert priority_tier["type"] == "priority"
    assert default_tier["type"] == "default"
    assert flex_tier["type"] == "flex"


def test_service_tier_in_config_blocks():
    """Test that serviceTier is included in get_config_blocks()."""
    config_blocks = AmazonConverseConfig.get_config_blocks()

    assert "serviceTier" in config_blocks
    assert config_blocks["serviceTier"] == ServiceTierBlock


def test_transform_request_with_service_tier():
    """Test that serviceTier is properly included in the transformed request."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]
    optional_params = {
        "serviceTier": {"type": "priority"},
    }

    result = config.transform_request(
        model="bedrock/converse/qwen.qwen3-235b-a22b-2507-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # serviceTier should be a top-level parameter, not in additionalModelRequestFields
    assert "serviceTier" in result
    assert result["serviceTier"]["type"] == "priority"

    # Verify it's NOT in additionalModelRequestFields
    additional_fields = result.get("additionalModelRequestFields", {})
    assert "serviceTier" not in additional_fields
    assert "service_tier" not in additional_fields


def test_transform_request_with_default_tier():
    """Test serviceTier with default value."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]
    optional_params = {
        "serviceTier": {"type": "default"},
    }

    result = config.transform_request(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "serviceTier" in result
    assert result["serviceTier"]["type"] == "default"


def test_transform_request_with_flex_tier():
    """Test serviceTier with flex value."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]
    optional_params = {
        "serviceTier": {"type": "flex"},
    }

    result = config.transform_request(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "serviceTier" in result
    assert result["serviceTier"]["type"] == "flex"


def test_transform_request_without_service_tier():
    """Test that requests without serviceTier work correctly."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]
    optional_params = {}

    result = config.transform_request(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # serviceTier should not be present if not specified
    assert "serviceTier" not in result


def test_service_tier_with_other_config_blocks():
    """Test serviceTier works alongside other config blocks like performanceConfig."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]
    optional_params = {
        "serviceTier": {"type": "priority"},
        "performanceConfig": {"latency": "optimized"},
    }

    result = config.transform_request(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Both should be top-level parameters
    assert "serviceTier" in result
    assert result["serviceTier"]["type"] == "priority"
    assert "performanceConfig" in result
    assert result["performanceConfig"]["latency"] == "optimized"
