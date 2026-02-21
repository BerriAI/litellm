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


# Tests for OpenAI-compatible service_tier parameter translation


def test_service_tier_in_supported_openai_params():
    """Test that service_tier is in the list of supported OpenAI params."""
    config = AmazonConverseConfig()
    supported_params = config.get_supported_openai_params(
        model="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    assert "service_tier" in supported_params


def test_map_openai_service_tier_priority():
    """Test that OpenAI service_tier='priority' maps to Bedrock serviceTier."""
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"service_tier": "priority"},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    )

    assert "serviceTier" in result
    assert result["serviceTier"] == {"type": "priority"}


def test_map_openai_service_tier_default():
    """Test that OpenAI service_tier='default' maps to Bedrock serviceTier."""
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"service_tier": "default"},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    )

    assert "serviceTier" in result
    assert result["serviceTier"] == {"type": "default"}


def test_map_openai_service_tier_flex():
    """Test that OpenAI service_tier='flex' maps to Bedrock serviceTier."""
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"service_tier": "flex"},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    )

    assert "serviceTier" in result
    assert result["serviceTier"] == {"type": "flex"}


def test_map_openai_service_tier_auto_maps_to_default():
    """Test that OpenAI service_tier='auto' maps to Bedrock serviceTier='default'.

    Bedrock doesn't support 'auto', so we map it to 'default'.
    """
    config = AmazonConverseConfig()

    result = config.map_openai_params(
        non_default_params={"service_tier": "auto"},
        optional_params={},
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        drop_params=False,
    )

    assert "serviceTier" in result
    assert result["serviceTier"] == {"type": "default"}


# Tests for service_tier in response


def test_transform_response_with_service_tier():
    """Test that serviceTier from Bedrock response is mapped to service_tier in OpenAI format."""
    from unittest.mock import Mock

    import httpx

    from litellm.types.utils import ModelResponse

    config = AmazonConverseConfig()

    # Mock Bedrock response with serviceTier
    mock_response_data = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello! How can I assist you today?"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "totalTokens": 30,
        },
        "serviceTier": {"type": "priority"},  # This should be mapped to service_tier
    }

    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = mock_response_data
    mock_response.text = json.dumps(mock_response_data)

    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello"}]

    result = config.transform_response(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=None,
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # Verify service_tier is present in the response
    assert hasattr(result, "service_tier")
    assert result.service_tier == "priority"


def test_transform_response_with_service_tier_default():
    """Test that serviceTier='default' is correctly mapped."""
    from unittest.mock import Mock

    import httpx

    from litellm.types.utils import ModelResponse

    config = AmazonConverseConfig()

    mock_response_data = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Response text"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "totalTokens": 30,
        },
        "serviceTier": {"type": "default"},
    }

    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = mock_response_data
    mock_response.text = json.dumps(mock_response_data)

    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello"}]

    result = config.transform_response(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=None,
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert hasattr(result, "service_tier")
    assert result.service_tier == "default"


def test_transform_response_with_service_tier_flex():
    """Test that serviceTier='flex' is correctly mapped."""
    from unittest.mock import Mock

    import httpx

    from litellm.types.utils import ModelResponse

    config = AmazonConverseConfig()

    mock_response_data = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Response text"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "totalTokens": 30,
        },
        "serviceTier": {"type": "flex"},
    }

    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = mock_response_data
    mock_response.text = json.dumps(mock_response_data)

    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello"}]

    result = config.transform_response(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=None,
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert hasattr(result, "service_tier")
    assert result.service_tier == "flex"


def test_transform_response_without_service_tier():
    """Test that responses without serviceTier don't have service_tier attribute."""
    from unittest.mock import Mock

    import httpx

    from litellm.types.utils import ModelResponse

    config = AmazonConverseConfig()

    # Mock Bedrock response WITHOUT serviceTier
    mock_response_data = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello! How can I assist you today?"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "totalTokens": 30,
        },
        # No serviceTier field
    }

    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = mock_response_data
    mock_response.text = json.dumps(mock_response_data)

    model_response = ModelResponse()
    messages = [{"role": "user", "content": "Hello"}]

    result = config.transform_response(
        model="bedrock/converse/anthropic.claude-3-sonnet-20240229-v1:0",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=None,
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    # service_tier should not be present if not in Bedrock response
    assert not hasattr(result, "service_tier")
