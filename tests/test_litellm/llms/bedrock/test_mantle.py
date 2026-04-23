"""
Unit tests for the Bedrock Mantle (Claude Mythos Preview) integration.

Tests cover route detection, URL construction, and config dispatch for both
the /chat/completions and /messages endpoints.
"""

from litellm.llms.bedrock.common_utils import BedrockModelInfo, get_bedrock_chat_config
from litellm.llms.bedrock.chat.mantle.transformation import AmazonMantleConfig
from litellm.llms.bedrock.messages.mantle_transformation import (
    AmazonMantleMessagesConfig,
)


def test_get_bedrock_route_mantle():
    assert (
        BedrockModelInfo.get_bedrock_route("mantle/anthropic.claude-mythos-preview")
        == "mantle"
    )


def test_get_bedrock_route_mantle_does_not_match_other_routes():
    assert (
        BedrockModelInfo.get_bedrock_route("anthropic.claude-3-sonnet-20240229-v1:0")
        != "mantle"
    )
    assert (
        BedrockModelInfo.get_bedrock_route("converse/anthropic.claude-3-sonnet")
        != "mantle"
    )


def test_explicit_mantle_route_flag():
    assert (
        BedrockModelInfo._explicit_mantle_route(
            "mantle/anthropic.claude-mythos-preview"
        )
        is True
    )
    assert BedrockModelInfo._explicit_mantle_route("anthropic.claude-3-sonnet") is False
    assert (
        BedrockModelInfo._explicit_mantle_route("converse/anthropic.claude-3-sonnet")
        is False
    )


def test_mantle_url_construction():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/v1/messages"


def test_mantle_url_construction_different_region():
    config = AmazonMantleConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-west-2"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-west-2.api.aws/v1/messages"


def test_get_bedrock_chat_config_returns_mantle_config():
    config = get_bedrock_chat_config("mantle/anthropic.claude-mythos-preview")
    assert isinstance(config, AmazonMantleConfig)


def test_get_bedrock_provider_config_for_messages_api_mantle():
    config = BedrockModelInfo.get_bedrock_provider_config_for_messages_api(
        "mantle/anthropic.claude-mythos-preview"
    )
    assert isinstance(config, AmazonMantleMessagesConfig)


def test_mantle_messages_url_construction():
    config = AmazonMantleMessagesConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="mantle/anthropic.claude-mythos-preview",
        optional_params={"aws_region_name": "us-east-1"},
        litellm_params={},
    )
    assert url == "https://bedrock-mantle.us-east-1.api.aws/v1/messages"


def test_mantle_transform_request_strips_prefix_and_adds_model():
    config = AmazonMantleConfig()
    request = config.transform_request(
        model="mantle/anthropic.claude-mythos-preview",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "anthropic.claude-mythos-preview"
    assert "mantle/" not in request["model"]
