"""
Tests for Nova imported/custom model support via spec prefixes (nova/, nova-2/).
"""

import pytest

from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo,
    get_bedrock_base_model,
    strip_bedrock_routing_prefix,
)
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


NOVA_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/a1b2c3d4e5f6"
NOVA_MODEL = f"bedrock/nova/{NOVA_ARN}"
NOVA2_MODEL = f"bedrock/nova-2/{NOVA_ARN}"


class TestGetBedrockRoute:
    def test_nova_prefix_routes_to_converse(self):
        assert BedrockModelInfo.get_bedrock_route(NOVA_MODEL) == "converse"

    def test_nova2_prefix_routes_to_converse(self):
        assert BedrockModelInfo.get_bedrock_route(NOVA2_MODEL) == "converse"

    def test_plain_arn_routes_to_invoke(self):
        # Without spec prefix, ARN doesn't match converse models
        result = BedrockModelInfo.get_bedrock_route(f"bedrock/{NOVA_ARN}")
        assert result == "invoke"


class TestGetBedrockBaseModel:
    def test_nova_prefix_returns_sentinel(self):
        assert get_bedrock_base_model(f"nova/{NOVA_ARN}") == "amazon.nova-custom"

    def test_nova2_prefix_returns_sentinel(self):
        assert get_bedrock_base_model(f"nova-2/{NOVA_ARN}") == "amazon.nova-2-custom"

    def test_bedrock_nova_prefix_returns_sentinel(self):
        assert get_bedrock_base_model(NOVA_MODEL) == "amazon.nova-custom"

    def test_bedrock_nova2_prefix_returns_sentinel(self):
        assert get_bedrock_base_model(NOVA2_MODEL) == "amazon.nova-2-custom"


class TestStripBedrockRoutingPrefix:
    def test_strips_nova_prefix(self):
        result = strip_bedrock_routing_prefix(f"nova/{NOVA_ARN}")
        assert result == NOVA_ARN

    def test_strips_nova2_prefix(self):
        result = strip_bedrock_routing_prefix(f"nova-2/{NOVA_ARN}")
        assert result == NOVA_ARN


class TestIsNova2Model:
    def setup_method(self):
        self.config = AmazonConverseConfig()

    def test_standard_nova2_model(self):
        assert self.config._is_nova_2_model("amazon.nova-2-lite-v1:0") is True

    def test_nova2_imported_model(self):
        assert self.config._is_nova_2_model(NOVA2_MODEL) is True

    def test_nova_imported_model_is_not_nova2(self):
        assert self.config._is_nova_2_model(NOVA_MODEL) is False

    def test_plain_nova_model(self):
        assert self.config._is_nova_2_model("amazon.nova-pro-v1:0") is False


class TestGetSupportedOpenaiParams:
    def setup_method(self):
        self.config = AmazonConverseConfig()

    def test_nova_imported_has_tools_and_web_search(self):
        params = self.config.get_supported_openai_params(NOVA_MODEL)
        assert "tools" in params
        assert "tool_choice" in params
        assert "web_search_options" in params

    def test_nova2_imported_has_reasoning_effort(self):
        params = self.config.get_supported_openai_params(NOVA2_MODEL)
        assert "reasoning_effort" in params
        assert "web_search_options" in params

    def test_nova2_imported_has_tools(self):
        params = self.config.get_supported_openai_params(NOVA2_MODEL)
        assert "tools" in params
        assert "tool_choice" in params
