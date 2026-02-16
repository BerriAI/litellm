"""
Test suite for AWS Bedrock extended beta model support
Tests model configuration, pricing, and regional availability for:
- DeepSeek V3.2
- Minimax M2.1
- Moonshot AI Kimi K2.5
- Qwen3 Coder Next
"""

import os

# Set env var to use local model cost map instead of fetching from remote
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

import pytest

from litellm import get_model_info

# Model configurations: (model_name, regions, max_input, max_output)
MODEL_CONFIGS = [
    (
        "deepseek.v3.2",
        [
            "ap-northeast-1",
            "ap-south-1",
            "ap-southeast-3",
            "eu-north-1",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-2",
        ],
        163840,
        163840,
    ),
    (
        "minimax.minimax-m2.1",
        [
            "ap-northeast-1",
            "ap-south-1",
            "ap-southeast-3",
            "eu-central-1",
            "eu-north-1",
            "eu-south-1",
            "eu-west-1",
            "eu-west-2",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-2",
        ],
        196000,
        8192,
    ),
    (
        "moonshotai.kimi-k2.5",
        [
            "ap-northeast-1",
            "ap-south-1",
            "ap-southeast-3",
            "eu-north-1",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-2",
        ],
        262144,
        262144,
    ),
    (
        "qwen.qwen3-coder-next",
        [
            "ap-northeast-1",
            "ap-south-1",
            "ap-southeast-3",
            "eu-central-1",
            "eu-south-1",
            "eu-west-1",
            "eu-west-2",
            "sa-east-1",
            "us-east-1",
            "us-east-2",
            "us-west-2",
        ],
        262144,
        8192,
    ),
]


class TestBedrockNewModels:
    """Unified test suite for all new Bedrock models"""

    @pytest.mark.parametrize("model_name,regions,max_input,max_output", MODEL_CONFIGS)
    def test_model_info_primary_region(
        self, model_name, regions, max_input, max_output
    ):
        """Test model configuration in primary region (us-east-1)"""
        model = f"bedrock/us-east-1/{model_name}"
        model_info = get_model_info(model)

        assert model_info is not None, f"Model {model_name} not found"
        assert model_info["max_input_tokens"] == max_input
        assert model_info["max_output_tokens"] == max_output
        assert model_info["litellm_provider"] == "bedrock"
        assert model_info["mode"] == "chat"
        assert model_info["supports_function_calling"] is True

    @pytest.mark.parametrize("model_name,regions,max_input,max_output", MODEL_CONFIGS)
    def test_pricing_configured(self, model_name, regions, max_input, max_output):
        """Verify pricing is set for all models"""
        model = f"bedrock/us-east-1/{model_name}"
        model_info = get_model_info(model)

        assert (
            model_info["input_cost_per_token"] > 0
        ), f"Missing input cost for {model_name}"
        assert (
            model_info["output_cost_per_token"] > 0
        ), f"Missing output cost for {model_name}"

    @pytest.mark.parametrize("model_name,regions,max_input,max_output", MODEL_CONFIGS)
    def test_region_count(self, model_name, regions, max_input, max_output):
        """Verify each bedrock/{region}/{model_name} resolves via get_model_info"""
        for region in regions:
            model = f"bedrock/{region}/{model_name}"
            model_info = get_model_info(model)
            assert model_info is not None, f"Model {model_name} not found in {region}"
            assert model_info["max_input_tokens"] == max_input
            assert model_info["max_output_tokens"] == max_output

    @pytest.mark.parametrize("model_name,regions,max_input,max_output", MODEL_CONFIGS)
    def test_sample_regional_variants(self, model_name, regions, max_input, max_output):
        """Test sample regional variants (us-east-1, eu-west-1, ap-northeast-1)"""
        for region in ["us-east-1", "ap-northeast-1"]:
            if region in regions:
                model = f"bedrock/{region}/{model_name}"
                model_info = get_model_info(model)
                assert (
                    model_info is not None
                ), f"Model {model_name} not found in {region}"
                assert model_info["max_input_tokens"] == max_input
                assert model_info["litellm_provider"] == "bedrock"


class TestModelSpecificFeatures:
    """Model-specific capability tests"""

    def test_deepseek_v3_2_context_window(self):
        """DeepSeek V3.2 has 163K context window"""
        model_info = get_model_info("bedrock/us-east-1/deepseek.v3.2")
        assert model_info["max_input_tokens"] == 163840

    def test_minimax_m2_1_context_window(self):
        """Minimax M2.1 has 196K input, 8K output"""
        model_info = get_model_info("bedrock/us-east-1/minimax.minimax-m2.1")
        assert model_info["max_input_tokens"] == 196000
        assert model_info["max_output_tokens"] == 8192

    def test_moonshotai_kimi_k2_5_context_window(self):
        """Moonshot AI Kimi K2.5 has 256K context window"""
        model_info = get_model_info("bedrock/us-east-1/moonshotai.kimi-k2.5")
        assert model_info["max_input_tokens"] == 262144
        assert model_info["max_output_tokens"] == 262144

    def test_qwen3_coder_next_context_window(self):
        """Qwen3 Coder Next has 256K input, 8K output"""
        model_info = get_model_info("bedrock/us-east-1/qwen.qwen3-coder-next")
        assert model_info["max_input_tokens"] == 262144
        assert model_info["max_output_tokens"] == 8192
