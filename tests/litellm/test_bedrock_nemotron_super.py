"""
Test suite for NVIDIA Nemotron Super 3 120B on AWS Bedrock
Verifies model configuration, pricing, and regional availability.
"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

import pytest

from litellm import get_model_info


MODEL_NAME = "nvidia.nemotron-super-3-120b"


class TestNemotronSuper3120B:
    """Test model definition for nvidia.nemotron-super-3-120b"""

    def test_model_info_primary_region(self):
        """Test model resolves in us-east-1"""
        model_info = get_model_info(f"bedrock/us-east-1/{MODEL_NAME}")

        assert model_info is not None, f"Model {MODEL_NAME} not found"
        assert model_info["max_input_tokens"] == 256000
        assert model_info["max_output_tokens"] == 32000
        assert model_info["litellm_provider"] == "bedrock_converse"
        assert model_info["mode"] == "chat"
        assert model_info["supports_function_calling"] is True

    def test_pricing_configured(self):
        """Verify pricing matches AWS Bedrock rates"""
        model_info = get_model_info(f"bedrock/us-east-1/{MODEL_NAME}")

        assert model_info["input_cost_per_token"] == 1.5e-07
        assert model_info["output_cost_per_token"] == 6.5e-07

    def test_context_window(self):
        """Nemotron Super 3 120B has 256K input, 32K output on Bedrock"""
        model_info = get_model_info(f"bedrock/us-east-1/{MODEL_NAME}")

        assert model_info["max_input_tokens"] == 256000
        assert model_info["max_output_tokens"] == 32000

    def test_resolves_without_region(self):
        """Test model resolves with just bedrock/ prefix"""
        model_info = get_model_info(f"bedrock/{MODEL_NAME}")

        assert model_info is not None, f"Model {MODEL_NAME} not found without region"
        assert model_info["max_input_tokens"] == 256000
