"""
Tests for SaladCloud AI Gateway provider configuration.
"""

import os
import sys

try:
    import pytest
except ImportError:
    pytest = None

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestSaladCloudProviderConfig:
    """Test SaladCloud provider configuration"""

    def test_salad_cloud_in_provider_list(self):
        """Test that salad_cloud is in the provider list via LlmProviders enum"""
        from litellm import LlmProviders

        assert LlmProviders.SALAD_CLOUD.value == "salad_cloud"
        assert "salad_cloud" in litellm.provider_list

    def test_salad_cloud_json_config_exists(self):
        """Test that salad_cloud is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("salad_cloud")

        salad_cloud = JSONProviderRegistry.get("salad_cloud")
        assert salad_cloud is not None
        assert salad_cloud.base_url == "https://ai.salad.cloud/v1"
        assert salad_cloud.api_key_env == "SALAD_API_KEY"

    def test_salad_cloud_provider_resolution(self):
        """Test that provider resolution correctly routes salad_cloud models"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="salad_cloud/qwen3.5-35b-a3b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.5-35b-a3b"
        assert provider == "salad_cloud"
        assert api_base == "https://ai.salad.cloud/v1"

    def test_salad_cloud_router_config(self):
        """Test that salad_cloud can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "qwen3.5-35b",
                    "litellm_params": {
                        "model": "salad_cloud/qwen3.5-35b-a3b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "qwen3.5-35b"

    def test_salad_cloud_models_in_pricing(self):
        """Test that all SaladCloud models are present in model_prices_and_context_window.json"""
        import json
        import pathlib

        pricing_file = (
            pathlib.Path(__file__).parent.parent.parent.parent.parent
            / "model_prices_and_context_window.json"
        )
        with open(pricing_file) as f:
            model_cost = json.load(f)

        expected_models = [
            "salad_cloud/qwen3.5-35b-a3b",
            "salad_cloud/qwen3.5-27b",
            "salad_cloud/qwen3.5-9b",
        ]

        for model in expected_models:
            assert (
                model in model_cost
            ), f"{model} not found in model_prices_and_context_window.json"
            entry = model_cost[model]
            assert entry["litellm_provider"] == "salad_cloud"
            assert entry["mode"] == "chat"


if __name__ == "__main__":
    print("Testing SaladCloud Provider...")

    test_config = TestSaladCloudProviderConfig()

    print("\n1. Testing JSON config...")
    test_config.test_salad_cloud_json_config_exists()
    print("   ✓ salad_cloud JSON config loaded")

    print("\n2. Testing provider resolution...")
    test_config.test_salad_cloud_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n3. Testing router configuration...")
    test_config.test_salad_cloud_router_config()
    print("   ✓ Router configuration works")

    print("\n4. Testing models in pricing data...")
    test_config.test_salad_cloud_models_in_pricing()
    print("   ✓ All models present in model_cost")

    print("\n" + "=" * 50)
    print("✓ All configuration tests passed!")
    print("=" * 50)
