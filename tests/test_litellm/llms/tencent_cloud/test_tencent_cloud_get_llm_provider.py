"""
End-to-end provider routing tests for Tencent Cloud TokenHub.

Validates that:
- ``model="tencent_cloud/<model_id>"`` resolves to the right provider, api_base, api_key
- A bare api_base pointing at tokenhub.tencentmaas.com auto-detects the provider
- The TENCENT_CLOUD_API_KEY env var is picked up
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm


class TestTencentCloudProviderRouting:
    def test_provider_prefix_routing(self, monkeypatch):
        monkeypatch.delenv("TENCENT_CLOUD_API_BASE", raising=False)
        monkeypatch.setenv("TENCENT_CLOUD_API_KEY", "env-key")

        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider(model="tencent_cloud/deepseek-v3.1-terminus")
        )

        assert custom_llm_provider == "tencent_cloud"
        assert model == "deepseek-v3.1-terminus"
        assert api_base == "https://tokenhub.tencentmaas.com/v1"
        assert dynamic_api_key == "env-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TENCENT_CLOUD_API_KEY", "env-key")

        _, _, dynamic_api_key, _ = litellm.get_llm_provider(
            model="tencent_cloud/glm-5",
            api_key="explicit-key",
        )

        assert dynamic_api_key == "explicit-key"

    def test_intl_endpoint_autodetection(self, monkeypatch):
        monkeypatch.setenv("TENCENT_CLOUD_API_KEY", "env-key")

        _, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
            model="kimi-k2.5",
            api_base="https://tokenhub-intl.tencentmaas.com/v1",
        )

        assert custom_llm_provider == "tencent_cloud"
        assert api_base == "https://tokenhub-intl.tencentmaas.com/v1"
        assert dynamic_api_key == "env-key"

    def test_provider_listed_in_openai_compatible_providers(self):
        assert "tencent_cloud" in litellm.openai_compatible_providers

    def test_endpoint_listed_in_openai_compatible_endpoints(self):
        assert (
            "https://tokenhub.tencentmaas.com/v1" in litellm.openai_compatible_endpoints
        )
        assert (
            "https://tokenhub-intl.tencentmaas.com/v1"
            in litellm.openai_compatible_endpoints
        )

    @pytest.mark.parametrize(
        "model_key",
        [
            "tencent_cloud/deepseek-v3.1-terminus",
            "tencent_cloud/hunyuan-2.0-instruct-20251111",
            "tencent_cloud/glm-5",
            "tencent_cloud/kimi-k2.5",
            "tencent_cloud/minimax-m2.7",
        ],
    )
    def test_model_metadata_in_cost_map(self, model_key):
        if model_key not in litellm.model_cost:
            pytest.skip(
                "model cost map loaded from remote — entry only present in local backup"
            )
        info = litellm.model_cost[model_key]
        assert info["litellm_provider"] == "tencent_cloud"
        assert info["mode"] == "chat"
