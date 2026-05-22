"""
Unit tests for Tencent Cloud TokenHub configuration.

These tests validate the TencentCloudChatConfig class which extends
OpenAIGPTConfig. Tencent Cloud TokenHub is an OpenAI-compatible provider.

Docs: https://cloud.tencent.com/document/product/1823/130079
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.tencent_cloud.chat.transformation import TencentCloudChatConfig


class TestTencentCloudChatConfig:
    def test_validate_environment_sets_bearer_auth(self):
        config = TencentCloudChatConfig()
        api_key = "sk-fake-tencent-cloud-key"

        result = config.validate_environment(
            headers={},
            model="tencent_cloud/deepseek-v3.1-terminus",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://tokenhub.tencentmaas.com/v1",
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_missing_api_key_raises(self):
        config = TencentCloudChatConfig()

        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="tencent_cloud/deepseek-v3.1-terminus",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base="https://tokenhub.tencentmaas.com/v1",
            )

        assert "Tencent Cloud API Key" in str(excinfo.value)

    def test_inheritance(self):
        config = TencentCloudChatConfig()
        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")

    def test_default_api_base(self, monkeypatch):
        # Ensure no env override leaks in from the host.
        monkeypatch.delenv("TENCENT_CLOUD_API_BASE", raising=False)
        monkeypatch.delenv("TENCENT_CLOUD_API_KEY", raising=False)

        config = TencentCloudChatConfig()
        api_base, dynamic_api_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )
        assert api_base == "https://tokenhub.tencentmaas.com/v1"
        assert dynamic_api_key is None

    def test_env_overrides_api_base_and_key(self, monkeypatch):
        monkeypatch.setenv(
            "TENCENT_CLOUD_API_BASE", "https://tokenhub-intl.tencentmaas.com/v1"
        )
        monkeypatch.setenv("TENCENT_CLOUD_API_KEY", "env-key")

        config = TencentCloudChatConfig()
        api_base, dynamic_api_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )
        assert api_base == "https://tokenhub-intl.tencentmaas.com/v1"
        assert dynamic_api_key == "env-key"

    def test_explicit_args_take_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv(
            "TENCENT_CLOUD_API_BASE", "https://tokenhub.tencentmaas.com/v1"
        )
        monkeypatch.setenv("TENCENT_CLOUD_API_KEY", "env-key")

        config = TencentCloudChatConfig()
        api_base, dynamic_api_key = config._get_openai_compatible_provider_info(
            api_base="https://custom-proxy.example/v1",
            api_key="explicit-key",
        )
        assert api_base == "https://custom-proxy.example/v1"
        assert dynamic_api_key == "explicit-key"

    def test_transform_request_strips_provider_prefix(self):
        config = TencentCloudChatConfig()
        result = config.transform_request(
            model="tencent_cloud/minimax-m2.7",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["model"] == "minimax-m2.7"

    def test_transform_request_preserves_model_without_prefix(self):
        config = TencentCloudChatConfig()
        result = config.transform_request(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["model"] == "minimax-m2.7"
