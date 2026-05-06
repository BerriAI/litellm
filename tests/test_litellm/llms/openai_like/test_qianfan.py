import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import Router
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_qianfan_in_provider_list():
    assert "qianfan" in litellm.provider_list
    assert "qianfan" in litellm.openai_compatible_providers
    assert LlmProviders.QIANFAN == "qianfan"


def test_qianfan_json_config_exists():
    config = JSONProviderRegistry.get("qianfan")

    assert config is not None
    assert config.base_url == "https://qianfan.baidubce.com/v2"
    assert config.api_key_env == "QIANFAN_API_KEY"
    assert config.api_base_env == "QIANFAN_API_BASE"
    assert config.param_mappings["max_completion_tokens"] == "max_tokens"


def test_qianfan_dynamic_config_defaults():
    provider = JSONProviderRegistry.get("qianfan")
    assert provider is not None

    config = create_config_class(provider)()
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )

    assert api_base == "https://qianfan.baidubce.com/v2"
    assert api_key is None
    assert config.custom_llm_provider == "qianfan"


def test_qianfan_provider_detection_by_prefix():
    model, provider, _, api_base = litellm.get_llm_provider(
        model="qianfan/ernie-4.5-turbo-128k"
    )

    assert provider == "qianfan"
    assert model == "ernie-4.5-turbo-128k"
    assert api_base == "https://qianfan.baidubce.com/v2"


def test_qianfan_provider_detection_by_api_base_preserves_explicit_key():
    model, provider, api_key, api_base = litellm.get_llm_provider(
        model="ernie-4.5-turbo-128k",
        api_base="https://qianfan.baidubce.com/v2",
        api_key="test-qianfan-key",
    )

    assert model == "ernie-4.5-turbo-128k"
    assert provider == "qianfan"
    assert api_key == "test-qianfan-key"
    assert api_base == "https://qianfan.baidubce.com/v2"


def test_qianfan_provider_detection_by_api_base_prefers_explicit_key_over_env():
    with patch.dict(os.environ, {"QIANFAN_API_KEY": "env-qianfan-key"}):
        model, provider, api_key, api_base = litellm.get_llm_provider(
            model="ernie-4.5-turbo-128k",
            api_base="https://qianfan.baidubce.com/v2",
            api_key="explicit-qianfan-key",
        )

    assert model == "ernie-4.5-turbo-128k"
    assert provider == "qianfan"
    assert api_key == "explicit-qianfan-key"
    assert api_base == "https://qianfan.baidubce.com/v2"


def test_qianfan_router_config():
    router = Router(
        model_list=[
            {
                "model_name": "qianfan-router-model",
                "litellm_params": {
                    "model": "qianfan/ernie-4.5-turbo-128k",
                    "api_key": "test-key",
                },
            }
        ]
    )

    assert len(router.model_list) == 1
    assert router.model_list[0]["model_name"] == "qianfan-router-model"
    assert router.deployment_names == ["qianfan/ernie-4.5-turbo-128k"]


def test_qianfan_provider_config_manager():
    config = ProviderConfigManager.get_provider_chat_config(
        model="ernie-4.5-turbo-128k",
        provider=LlmProviders.QIANFAN,
    )

    assert config is not None
    assert config.custom_llm_provider == "qianfan"


def test_qianfan_provider_config_manager_maps_max_completion_tokens():
    config = ProviderConfigManager.get_provider_chat_config(
        model="ernie-4.5-turbo-128k",
        provider=LlmProviders.QIANFAN,
    )

    mapped_params = config.map_openai_params(
        non_default_params={
            "max_completion_tokens": 123,
            "temperature": 0.7,
        },
        optional_params={},
        model="ernie-4.5-turbo-128k",
        drop_params=False,
    )

    assert mapped_params["max_tokens"] == 123
    assert "max_completion_tokens" not in mapped_params
    assert mapped_params["temperature"] == 0.7
