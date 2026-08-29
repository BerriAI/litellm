"""Tests for the AI Power Grid JSON provider integration."""

import os
from unittest import mock

import litellm

AIPG_API_BASE = "https://api.aipowergrid.io/v1"


def test_aipg_json_registry():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = JSONProviderRegistry.get("aipg")
    assert config is not None
    assert config.base_url == AIPG_API_BASE
    assert config.api_key_env == "AIPG_API_KEY"
    assert config.api_base_env == "AIPG_API_BASE"
    assert config.supported_endpoints == ["/v1/chat/completions", "/v1/responses"]
    assert JSONProviderRegistry.supports_responses_api("aipg") is True


def test_aipg_get_openai_compatible_provider_info():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("aipg")
    assert provider is not None
    config = create_config_class(provider)()

    with mock.patch.dict(os.environ, {"AIPG_API_KEY": "grid-test"}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == AIPG_API_BASE
        assert api_key == "grid-test"

    with mock.patch.dict(
        os.environ,
        {
            "AIPG_API_KEY": "env-key",
            "AIPG_API_BASE": "https://operator.example/v1",
        },
        clear=True,
    ):
        api_base, api_key = config._get_openai_compatible_provider_info("https://explicit.example/v1", "explicit-key")
        assert api_base == "https://explicit.example/v1"
        assert api_key == "explicit-key"

    mapped = config.map_openai_params(
        non_default_params={"max_completion_tokens": 12, "temperature": 0.2},
        optional_params={},
        model="gpt-oss-120b",
        drop_params=False,
    )
    assert mapped["max_tokens"] == 12
    assert mapped["temperature"] == 0.2


def test_get_llm_provider_aipg():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    with mock.patch.dict(os.environ, {"AIPG_API_KEY": "grid-test"}, clear=True):
        model, provider, api_key, api_base = get_llm_provider("aipg/gpt-oss-120b")

    assert model == "gpt-oss-120b"
    assert provider == "aipg"
    assert api_key == "grid-test"
    assert api_base == AIPG_API_BASE


def test_aipg_model_metadata():
    model_cost = litellm.get_model_cost_map(url="")
    expected = {
        "aipg/gpt-oss-120b": (60000, 7.5e-08, 3e-07),
        "aipg/deepseek-v4-flash-nvfp4": (262144, 7e-08, 1.4e-07),
        "aipg/Smollm-135m": (2048, 5e-09, 1e-08),
    }
    for model, (context, input_cost, output_cost) in expected.items():
        info = model_cost[model]
        assert info["litellm_provider"] == "aipg"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == context
        assert info["input_cost_per_token"] == input_cost
        assert info["output_cost_per_token"] == output_cost
