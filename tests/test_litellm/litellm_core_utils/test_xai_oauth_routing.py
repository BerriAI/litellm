import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import LlmProviders
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
from litellm.litellm_core_utils.get_llm_provider_logic import (
    _get_openai_compatible_provider_info,
)
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import (
    ProviderConfigManager,
    get_optional_params,
    validate_environment,
)


def test_xai_provider_config_routing():
    chat_config = ProviderConfigManager.get_provider_chat_config(
        model="grok-3-mini",
        provider=LlmProviders.XAI,
    )
    responses_config = ProviderConfigManager.get_provider_responses_api_config(
        model="grok-3-mini",
        provider=LlmProviders.XAI,
    )

    assert isinstance(chat_config, XAIChatConfig)
    assert isinstance(responses_config, XAIResponsesAPIConfig)


def test_xai_openai_compatible_provider_info():
    model, custom_llm_provider, dynamic_api_key, api_base = (
        _get_openai_compatible_provider_info(
            model="xai/grok-3-mini",
            api_base="https://api.x.ai/v1",
            api_key="api-key",
            dynamic_api_key=None,
        )
    )

    assert model == "grok-3-mini"
    assert custom_llm_provider == "xai"
    assert api_base == "https://api.x.ai/v1"
    assert dynamic_api_key == "api-key"


def test_xai_get_model_info_uses_xai_pricing_metadata():
    model_info = litellm.get_model_info("xai/grok-3-mini")

    assert model_info["litellm_provider"] == "xai"
    assert model_info["key"] == "xai/grok-3-mini"
    assert model_info["mode"] == "chat"


def test_xai_validate_environment_reads_api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "api-key")

    result = validate_environment(model="xai/grok-3-mini")

    assert result == {"keys_in_environment": True, "missing_keys": []}


def test_xai_oauth_flag_is_generic_litellm_param():
    litellm_params = GenericLiteLLMParams(use_xai_oauth=True)
    runtime_params = get_litellm_params(use_xai_oauth=True)
    result = get_optional_params(
        model="grok-3-mini",
        custom_llm_provider="xai",
        temperature=0.2,
        drop_params=True,
    )

    assert result["temperature"] == 0.2
    assert litellm_params.use_xai_oauth is True
    assert runtime_params["use_xai_oauth"] is True
    assert "use_xai_oauth" not in result
