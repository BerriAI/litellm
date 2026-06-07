import json
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import LlmProviders
from litellm.litellm_core_utils.get_llm_provider_logic import (
    _get_openai_compatible_provider_info,
)
from litellm.llms.xai.oauth import XAIOAuthChatConfig, XAIOAuthResponsesAPIConfig
from litellm.utils import (
    ProviderConfigManager,
    get_optional_params,
    validate_environment,
)


def test_xai_oauth_provider_config_routing():
    chat_config = ProviderConfigManager.get_provider_chat_config(
        model="grok-3-mini",
        provider=LlmProviders.XAI_OAUTH,
    )
    responses_config = ProviderConfigManager.get_provider_responses_api_config(
        model="grok-3-mini",
        provider=LlmProviders.XAI_OAUTH,
    )

    assert isinstance(chat_config, XAIOAuthChatConfig)
    assert isinstance(responses_config, XAIOAuthResponsesAPIConfig)


def test_xai_oauth_openai_compatible_provider_info():
    model, custom_llm_provider, dynamic_api_key, api_base = (
        _get_openai_compatible_provider_info(
            model="xai_oauth/grok-3-mini",
            api_base="https://api.x.ai/v1",
            api_key="oauth-token",
            dynamic_api_key=None,
        )
    )

    assert model == "grok-3-mini"
    assert custom_llm_provider == "xai_oauth"
    assert api_base == "https://api.x.ai/v1"
    assert dynamic_api_key == "oauth-token"


def test_xai_oauth_get_model_info_uses_xai_pricing_metadata():
    model_info = litellm.get_model_info("xai_oauth/grok-3-mini")

    assert model_info["litellm_provider"] == "xai_oauth"
    assert model_info["key"] == "xai_oauth/grok-3-mini"
    assert model_info["mode"] == "chat"


def test_xai_oauth_validate_environment_reads_auth_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"access_token": "token"}))

    result = validate_environment(model="xai_oauth/grok-3-mini")

    assert result == {"keys_in_environment": True, "missing_keys": []}


def test_xai_oauth_optional_params_use_xai_config():
    result = get_optional_params(
        model="grok-3-mini",
        custom_llm_provider="xai_oauth",
        temperature=0.2,
        drop_params=True,
    )

    assert result["temperature"] == 0.2
