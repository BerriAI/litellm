import os
from unittest.mock import patch

TRUSTEDROUTER_API_BASE = "https://api.trustedrouter.com/v1"


def test_trustedrouter_json_registry():
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.TRUSTEDROUTER.value == "trustedrouter"
    assert litellm.LlmProviders("trustedrouter") == litellm.LlmProviders.TRUSTEDROUTER
    assert JSONProviderRegistry.exists("trustedrouter")
    config = JSONProviderRegistry.get("trustedrouter")
    assert config is not None
    assert config.base_url == TRUSTEDROUTER_API_BASE
    assert config.api_key_env == "TRUSTEDROUTER_API_KEY"
    assert config.api_base_env == "TRUSTEDROUTER_API_BASE"
    assert "/v1/chat/completions" in config.supported_endpoints
    assert "/v1/responses" in config.supported_endpoints


def test_trustedrouter_listed_in_openai_compatible_providers():
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
    )

    assert "trustedrouter" in openai_compatible_providers
    assert TRUSTEDROUTER_API_BASE in openai_compatible_endpoints


def test_trustedrouter_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("trustedrouter"))()

    with patch.dict(
        os.environ,
        {
            "TRUSTEDROUTER_API_KEY": "test-key",
            "TRUSTEDROUTER_API_BASE": "https://example.com/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://example.com/v1"
    assert api_key == "test-key"


def test_trustedrouter_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("trustedrouter/zdr")

    assert model == "zdr"
    assert provider == "trustedrouter"
    assert api_base == TRUSTEDROUTER_API_BASE
