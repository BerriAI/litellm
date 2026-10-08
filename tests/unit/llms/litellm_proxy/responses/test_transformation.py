from typing import Final

import pytest

from litellm.llms.litellm_proxy.responses.transformation import LiteLLMProxyResponsesAPIConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_litellm_proxy_responses_config() -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(
        model="litellm_proxy/gpt-5.5", provider=LlmProviders.LITELLM_PROXY
    )
    assert isinstance(config, LiteLLMProxyResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.LITELLM_PROXY


@pytest.mark.parametrize("api_base", ["https://my-proxy.example.com", "https://my-proxy.example.com/"])
def test_get_complete_url_appends_responses_path(api_base: str) -> None:
    assert (
        LiteLLMProxyResponsesAPIConfig().get_complete_url(api_base=api_base, litellm_params={})
        == "https://my-proxy.example.com/responses"
    )


def test_get_complete_url_requires_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    with pytest.raises(ValueError, match="api_base not set"):
        LiteLLMProxyResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={})
