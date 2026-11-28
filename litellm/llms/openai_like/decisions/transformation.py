from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig


class JSONProviderDecisionsConfig(BaseDecisionsConfig):
    def __init__(self, provider: SimpleProviderConfig, model: str = "") -> None:
        self.api_key_env = provider.api_key_envs
        self.api_base_env = (provider.api_base_env,) if provider.api_base_env else ()
        self._base_url = provider.default_base_url(model)

    def get_default_api_base(self) -> str | None:
        return self._base_url


def json_provider_decisions_config(model: str, slug: str) -> JSONProviderDecisionsConfig | None:
    provider: Final = JSONProviderRegistry.get(slug)
    if provider is None or not JSONProviderRegistry.serves_endpoint(slug, model, "/v1/systemone"):
        return None
    return JSONProviderDecisionsConfig(provider, model)
