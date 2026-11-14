from collections.abc import Mapping

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig


class LiteLLMProxyDecisionsConfig(OpenAIDecisionsConfig):
    """A LiteLLM proxy serves /v1/decisions in the OpenAI format, with its own base URL and key."""

    api_base_env = ("LITELLM_PROXY_API_BASE",)
    api_key_env = ("LITELLM_PROXY_API_KEY",)

    def get_default_api_base(self) -> str | None:
        return None

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "api_base is required for the LiteLLM proxy, set api_base or LITELLM_PROXY_API_BASE"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return BaseDecisionsConfig.resolve_api_base(self, api_base)

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return BaseDecisionsConfig.resolve_api_key(self, api_key)

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key
