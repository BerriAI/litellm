"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

from typing import List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LiteLLMProxyChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("LITELLM_PROXY_API_BASE")  # type: ignore
        dynamic_api_key = api_key or get_secret_str("LITELLM_PROXY_API_KEY")
        return api_base, dynamic_api_key

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base, api_key = self._get_openai_compatible_provider_info(api_base, api_key)
        if api_base is None:
            raise ValueError(
                "api_base not set for LiteLLM Proxy route. Set in env via `LITELLM_PROXY_API_BASE`"
            )
        models = super().get_models(api_key=api_key, api_base=api_base)
        return [f"litellm_proxy/{model}" for model in models]
