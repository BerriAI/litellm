"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

from typing import List, Optional, Tuple

from litellm.secret_managers.main import get_secret_bool, get_secret_str
from litellm.types.router import LiteLLM_Params

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LiteLLMProxyChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List:
        params_list = super().get_supported_openai_params(model)
        params_list.append("thinking")
        params_list.append("reasoning_effort")
        return params_list

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "thinking":
                optional_params.setdefault("extra_body", {})["thinking"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

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

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("LITELLM_PROXY_API_KEY")

    @staticmethod
    def _should_use_litellm_proxy_by_default(
        litellm_params: Optional[LiteLLM_Params] = None,
    ):
        """
        Returns True if litellm proxy should be used by default for a given request

        Issue: https://github.com/BerriAI/litellm/issues/10559

        Use case:
        - When using Google ADK, users want a flag to dynamically enable sending the request to litellm proxy or not
        """
        import litellm

        if get_secret_bool("USE_LITELLM_PROXY") is True:
            return True
        litellm_params = litellm_params or {}
        if litellm_params.get("use_litellm_proxy", False) is True:
            return True
        if litellm.use_litellm_proxy is True:
            return True
        return False
