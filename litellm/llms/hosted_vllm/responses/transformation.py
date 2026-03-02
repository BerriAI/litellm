from typing import Optional

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class HostedVLLMResponsesAPIConfig(OpenAIResponsesAPIConfig):
    supports_fallback_to_chat: bool = True

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.HOSTED_VLLM

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("HOSTED_VLLM_API_KEY")
            or "fake-api-key"
        )
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        if api_base is None:
            raise ValueError(
                "api_base must be provided for hosted_vllm. Set in call or via HOSTED_VLLM_API_BASE env var."
            )

        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

