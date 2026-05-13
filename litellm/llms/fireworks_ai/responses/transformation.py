"""
FireworksAIResponsesConfig for OpenAI-compatible Responses API support.
https://docs.fireworks.ai/api-reference/post-responses
"""

from typing import Optional, Union

from litellm.llms.openai_like.responses.transformation import (
    OpenAILikeResponsesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class FireworksAIResponsesConfig(OpenAILikeResponsesConfig):
    @property
    def custom_llm_provider(self) -> Union[str, LlmProviders]:  # type: ignore[override]
        return "fireworks_ai"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base
            or get_secret_str("FIREWORKS_API_BASE")
            or "https://api.fireworks.ai/inference/v1"
        )
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/responses"):
            if api_base.endswith("/v1"):
                api_base = f"{api_base}/responses"
            else:
                api_base = f"{api_base}/v1/responses"
        return api_base
