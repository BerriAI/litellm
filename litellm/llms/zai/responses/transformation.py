"""
Z.AI OpenAI-compatible Responses API transformation config.
"""

from typing import Final

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class ZAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Z.AI exposes an OpenAI-compatible Responses API at
    https://api.z.ai/api/v1 (see https://docs.z.ai/guides/llm/glm-5.3).

    The endpoint authenticates with the Z.AI API key sent as an
    ``Authorization: Bearer`` header.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ZAI

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()

        api_key: Final = litellm_params.api_key or litellm.api_key or get_secret_str("ZAI_API_KEY")

        headers.setdefault("Content-Type", "application/json")
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url = api_base or get_secret_str("ZAI_RESPONSES_API_BASE") or "https://api.z.ai/api/v1"

        base_url = base_url.rstrip("/")
        if base_url.endswith("/responses"):
            return base_url
        return f"{base_url}/responses"
