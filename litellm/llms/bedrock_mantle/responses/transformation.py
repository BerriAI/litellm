"""
Amazon Bedrock Mantle - OpenAI-compatible Responses API.

Routes /v1/responses to the provider's native Responses endpoint instead of
the chat-completions translation fallback.
"""

from typing import Optional

from litellm.llms.bedrock_mantle.chat.transformation import BedrockMantleChatConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        _, api_key = BedrockMantleChatConfig()._get_openai_compatible_provider_info(
            api_base=litellm_params.api_base,
            api_key=litellm_params.api_key,
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        (
            resolved_api_base,
            _,
        ) = BedrockMantleChatConfig()._get_openai_compatible_provider_info(
            api_base=api_base or litellm_params.get("api_base"),
            api_key=litellm_params.get("api_key"),
        )
        if not resolved_api_base:
            raise ValueError(
                "api_base is required for bedrock_mantle Responses API. "
                "Set BEDROCK_MANTLE_API_BASE or BEDROCK_MANTLE_REGION."
            )
        api_base = resolved_api_base.rstrip("/")
        if api_base.endswith("/responses"):
            return api_base
        if api_base.endswith("/v1"):
            return f"{api_base}/responses"
        return f"{api_base}/v1/responses"

    def supports_native_websocket(self) -> bool:
        return False
