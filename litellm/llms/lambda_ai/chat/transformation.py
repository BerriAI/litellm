"""
Translate from OpenAI's `/v1/chat/completions` to Lambda's `/v1/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class LambdaAIChatConfig(OpenAILikeChatConfig):
    """
    Lambda AI is OpenAI-compatible with standard endpoints
    """
    
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "lambda_ai"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Lambda AI is openai compatible, we just need to set the api_base
        api_base = (
            api_base
            or get_secret_str("LAMBDA_API_BASE")
            or "https://api.lambda.ai/v1"  # Default Lambda API base URL
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("LAMBDA_API_KEY")
        return api_base, dynamic_api_key