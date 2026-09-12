"""
Support for OpenAI's `/v1/chat/completions` endpoint via Merge AI Gateway.

Calls done in OpenAI/openai.py as Merge AI Gateway is openai-compatible.

Docs: https://docs.merge.dev/
"""

from typing import Final

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import MergeAIGatewayException

DEFAULT_API_BASE: Final = "https://api-gateway.merge.dev/v1/openai"


class MergeAIGatewayConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "merge_ai_gateway"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("MERGE_AI_GATEWAY_API_KEY") or get_secret_str("MERGE_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("MERGE_AI_GATEWAY_API_BASE") or DEFAULT_API_BASE

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self.get_api_base(api_base), self.get_api_key(api_key)

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return super().get_models(api_key=api_key, api_base=self.get_api_base(api_base))

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return MergeAIGatewayException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
