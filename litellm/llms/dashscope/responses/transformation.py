"""
Responses API transformation for DashScope (Alibaba Qwen) provider.

DashScope exposes an OpenAI-compatible endpoint that serves the `/responses`
route, so this config enables direct routing to `{api_base}/responses` without
rewriting the upstream model id, instead of falling back to the chat
completions -> responses conversion pipeline.
"""

from typing import Optional

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for DashScope Responses API support.

    Extends OpenAI's config since DashScope follows the OpenAI API spec, but
    resolves the base URL/key from DASHSCOPE_API_BASE / DASHSCOPE_API_KEY and
    routes to the DashScope compatible-mode endpoint.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DASHSCOPE

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or get_secret_str("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "DashScope API key not set for responses API. "
                "Set via api_key parameter or DASHSCOPE_API_KEY environment variable"
            )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base
            or get_secret_str("DASHSCOPE_API_BASE")
            or DEFAULT_DASHSCOPE_API_BASE
        ).rstrip("/")

        if api_base.endswith("/v1"):
            return f"{api_base}/responses"

        return f"{api_base}/v1/responses"

    def supports_native_websocket(self) -> bool:
        return False
