from typing import Final

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ALL_RESPONSES_API_TOOL_PARAMS, ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ...openai.responses.transformation import OpenAIResponsesAPIConfig

DEFAULT_DASHSCOPE_API_BASE: Final[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeResponsesAPIConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DASHSCOPE

    def remove_cache_control_flag_from_input_and_tools(
        self,
        model: str,
        input: str | ResponseInputParam,
        tools: list[ALL_RESPONSES_API_TOOL_PARAMS]  # mutable-ok: override BaseResponsesAPIConfig signature
        | None = None,  # mutable-ok: override BaseResponsesAPIConfig signature
    ) -> tuple[  # mutable-ok: override BaseResponsesAPIConfig signature
        str | ResponseInputParam,
        list[ALL_RESPONSES_API_TOOL_PARAMS] | None,  # mutable-ok: override BaseResponsesAPIConfig signature
    ]:
        """
        Override to preserve cache_control for DashScope.
        DashScope supports cache_control - don't strip it.
        """
        return input, tools

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: override BaseResponsesAPIConfig signature
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: override BaseResponsesAPIConfig signature
        resolved_params: Final = litellm_params or GenericLiteLLMParams()

        api_key: Final = resolved_params.api_key or litellm.api_key or get_secret_str("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {  # mutable-ok: header dict update
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: override BaseResponsesAPIConfig signature
    ) -> str:
        """
        Get the endpoint for DashScope Responses API.

        Resolves the base URL in order:
        1. litellm_params["responses_api_base"] (dedicated Responses base URL, e.g. for MaaS workspace URLs)
        2. DASHSCOPE_RESPONSES_API_BASE environment variable
        3. api_base passed in request or litellm_params["api_base"]
        4. DASHSCOPE_API_BASE environment variable
        5. Default: https://dashscope.aliyuncs.com/compatible-mode/v1
        """
        raw_base_url: Final[str] = str(
            litellm_params.get("responses_api_base")
            or get_secret_str("DASHSCOPE_RESPONSES_API_BASE")
            or api_base
            or litellm.api_base
            or get_secret_str("DASHSCOPE_API_BASE")
            or DEFAULT_DASHSCOPE_API_BASE
        )

        # Remove trailing slashes
        clean_base_url: Final[str] = raw_base_url.rstrip("/")

        if clean_base_url.endswith("/responses"):
            return clean_base_url

        if clean_base_url.endswith("/v1"):
            return f"{clean_base_url}/responses"

        return f"{clean_base_url}/v1/responses"

    def supports_native_websocket(self) -> bool:
        """DashScope does not support native WebSocket for Responses API"""
        return False
