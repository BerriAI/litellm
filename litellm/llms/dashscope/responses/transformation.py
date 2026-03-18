"""
Translates from OpenAI's `/v1/responses` to DashScope's `/compatible-mode/v1/responses`

DashScope (Alibaba Cloud) provides an OpenAI-compatible endpoint, so most of
the heavy lifting is delegated to ``OpenAIResponsesAPIConfig``.  This subclass
only overrides provider identification, authentication, URL construction, and
the supported-parameter whitelist.
"""

from typing import Dict, List, Optional, Union

import httpx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

_DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_SUPPORTED_OPTIONAL_PARAMS: List[str] = [
    "instructions",
    "max_output_tokens",
    "metadata",
    "previous_response_id",
    "reasoning",
    "store",
    "stream",
    "temperature",
    "text",
    "tools",
    "tool_choice",
    "top_p",
    # LiteLLM request plumbing helpers
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
]


class DashScopeResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """Responses API configuration for DashScope (Alibaba Cloud)."""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DASHSCOPE

    def get_supported_openai_params(self, model: str) -> list:
        """Return the parameter whitelist for DashScope Responses API."""
        supported = ["input", "model"] + list(_SUPPORTED_OPTIONAL_PARAMS)
        # metadata is LiteLLM-internal; advertise all others
        if "metadata" in supported:
            supported.remove("metadata")
        return supported

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """Filter parameters to the DashScope-supported set."""
        params = {
            key: value
            for key, value in dict(response_api_optional_params).items()
            if key in _SUPPORTED_OPTIONAL_PARAMS
        }
        # LiteLLM metadata is internal-only; don't send to provider
        params.pop("metadata", None)
        return params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """Build auth headers for DashScope Responses API."""
        if litellm_params is None:
            litellm_params = GenericLiteLLMParams()
        elif isinstance(litellm_params, dict):
            litellm_params = GenericLiteLLMParams(**litellm_params)

        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("DASHSCOPE_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "DashScope API key is required. "
                "Set DASHSCOPE_API_KEY or pass api_key."
            )

        result_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if headers:
            result_headers.update(headers)
        return result_headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Construct DashScope Responses API endpoint."""
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("DASHSCOPE_API_BASE")
            or _DEFAULT_API_BASE
        )

        base_url = base_url.rstrip("/")

        if base_url.endswith("/responses"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/responses"
        if base_url.endswith("/compatible-mode/v1"):
            return f"{base_url}/responses"
        return f"{base_url}/compatible-mode/v1/responses"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> Exception:
        from litellm.llms.openai.common_utils import OpenAIError

        typed_headers: httpx.Headers = (
            headers
            if isinstance(headers, httpx.Headers)
            else httpx.Headers(headers or {})
        )
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )
