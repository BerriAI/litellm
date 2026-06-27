"""
Responses API transformation for Hosted VLLM provider.

vLLM natively supports the OpenAI-compatible /v1/responses endpoint,
so this config enables direct routing instead of falling back to
the chat completions → responses conversion pipeline.
"""

import base64
import hashlib
import secrets
from typing import Dict, Optional, Union

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.llms.openai import ResponsesAPIRequestParams
from litellm.types.llms.openai import ResponseInputParam


class HostedVLLMResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Hosted VLLM Responses API support.

    Extends OpenAI's config since vLLM follows OpenAI's API spec,
    but uses HOSTED_VLLM_API_BASE for the base URL and defaults
    to "fake-api-key" when no API key is provided (vLLM does not
    require authentication by default).
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.HOSTED_VLLM

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        input = self._validate_input_param(input)
        final_request_params = dict(
            ResponsesAPIRequestParams(model=model, input=input, **response_api_optional_request_params)
        )

        if final_request_params.get("cache_salt"):
            return final_request_params

        metadata = getattr(litellm_params, "metadata", {}) or {}
        caller_id = (
            metadata.get("user_api_key_user_id")
            or metadata.get("user_api_key_team_id")
            or metadata.get("user_api_key_end_user_id")
        )
        if caller_id:
            cache_salt = base64.b64encode(hashlib.sha256(caller_id.encode()).digest()).decode()
        else:
            cache_salt = base64.b64encode(secrets.token_bytes(16)).decode()
        final_request_params["cache_salt"] = cache_salt
        return final_request_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("HOSTED_VLLM_API_KEY")
            or "fake-api-key"
        )  # vllm does not require an api key
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
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")

        if api_base is None:
            raise ValueError(
                "api_base not set for Hosted VLLM responses API. "
                "Set via api_base parameter or HOSTED_VLLM_API_BASE environment variable"
            )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # If api_base already ends with /v1, append /responses
        # Otherwise append /v1/responses
        if api_base.endswith("/v1"):
            return f"{api_base}/responses"

        return f"{api_base}/v1/responses"

    def supports_native_websocket(self) -> bool:
        """Hosted vLLM does not support native WebSocket for Responses API"""
        return False
