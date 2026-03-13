"""
Perplexity Responses API — OpenAI-compatible.

The only provider quirks:
- cost returned as dict → handled by ResponseAPIUsage.parse_cost validator
- preset models (preset/pro-search) → handled by transform_responses_api_request
- HTTP 200 with status:"failed" → raised as exception in transform_response_api_response

Ref: https://docs.perplexity.ai/api-reference/responses-post
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class PerplexityResponsesConfig(OpenAIResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """Ref: https://docs.perplexity.ai/api-reference/responses-post"""
        return [
            "max_output_tokens",
            "stream",
            "temperature",
            "top_p",
            "tools",
            "reasoning",
            "instructions",
            "models",
        ]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.PERPLEXITY

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("PERPLEXITYAI_API_KEY")
            or get_secret_str("PERPLEXITY_API_KEY")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(self, api_base: Optional[str], litellm_params: dict) -> str:
        api_base = (
            api_base
            or get_secret_str("PERPLEXITY_API_BASE")
            or "https://api.perplexity.ai"
        )
        return f"{api_base.rstrip('/')}/v1/responses"

    def _ensure_message_type(
        self, input: Union[str, ResponseInputParam]
    ) -> Union[str, ResponseInputParam]:
        """Ensure list input items have type='message' (required by Perplexity)."""
        if isinstance(input, str):
            return input
        if isinstance(input, list):
            result: List[Any] = []
            for item in input:
                if isinstance(item, dict) and "type" not in item:
                    item = {**item, "type": "message"}
                result.append(item)
            return result
        return input

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Handle preset/ model prefix: send as {"preset": name} instead of {"model": name}."""
        input = self._ensure_message_type(input)
        if model.startswith("preset/"):
            input = self._validate_input_param(input)
            data: Dict = {
                "preset": model[len("preset/") :],
                "input": input,
            }
            data.update(response_api_optional_request_params)
            return data
        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """Check for Perplexity's status:'failed' on HTTP 200 before delegating to base."""
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raw_response_json = None

        if (
            isinstance(raw_response_json, dict)
            and raw_response_json.get("status") == "failed"
        ):
            error = raw_response_json.get("error", {})
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=error.get("message", "Unknown Perplexity error"),
            )

        return super().transform_response_api_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def supports_native_websocket(self) -> bool:
        """Perplexity does not support native WebSocket for Responses API"""
        return False
