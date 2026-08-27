"""
OpenRouter Responses API Configuration.

OpenRouter supports the Responses API at https://openrouter.ai/api/v1/responses
with OpenAI-compatible request/response format, including reasoning with
encrypted_content for multi-turn stateless workflows.

Docs: https://openrouter.ai/docs/api/reference/responses/overview
"""

from typing import Any, Final

import httpx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenRouterResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OpenRouter's Responses API.

    Inherits from OpenAIResponsesAPIConfig since OpenRouter's Responses API
    is compatible with OpenAI's Responses API specification.

    Key difference from direct OpenAI:
    - Uses https://openrouter.ai/api/v1 as the API base
    - Uses OPENROUTER_API_KEY for authentication
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENROUTER

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key: Final = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
        )

        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """OpenRouter does not support native WebSocket for Responses API"""
        return False

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """
        Same as the chat completions path: always ask OpenRouter to include
        cost data in the response's `usage` object, so `transform_response_api_response`
        below has something to extract. Without this, OpenRouter omits `usage.cost`
        and every Responses API request gets logged with $0 spend.
        """
        request: Final = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        if "usage" not in request:
            request["usage"] = {"include": True}

        return request

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ResponsesAPIResponse:
        """
        Extracts cost information from the response body, mirroring
        `OpenrouterConfig.transform_response` on the chat completions path.

        OpenRouter returns cost information in the `usage` object when
        `usage.include=true` is set on the request (see `transform_responses_api_request`).
        """
        response: Final = super().transform_response_api_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        try:
            response_json: Final = raw_response.json()
            if "usage" in response_json and response_json["usage"]:
                response_cost: Final = response_json["usage"].get("cost")
                if response_cost is not None:
                    # Store cost in hidden params for the cost calculator to use
                    if not hasattr(response, "_hidden_params"):
                        response._hidden_params = {}
                    if "additional_headers" not in response._hidden_params:
                        response._hidden_params["additional_headers"] = {}
                    response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(
                        response_cost
                    )
        except Exception:
            # If we can't extract cost, continue without it - don't fail the response
            pass

        return response
