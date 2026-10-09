"""
OpenRouter Responses API Configuration.

OpenRouter supports the Responses API at https://openrouter.ai/api/v1/responses
with OpenAI-compatible request/response format, including reasoning with
encrypted_content for multi-turn stateless workflows.

Docs: https://openrouter.ai/docs/api/reference/responses/overview
"""

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER
from litellm.litellm_core_utils.hidden_params import set_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
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

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        response: Final = super().transform_response_api_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )
        self._set_provider_reported_cost(response)
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        response_event: Final = super().transform_streaming_response(
            model=model,
            parsed_chunk=dict(parsed_chunk),
            logging_obj=logging_obj,
        )
        terminal_response: Final = (
            response_event.response
            if isinstance(
                response_event,
                (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent),
            )
            else None
        )
        if terminal_response is not None:
            self._set_provider_reported_cost(terminal_response)
        return response_event

    @staticmethod
    def _set_provider_reported_cost(response: ResponsesAPIResponse) -> None:
        cost: Final = response.usage.cost if response.usage is not None else None
        if cost is None:
            return
        additional_headers: Final = TypeAdapter(dict[str, object]).validate_python(
            response.hidden_params.get("additional_headers", {})
        )
        set_hidden_params(
            response,
            {
                **response.hidden_params,
                "additional_headers": {
                    **additional_headers,
                    RESPONSE_COST_HEADER: float(cost),
                },
            },
        )

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
