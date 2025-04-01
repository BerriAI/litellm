import os
from typing import Dict, List, Optional

import httpx

from litellm.constants import DEFAULT_ANTHROPIC_API_BASE, DEFAULT_ANTHROPIC_API_VERSION
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_request import (
    AnthropicMessagesRequestOptionalParams,
    AnthropicMessagesRequestParams,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicMessagesStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    def get_supported_anthropic_messages_optional_params(self, model: str) -> list:
        return [
            "max_tokens",
            "metadata",
            "stop_sequences",
            "stream",
            "system",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
            "timeout",
        ]

    def map_anthropic_messages_optional_params(
        self,
        anthropic_messages_optional_params: AnthropicMessagesRequestOptionalParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        return dict(anthropic_messages_optional_params)

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        if "x-api-key" not in headers:
            headers["x-api-key"] = api_key or os.getenv("ANTHROPIC_API_KEY")
        return headers

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """No transform applied since inputs from Anthropic are in Anthropic spec already"""
        return dict(
            AnthropicMessagesRequestParams(
                model=model,
                messages=messages,
                **anthropic_messages_optional_request_params,
            )
        )

    def transform_response_to_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """No transform applied since outputs from Anthropic are in Anthropic spec already"""
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise AnthropicError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return AnthropicMessagesResponse(**raw_response_json)

    def transform_response_to_anthropic_messages_streaming_response(
        self,
        model: str,
        parsed_chunk: Dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesStreamingResponse:
        """No transform applied since outputs from Anthropic are in Anthropic spec already"""
        return AnthropicMessagesStreamingResponse(**parsed_chunk)
