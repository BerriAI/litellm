from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import AnthropicMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams

from ...common_utils import AnthropicError

DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return [
            "messages",
            "model",
            "system",
            "max_tokens",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if "x-api-key" not in headers:
            headers["x-api-key"] = api_key
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        return headers

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        No transformation is needed for Anthropic messages


        This takes in a request in the Anthropic /v1/messages API spec -> transforms it to /v1/messages API spec (i.e) no transformation is needed
        """
        max_tokens = anthropic_messages_optional_request_params.pop("max_tokens", None)
        if max_tokens is None:
            raise AnthropicError(
                message="max_tokens is required for Anthropic /v1/messages API",
                status_code=400,
            )
        ####### get required params for all anthropic messages requests ######
        anthropic_messages_request: AnthropicMessagesRequest = AnthropicMessagesRequest(
            messages=messages,
            max_tokens=max_tokens,
            model=model,
            **anthropic_messages_optional_request_params,
        )
        return dict(anthropic_messages_request)

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """
        No transformation is needed for Anthropic messages, since we want the response in the Anthropic /v1/messages API spec
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise AnthropicError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return AnthropicMessagesResponse(**raw_response_json)

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
        from datetime import datetime

        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging,
        )
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        # Create success handler object
        passthrough_success_handler_obj = PassThroughEndpointLogging()

        # Use the existing streaming handler for Anthropic
        start_time = datetime.now()
        return PassThroughStreamingHandler.chunk_processor(
            response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=start_time,
            passthrough_success_handler_obj=passthrough_success_handler_obj,
            url_route="/v1/messages",
        )
