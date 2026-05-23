"""
Pass Anthropic /v1/messages requests through to vLLM's native Anthropic-compatible endpoint.

When `disable_anthropic_translation` is set on a `hosted_vllm` deployment (or the
DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION env var is truthy), LiteLLM skips the
Anthropic→OpenAI chat/completions translation and POSTs the original Anthropic payload
directly to `{api_base}/v1/messages`.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams


class HostedVLLMAnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    """
    Sends Anthropic-format /v1/messages requests directly to a vLLM instance.

    No format translation is performed — the request body is forwarded as-is.
    The response is parsed back into AnthropicMessagesResponse.
    """

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
            "stream",
        ]

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        if api_key and "authorization" not in headers:
            headers["authorization"] = f"Bearer {api_key}"
        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            raise ValueError(
                "api_base is required for hosted_vllm Anthropic passthrough. "
                "Set it via `api_base` in litellm_params or HOSTED_VLLM_API_BASE env var."
            )
        base = api_base.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        # Standard api_base is http://host/v1 — strip the /v1 suffix so we can
        # append the canonical /v1/messages path without doubling it.
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/v1/messages"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        max_tokens = anthropic_messages_optional_request_params.pop("max_tokens", None)
        if max_tokens is None:
            raise ValueError("max_tokens is required for Anthropic /v1/messages API")

        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **anthropic_messages_optional_request_params,
        }

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise ValueError(
                f"Failed to parse vLLM Anthropic response: {raw_response.text}"
            )
        return AnthropicMessagesResponse(**raw_response_json)

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )

        handler = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )
        return handler.get_async_streaming_response_iterator(
            httpx_response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )
