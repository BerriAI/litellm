"""
Pass Anthropic /v1/messages requests through to vLLM's native Anthropic-compatible endpoint.

When `disable_anthropic_translation` is set on a `hosted_vllm` deployment (or the
DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION env var is truthy), LiteLLM skips the
Anthropic→OpenAI chat/completions translation and POSTs the original Anthropic payload
directly to `{api_base}/v1/messages`.
"""

import os
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


def _should_skip_anthropic_translation(litellm_params: GenericLiteLLMParams) -> bool:
    """Return True when Anthropic→OpenAI translation should be bypassed for hosted_vllm.

    Checked in priority order:
    1. Per-deployment ``disable_anthropic_translation`` in litellm_params
    2. Global env var ``DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION``
    """
    param_flag = litellm_params.get("disable_anthropic_translation")
    if param_flag is not None:
        return bool(param_flag)
    return os.environ.get("DISABLE_HOSTED_VLLM_ANTHROPIC_TRANSLATION", "").lower() in (
        "true",
        "1",
        "yes",
    )


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

    # Anthropic built-in tool types that vLLM does not support in /v1/messages
    _UNSUPPORTED_TOOL_TYPES = frozenset(
        {
            "web_search",
            "computer",
            "text_editor",
            "bash",
        }
    )

    def _strip_unsupported_tools(
        self, tools: Optional[List[Dict]]
    ) -> Optional[List[Dict]]:
        """Remove Anthropic built-in tools (web_search, computer_use, etc.) that
        vLLM's /v1/messages endpoint does not understand."""
        if not tools:
            return tools
        return [
            t
            for t in tools
            if not isinstance(t, dict)
            or t.get("type", "") not in self._UNSUPPORTED_TOOL_TYPES
            and not t.get("type", "").startswith(
                tuple(f"{p}_" for p in self._UNSUPPORTED_TOOL_TYPES)
            )
        ]

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

        tools = self._strip_unsupported_tools(
            anthropic_messages_optional_request_params.get("tools")
        )
        if tools:
            anthropic_messages_optional_request_params["tools"] = tools
        else:
            anthropic_messages_optional_request_params.pop("tools", None)

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
