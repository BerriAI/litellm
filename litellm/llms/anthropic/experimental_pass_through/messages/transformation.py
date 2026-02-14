from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import verbose_logger
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    AnthropicMessagesRequest,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.anthropic_tool_search import get_tool_search_beta_header
from litellm.types.router import GenericLiteLLMParams

from ...common_utils import (
    AnthropicError,
    AnthropicModelInfo,
    optionally_handle_anthropic_oauth,
)

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
            "context_management",
            "output_format",
            "inference_geo",
            "speed",
            "output_config",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    @staticmethod
    def _filter_billing_headers_from_system(system_param):
        """
        Filter out x-anthropic-billing-header metadata from system parameter.

        Args:
            system_param: Can be a string or a list of system message content blocks

        Returns:
            Filtered system parameter (string or list), or None if all content was filtered
        """
        if isinstance(system_param, str):
            # If it's a string and starts with billing header, filter it out
            if system_param.startswith("x-anthropic-billing-header:"):
                return None
            return system_param
        elif isinstance(system_param, list):
            # Filter list of system content blocks
            filtered_list = []
            for content_block in system_param:
                if isinstance(content_block, dict):
                    text = content_block.get("text", "")
                    content_type = content_block.get("type", "")
                    # Skip text blocks that start with billing header
                    if content_type == "text" and text.startswith(
                        "x-anthropic-billing-header:"
                    ):
                        continue
                    filtered_list.append(content_block)
                else:
                    # Keep non-dict items as-is
                    filtered_list.append(content_block)
            return filtered_list if len(filtered_list) > 0 else None
        else:
            return system_param

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
        import os

        # Check for Anthropic OAuth token in Authorization header
        headers, api_key = optionally_handle_anthropic_oauth(
            headers=headers, api_key=api_key
        )
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if "x-api-key" not in headers and "authorization" not in headers and api_key:
            headers["x-api-key"] = api_key
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
        )

        return headers, api_base

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

        # Filter out x-anthropic-billing-header from system messages
        system_param = anthropic_messages_optional_request_params.get("system")
        if system_param is not None:
            filtered_system = self._filter_billing_headers_from_system(system_param)
            if filtered_system is not None and len(filtered_system) > 0:
                anthropic_messages_optional_request_params["system"] = filtered_system
            else:
                # Remove system parameter if all content was filtered out
                anthropic_messages_optional_request_params.pop("system", None)

        ####### get required params for all anthropic messages requests ######
        verbose_logger.debug(f"TRANSFORMATION DEBUG - Messages: {messages}")
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
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )

        # Use the shared streaming handler for Anthropic
        handler = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )
        return handler.get_async_streaming_response_iterator(
            httpx_response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )

    @staticmethod
    def _update_headers_with_anthropic_beta(
        headers: dict,
        optional_params: dict,
        custom_llm_provider: str = "anthropic",
    ) -> dict:
        """
        Auto-inject anthropic-beta headers based on features used.

        Handles:
        - context_management: adds 'context-management-2025-06-27'
        - tool_search: adds provider-specific tool search header
        - output_format: adds 'structured-outputs-2025-11-13'
        - speed: adds 'fast-mode-2026-02-01'

        Args:
            headers: Request headers dict
            optional_params: Optional parameters including tools, context_management, output_format, speed
            custom_llm_provider: Provider name for looking up correct tool search header
        """
        beta_values: set = set()

        # Get existing beta headers if any
        existing_beta = headers.get("anthropic-beta")
        if existing_beta:
            beta_values.update(b.strip() for b in existing_beta.split(","))

        # Check for context management
        context_management_param = optional_params.get("context_management")
        if context_management_param is not None:
            # Check edits array for compact_20260112 type
            edits = context_management_param.get("edits", [])
            has_compact = False
            has_other = False

            for edit in edits:
                edit_type = edit.get("type", "")
                if edit_type == "compact_20260112":
                    has_compact = True
                else:
                    has_other = True

            # Add compact header if any compact edits exist
            if has_compact:
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_01_12.value)

            # Add context management header if any other edits exist
            if has_other:
                beta_values.add(
                    ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value
                )

        # Check for structured outputs
        if optional_params.get("output_format") is not None:
            beta_values.add(
                ANTHROPIC_BETA_HEADER_VALUES.STRUCTURED_OUTPUT_2025_09_25.value
            )

        # Check for fast mode
        if optional_params.get("speed") == "fast":
            beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.FAST_MODE_2026_02_01.value)

        # Check for tool search tools
        tools = optional_params.get("tools")
        if tools:
            anthropic_model_info = AnthropicModelInfo()
            if anthropic_model_info.is_tool_search_used(tools):
                # Use provider-specific tool search header
                tool_search_header = get_tool_search_beta_header(custom_llm_provider)
                beta_values.add(tool_search_header)

        if beta_values:
            headers["anthropic-beta"] = ",".join(sorted(beta_values))

        return headers
