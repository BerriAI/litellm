"""
Claude Code Native Provider Transformation

This module provides the configuration for the Claude Code Native provider,
a variant of Anthropic that uses OAuth authentication and a specific system prompt.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import httpx

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.llms.anthropic import AnthropicSystemMessageContent
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class ClaudeCodeNativeConfig(AnthropicConfig):
    """
    Configuration for Claude Code Native Provider.

    This provider is a variant of Anthropic with the following differences:
    1. Always includes specific headers: "anthropic-beta": "oauth-2025-04-20" and "anthropic-version": "2023-06-01"
    2. Always prepends "You are Claude Code, Anthropic's official CLI for Claude." as the first system message
    """

    CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."
    REQUIRED_HEADERS = {
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
    }

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
        system: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "claude_code_native"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set required headers.

        The provider headers take priority for the required header keys.
        User-specified headers are merged under the provider headers.
        """
        # Call parent's validate_environment first
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Merge with required headers (provider headers take priority for these keys)
        # First, apply any user-provided headers from extra_headers
        extra_headers = optional_params.get("extra_headers", {})
        headers.update(extra_headers)

        # Then, override with our required headers (provider takes priority)
        headers.update(self.REQUIRED_HEADERS)

        # Use Authorization: Bearer <key> instead of x-api-key: <key>
        if "x-api-key" in headers:
            api_key_value = headers.pop("x-api-key")
            headers["Authorization"] = f"Bearer {api_key_value}"

        return headers

    def update_headers_with_optional_anthropic_beta(
        self, headers: dict, optional_params: dict
    ) -> dict:
        """
        Update headers with optional anthropic beta, ensuring our required beta header is preserved.
        """
        headers = super().update_headers_with_optional_anthropic_beta(
            headers=headers, optional_params=optional_params
        )

        # Ensure our required beta header is present
        required_beta = self.REQUIRED_HEADERS.get("anthropic-beta")
        if required_beta:
            existing_beta = headers.get("anthropic-beta")
            if existing_beta:
                if required_beta not in existing_beta:
                    headers["anthropic-beta"] = f"{existing_beta},{required_beta}"
            else:
                headers["anthropic-beta"] = required_beta

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request to Anthropic format and prepend the Claude Code system message.

        The system messages are prepended to ensure our required system message is always first.
        """
        # Call parent's transform_request first
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Prepend the Claude Code system message
        claude_code_system_message = AnthropicSystemMessageContent(
            type="text",
            text=self.CLAUDE_CODE_SYSTEM_PROMPT,
        )

        # If system messages exist, prepend our system message
        if "system" in data:
            existing_system = data["system"]
            if isinstance(existing_system, list):
                # Prepend to existing list
                data["system"] = [claude_code_system_message] + existing_system
            else:
                # Convert to list and prepend
                data["system"] = [claude_code_system_message, existing_system]
        else:
            # No existing system messages, just set ours
            data["system"] = [claude_code_system_message]

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform response from Anthropic format.

        Simply delegates to parent's transform_response since we don't modify the response.
        """
        return super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
