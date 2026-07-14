import json
from typing import Any, List, Tuple

import os

import httpx

from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolCallChunk
from litellm.types.utils import ModelResponse

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAPIKeyError,
    get_copilot_default_headers,
)


class GithubCopilotConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str,
    ) -> Tuple[str | None, str | None, str]:
        dynamic_api_base = (
            api_base
            or self.authenticator.get_api_base()
            or os.getenv("GITHUB_COPILOT_API_BASE")
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )
        try:
            dynamic_api_key = api_key or self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def _transform_messages(
        self,
        messages,
        model: str,
    ):
        import litellm

        # Check if system-to-assistant conversion is disabled
        if litellm.disable_copilot_system_to_assistant:
            # GitHub Copilot API now supports system prompts for all models (Claude, GPT, etc.)
            # No conversion needed - just return messages as-is
            return messages

        # Default behavior: convert system messages to assistant for compatibility
        transformed_messages = []
        for message in messages:
            if message.get("role") == "system":
                # Convert system message to assistant message
                transformed_message = message.copy()
                transformed_message["role"] = "assistant"
                transformed_messages.append(transformed_message)
            else:
                transformed_messages.append(message)

        return transformed_messages

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        # Get base headers from parent
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        # Add Copilot-specific headers (editor-version, user-agent, etc.)
        try:
            copilot_api_key = self.authenticator.get_api_key()
            copilot_headers = get_copilot_default_headers(copilot_api_key)
            validated_headers = {**copilot_headers, **validated_headers}
        except GetAPIKeyError:
            pass  # Will be handled later in the request flow

        # Add X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        validated_headers["X-Initiator"] = initiator

        # Add Copilot-Vision-Request header if request contains images
        if self._has_vision_content(messages):
            validated_headers["Copilot-Vision-Request"] = "true"

        return validated_headers

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GitHub Copilot.

        For Claude models that support extended thinking (Claude 4 family and Claude 3-7), includes thinking and reasoning_effort parameters.
        For other models, returns standard OpenAI parameters (which may include reasoning_effort for o-series models).
        """
        from litellm.utils import supports_reasoning

        # Get base OpenAI parameters
        base_params = super().get_supported_openai_params(model)

        # Add Claude-specific parameters for models that support extended thinking
        if "claude" in model.lower() and supports_reasoning(
            model=model.lower(),
        ):
            if "thinking" not in base_params:
                base_params.append("thinking")
            # reasoning_effort is not included by parent for Claude models, so add it
            if "reasoning_effort" not in base_params:
                base_params.append("reasoning_effort")

        return base_params

    def _determine_initiator(self, messages: List[AllMessageValues]) -> str:
        """
        Determine if request is user or agent initiated based on message roles.
        Returns 'agent' if any message has role 'tool' or 'assistant', otherwise 'user'.
        """
        for message in messages:
            role = message.get("role")
            if role in ["tool", "assistant"]:
                return "agent"
        return "user"

    def _has_vision_content(self, messages: List[AllMessageValues]) -> bool:
        """
        Check if any message contains vision content (images).
        Returns True if any message has content with vision-related types, otherwise False.

        Checks for:
        - image_url content type (OpenAI format)
        - Content items with type 'image_url'
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Check if any content item indicates vision content
                for content_item in content:
                    if isinstance(content_item, dict):
                        # Check for image_url field (direct image URL)
                        if "image_url" in content_item:
                            return True
                        # Check for type field indicating image content
                        content_type = content_item.get("type")
                        if content_type == "image_url":
                            return True
        return False

    @staticmethod
    def _parse_anthropic_native_content(
        content_blocks: List[Any],
    ) -> Tuple[str, List[ChatCompletionToolCallChunk], List[Any] | None]:
        """
        Parse Anthropic-native content blocks into OpenAI-compatible fields.

        Concatenates all text blocks, extracts tool_use blocks as tool_calls, and
        preserves thinking blocks when present.
        """
        (
            text_content,
            _citations,
            thinking_blocks,
            _reasoning_content,
            tool_calls,
            _web_search_results,
            _tool_results,
            _compaction_blocks,
        ) = AnthropicConfig().extract_response_content(completion_response={"content": content_blocks})
        return text_content, tool_calls, thinking_blocks

    @staticmethod
    def _normalize_anthropic_usage(usage: dict) -> dict:
        normalized = dict(usage)
        if "input_tokens" in usage and "prompt_tokens" not in usage:
            normalized["prompt_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage and "completion_tokens" not in usage:
            normalized["completion_tokens"] = usage["output_tokens"]
        if "total_tokens" not in normalized:
            normalized["total_tokens"] = normalized.get("prompt_tokens", 0) + normalized.get("completion_tokens", 0)
        return normalized

    @classmethod
    def _synthesize_choices_for_anthropic_native(cls, response_json: dict) -> dict:
        """
        Synthesize a `choices` array from an Anthropic-native Copilot response.

        Newer Copilot Claude models (e.g. opus-4.7, opus-4.8) return content
        blocks and `stop_reason` without an OpenAI-style `choices` array, and the
        max_tokens=1 probe returns no content at all. Returns the response
        unchanged when it already carries choices.

        See: https://github.com/BerriAI/litellm/issues/29391
        """
        if response_json.get("choices"):
            return response_json

        content = ""
        tool_calls: List[ChatCompletionToolCallChunk] = []
        thinking_blocks: List[Any] | None = None
        raw_content = response_json.get("content")
        if isinstance(raw_content, list):
            content, tool_calls, thinking_blocks = cls._parse_anthropic_native_content(raw_content)
        elif isinstance(raw_content, str):
            content = raw_content

        stop_reason = response_json.get("stop_reason")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        if tool_calls:
            finish_reason = "tool_calls"
        elif stop_reason in finish_reason_map:
            finish_reason = finish_reason_map[stop_reason]
        elif content:
            finish_reason = "stop"
        else:
            finish_reason = "length"

        message: dict = {
            "role": "assistant",
            "content": content if content or not tool_calls else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking_blocks:
            message["thinking_blocks"] = thinking_blocks

        synthesized = {
            **response_json,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            synthesized["usage"] = cls._normalize_anthropic_usage(usage)
        return synthesized

    def transform_parsed_response_dict(self, parsed_response: dict) -> dict:
        """
        Repair the OpenAI-SDK-parsed response on the handler path that bypasses
        transform_response. See: https://github.com/BerriAI/litellm/issues/30927
        """
        return self._synthesize_choices_for_anthropic_native(parsed_response)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: "ModelResponse",
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> "ModelResponse":
        try:
            response_json = raw_response.json()
        except Exception:
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

        if not response_json.get("choices"):
            response_json = self._synthesize_choices_for_anthropic_native(response_json)
            raw_response = httpx.Response(
                status_code=raw_response.status_code,
                headers=raw_response.headers,
                content=json.dumps(response_json).encode(),
            )

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
