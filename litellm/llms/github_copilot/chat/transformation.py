from typing import Any, List, Optional, Tuple, Union, cast

import os

import httpx

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues
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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = (
            api_base
            or self.authenticator.get_api_base()
            or os.getenv("GITHUB_COPILOT_API_BASE")
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )
        try:
            dynamic_api_key = self.authenticator.get_api_key()
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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
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
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> "ModelResponse":
        """
        Override transform_response to handle newer Copilot models (e.g. claude-opus-4.7,
        claude-opus-4.8) that may return Anthropic-native format responses without
        the standard OpenAI `choices` array.

        When `choices` is missing or empty in the response, this method synthesizes
        a minimal valid `choices` structure from the available Anthropic-native fields
        (e.g. `content`, `stop_reason`) before delegating to the parent's response
        conversion logic.

        This prevents IndexError crashes when the response has no choices, which
        commonly occurs with max_tokens=1 on newer models.

        See: https://github.com/BerriAI/litellm/issues/29391
        """
        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers

        try:
            response_json = raw_response.json()
        except Exception:
            # If we can't parse JSON, fall back to parent behavior
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

        # Guard: if choices is missing or empty, synthesize from Anthropic-native fields
        if not response_json.get("choices"):
            content = ""
            # Try to extract text content from Anthropic-native response format
            if "content" in response_json and isinstance(
                response_json["content"], list
            ):
                for block in response_json["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break

            # Map Anthropic stop_reason to OpenAI finish_reason
            stop_reason = response_json.get("stop_reason", "max_tokens")
            finish_reason_map = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }
            finish_reason = finish_reason_map.get(stop_reason, "length")

            response_json["choices"] = [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ]

            # Normalize usage fields if in Anthropic format
            if "usage" in response_json:
                usage = response_json["usage"]
                if "input_tokens" in usage and "prompt_tokens" not in usage:
                    usage["prompt_tokens"] = usage["input_tokens"]
                if "output_tokens" in usage and "completion_tokens" not in usage:
                    usage["completion_tokens"] = usage["output_tokens"]
                if "total_tokens" not in usage:
                    usage["total_tokens"] = usage.get(
                        "prompt_tokens", 0
                    ) + usage.get("completion_tokens", 0)

        final_response_obj = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=response_json,
                model_response_object=model_response,
                hidden_params={"headers": raw_response.headers},
                _response_headers=dict(raw_response.headers),
            ),
        )

        return final_response_obj
