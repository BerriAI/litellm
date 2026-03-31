"""
Translates from OpenAI's `/v1/chat/completions` to Moonshot AI's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class MoonshotChatConfig(OpenAIGPTConfig):
    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Moonshot text-only models don't support content in list format.
        Multimodal models (kimi-k2.5, kimi-latest, etc.) accept the
        standard OpenAI content array with non-text blocks (image_url,
        input_audio, video_url, file, etc.).

        If any message contains a non-text content part, skip flattening
        so the multimodal payload is preserved.
        """
        has_non_text = False
        for m in messages:
            _content = m.get("content")
            if _content and isinstance(_content, list):
                if any(c.get("type") != "text" for c in _content):
                    has_non_text = True
                    break

        if not has_non_text:
            messages = handle_messages_with_content_list_to_str_conversion(messages)

        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("MOONSHOT_API_BASE") or "https://api.moonshot.ai/v1"  # type: ignore
        dynamic_api_key = api_key or get_secret_str("MOONSHOT_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default Moonshot AI /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.moonshot.ai/v1"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for Moonshot AI models

        Moonshot AI limitations:
        - functions parameter is not supported (use tools instead)
        - tool_choice doesn't support "required" value
        - kimi-thinking-preview doesn't support tool calls at all
        """
        excluded_params: List[str] = ["functions"]

        # kimi-thinking-preview has additional limitations
        if "kimi-thinking-preview" in model:
            excluded_params.extend(["tools", "tool_choice"])

        base_openai_params = super().get_supported_openai_params(model=model)
        final_params: List[str] = []
        for param in base_openai_params:
            if param not in excluded_params:
                final_params.append(param)

        return final_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Moonshot AI parameters

        Handles Moonshot AI specific limitations:
        - tool_choice doesn't support "required" value
        - Temperature <0.3 limitation for n>1
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value

        ##########################################
        # temperature limitations
        # 1. `temperature` on KIMI API is [0, 1] but OpenAI is [0, 2]
        # 2. If temperature < 0.3 and n > 1, KIMI will raise an exception.
        #       If we enter this condition, we set the temperature to 0.3 as suggested by Moonshot AI
        ##########################################
        if "temperature" in optional_params:
            if optional_params["temperature"] > 1:
                optional_params["temperature"] = 1
            if optional_params["temperature"] < 0.3 and optional_params.get("n", 1) > 1:
                optional_params["temperature"] = 0.3
        return optional_params

    def fill_reasoning_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        Moonshot reasoning models require `reasoning_content` on every assistant
        message that contains tool_calls (multi-turn tool-calling flows).

        For each such message that is missing the field:
          1. Check if reasoning_content exists at the top level (for Pydantic models
             that have the attribute but don't support 'in' operator)
          2. Promote provider_specific_fields["reasoning_content"] if present and non-empty
             (this is where LiteLLM stores it from a previous response)
          3. Otherwise inject a single space — the minimum value the API accepts
        Messages that already carry the field, or are not assistant/tool-call messages,
        are appended as-is (no copy made).
        """
        result: List[AllMessageValues] = []
        for msg in messages:
            if (
                msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and not msg.get(
                    "reasoning_content"
                )  # Check using .get() which works for both dicts and Pydantic models
            ):
                patched = dict(cast(dict, msg))
                provider_fields = patched.get("provider_specific_fields") or {}
                stored = provider_fields.get("reasoning_content")
                if stored:
                    patched["reasoning_content"] = stored
                    # Remove the promoted key from provider_specific_fields to
                    # avoid sending the value twice in the serialised request body
                    cleaned_provider_fields = dict(provider_fields)
                    cleaned_provider_fields.pop("reasoning_content", None)
                    patched["provider_specific_fields"] = cleaned_provider_fields
                else:
                    litellm.verbose_logger.warning(
                        "Moonshot reasoning model: assistant tool-call message is missing "
                        "`reasoning_content`. Injecting a placeholder to satisfy API validation. "
                        "For best results, preserve `reasoning_content` from the original "
                        "assistant response when building multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
                result.append(cast(AllMessageValues, patched))
            else:
                result.append(msg)
        return result

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.
        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        # Add tool_choice="required" message if needed
        if optional_params.get("tool_choice", None) == "required":
            messages = self._add_tool_choice_required_message(
                messages=messages,
                optional_params=optional_params,
            )

        # Moonshot reasoning models: fill in reasoning_content before the API call
        if supports_reasoning(model=model, custom_llm_provider="moonshot"):
            messages = self.fill_reasoning_content(messages)

        # Call parent transform_request which handles _transform_messages
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _add_tool_choice_required_message(
        self, messages: List[AllMessageValues], optional_params: dict
    ) -> List[AllMessageValues]:
        """
        Add a message to the messages list to indicate that the tool choice is required.

        https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-tool_choice
        """
        messages.append(
            {
                "role": "user",
                "content": "Please select a tool to handle the current issue.",  # Usually, the Kimi large language model understands the intention to invoke a tool and selects one for invocation
            }
        )
        optional_params.pop("tool_choice")
        return messages
