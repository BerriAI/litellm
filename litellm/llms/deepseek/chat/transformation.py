"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
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


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to DeepSeek params.

        Handles `thinking` and `reasoning_effort` parameters for DeepSeek reasoner models.
        DeepSeek supports `{"type": "enabled"}` and `{"type": "disabled"}` for thinking,
        and `reasoning_effort` values of `"high"` or `"max"` for V4 models.

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Pop thinking/reasoning_effort from optional_params first (parent may have added them)
        # Then re-add only if valid for DeepSeek
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Handle thinking parameter - the latest version accepts both "enabled and "disabled" as dicts, so we check for that first
        if thinking_value is not None:                                                                                                                                                                               
            if (                                                                                                                                                                                                     
                isinstance(thinking_value, dict)                                                                                                                                                                     
                and thinking_value.get("type") in ("enabled", "disabled")
            ):
                optional_params["thinking"] = {"type": thinking_value.get("type")}

        # Handle reasoning_effort - map to thinking enabled

        elif reasoning_effort is not None:
            if reasoning_effort == "none":
                # Only send thinking: disabled on V4 opt-in models.
                # deepseek-reasoner/R1 have always-on thinking and reject {"type": "disabled"}.
                if not supports_reasoning(model=model, custom_llm_provider="deepseek"):
                    optional_params["thinking"] = {"type": "disabled"}
            else:
                # Normalize to DeepSeek's two supported values
                normalized = "max" if reasoning_effort in ("max", "xhigh") else "high"
                optional_params["thinking"] = {"type": "enabled"}
                # Only forward reasoning_effort on V4 opt-in models.
                # deepseek-reasoner/R1 have supports_reasoning=True but don't accept reasoning_effort field.
                if not supports_reasoning(model=model, custom_llm_provider="deepseek"):
                    optional_params["reasoning_effort"] = normalized

        return optional_params

    def _fill_reasoning_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        DeepSeek thinking mode requires `reasoning_content` to be passed back on
        every assistant message in multi-turn conversations. If it is missing,
        the API returns:
          "The reasoning_content in the thinking mode must be passed back to the API."

        For each assistant message that is missing `reasoning_content`:
          1. Promote it from `provider_specific_fields["reasoning_content"]` if present
             (LiteLLM stores provider-specific response fields there).
          2. Otherwise inject a single space — the minimum value the API accepts.
        """
        result: List[AllMessageValues] = []
        for msg in messages:
            if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
                patched = dict(cast(dict, msg))
                provider_fields = patched.get("provider_specific_fields") or {}
                stored = provider_fields.get("reasoning_content")
                if stored:
                    patched["reasoning_content"] = stored
                    cleaned = dict(provider_fields)
                    cleaned.pop("reasoning_content", None)
                    patched["provider_specific_fields"] = cleaned
                else:
                    litellm.verbose_logger.debug(
                        "DeepSeek thinking mode: assistant message is missing "
                        "`reasoning_content`. Injecting a placeholder to satisfy "
                        "API validation. For best results, preserve "
                        "`reasoning_content` from the original assistant response "
                        "when building multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
                result.append(cast(AllMessageValues, patched))
            else:
                result.append(msg)
        return result

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]: ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        DeepSeek does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _thinking_mode_active(self, model: str, optional_params: dict) -> bool:
        """
        Returns True only when thinking mode is actually active for this request:
          - model supports reasoning (capability check)
          - user explicitly passed thinking={"type": "enabled"} (opt-in check)
        """
        return (
            supports_reasoning(model=model, custom_llm_provider="deepseek")
            and (optional_params.get("thinking") or {}).get("type") == "enabled"
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Ensures `reasoning_content` is forwarded on assistant messages for
        multi-turn thinking-mode conversations (issue #28045).

        Only runs when thinking mode is actually active - guarded by both
        supports_reasoning() (model capability) and optional_params["thinking"]
        (user explicitly enabled it), preventing spurious injection on models
        like deepseek-v3.2 that support thinking as opt-in but not always-on.
        """
        if self._thinking_mode_active(model=model, optional_params=optional_params):
            messages = self._fill_reasoning_content(messages)
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Async equivalent of transform_request — applies the same reasoning_content
        fix for multi-turn thinking-mode conversations.
        """
        if self._thinking_mode_active(model=model, optional_params=optional_params):
            messages = self._fill_reasoning_content(messages)
        return await super().async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
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
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.deepseek.com/beta"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
