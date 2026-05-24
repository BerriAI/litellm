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

        is_always_on_reasoner = self._is_always_on_reasoner(model=model)

        # Handle thinking parameter - accepts {"type": "enabled"} or {"type": "disabled"}.
        # Guard: deepseek-reasoner has always-on thinking and rejects {"type": "disabled"}.
        if thinking_value is not None:
            if isinstance(thinking_value, dict) and thinking_value.get("type") in (
                "enabled",
                "disabled",
            ):
                thinking_type = thinking_value.get("type")
                if thinking_type == "disabled" and is_always_on_reasoner:
                    pass  # no-op: deepseek-reasoner rejects {"type": "disabled"}
                else:
                    optional_params["thinking"] = {"type": thinking_type}

        # Handle reasoning_effort when thinking was not explicitly provided.
        if reasoning_effort is not None and "thinking" not in optional_params:
            if reasoning_effort == "none":
                # Only send thinking: disabled on V4 opt-in models.
                # deepseek-reasoner/R1 have always-on thinking and reject {"type": "disabled"}.
                if not is_always_on_reasoner:
                    optional_params["thinking"] = {"type": "disabled"}
            else:
                # Normalize to DeepSeek's two supported values
                normalized = "max" if reasoning_effort in ("max", "xhigh") else "high"
                optional_params["thinking"] = {"type": "enabled"}
                # Only forward reasoning_effort on V4 opt-in models.
                # deepseek-reasoner/R1 have always-on thinking and don't accept the reasoning_effort field.
                if not is_always_on_reasoner:
                    optional_params["reasoning_effort"] = normalized

        # When both thinking=enabled and reasoning_effort are provided for V4 models,
        # also forward the effort level (not dropped by the thinking branch above).
        if (
            reasoning_effort is not None
            and reasoning_effort != "none"
            and optional_params.get("thinking", {}).get("type") == "enabled"
            and "reasoning_effort" not in optional_params
            and not is_always_on_reasoner
        ):
            optional_params["reasoning_effort"] = (
                "max" if reasoning_effort in ("max", "xhigh") else "high"
            )

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
                    litellm.verbose_logger.warning(
                        "DeepSeek thinking mode: assistant message is missing "
                        "`reasoning_content` and none was saved in "
                        "`provider_specific_fields`. A single-space placeholder "
                        "is being injected to satisfy API validation, but the "
                        "model will receive a blank reasoning chain for this turn, "
                        "which may silently degrade multi-turn response quality. "
                        "Preserve `reasoning_content` from the original assistant "
                        "response when building multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
                result.append(cast(AllMessageValues, patched))
            else:
                result.append(msg)
        return result

    @staticmethod
    def _is_always_on_reasoner(model: str) -> bool:
        """
        Returns True for models with always-on thinking (deepseek-reasoner, R1 variants).
        These models reject reasoning_effort, thinking: {"type": "disabled"}, and
        require reasoning_content on every assistant message unconditionally.

        Uses the litellm model registry (supports_reasoning field) as the primary
        signal — deepseek-reasoner and R1 variants have supports_reasoning: true while
        V4 opt-in models (deepseek-chat, deepseek-v3, etc.) do not. Falls back to
        string-pattern matching for unregistered or custom-deployment model names.
        """
        # Primary: registry-based check
        try:
            from litellm.utils import supports_reasoning

            if supports_reasoning(model=model, custom_llm_provider="deepseek"):
                return True
        except Exception:
            pass
        # Fallback: string patterns for unregistered variants / custom deployments
        m = model.lower()
        return "reasoner" in m or "-r1" in m or "/r1" in m

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
        Returns True when thinking mode is active for this request:
          - deepseek-reasoner/R1: always-on thinking, no explicit param needed
          - V4 opt-in models: only when user explicitly passed thinking={"type": "enabled"}
        """
        if self._is_always_on_reasoner(model=model):
            return True  # deepseek-reasoner always has thinking on
        return (optional_params.get("thinking") or {}).get("type") == "enabled"

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

        Runs when thinking mode is active: always for deepseek-reasoner (always-on),
        and for V4 opt-in models only when the user explicitly enabled thinking.
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
