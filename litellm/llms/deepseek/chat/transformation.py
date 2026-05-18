"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import _supports_factory

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    @staticmethod
    def _supports_reasoning_effort_level(model: str, level: str) -> bool:
        """Check whether the DeepSeek model supports a specific reasoning_effort
        level natively, per ``supports_{level}_reasoning_effort`` in
        ``model_prices_and_context_window.json``.

        Returns False for unknown models (safe fallback) — the caller will
        still emit ``thinking: {"type": "enabled"}`` so the model thinks,
        just without the explicit effort hint.
        """
        return _supports_factory(
            model=model,
            custom_llm_provider="deepseek",
            key=f"supports_{level}_reasoning_effort",
        )

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
        DeepSeek only supports `{"type": "enabled"}` - no budget_tokens like Anthropic.

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

        # Normalize reasoning_effort values to DeepSeek's supported levels.
        # OpenAI levels low/medium → DeepSeek "high"; xhigh → DeepSeek "max".
        if reasoning_effort is not None and reasoning_effort != "none":
            if reasoning_effort in ("low", "medium"):
                reasoning_effort = "high"
            elif reasoning_effort == "xhigh":
                reasoning_effort = "max"

        # Handle thinking parameter - only accept {"type": "enabled"}
        if thinking_value is not None:
            if (
                isinstance(thinking_value, dict)
                and thinking_value.get("type") == "enabled"
            ):
                optional_params["thinking"] = {"type": "enabled"}
                # Forward reasoning_effort natively only if the model declares
                # support for this specific level in model_prices_and_context_window.json.
                if (
                    reasoning_effort is not None
                    and reasoning_effort != "none"
                    and self._supports_reasoning_effort_level(model, reasoning_effort)
                ):
                    optional_params["reasoning_effort"] = reasoning_effort

        # Handle reasoning_effort alone (without explicit thinking dict)
        elif reasoning_effort is not None and reasoning_effort != "none":
            optional_params["thinking"] = {"type": "enabled"}
            if self._supports_reasoning_effort_level(model, reasoning_effort):
                optional_params["reasoning_effort"] = reasoning_effort

        return optional_params

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
