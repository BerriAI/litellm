from typing import List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE = "https://api.z.ai/api/paas/v4"


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        import litellm

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.extend(["thinking", "reasoning_effort"])
        except Exception:
            pass

        return base_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map Anthropic-style thinking params to Z.ai's thinking format.

        Z.ai supports a ``thinking`` dict with ``type`` and ``clear_thinking``::

            {"type": "enabled", "clear_thinking": false}

        This translates LiteLLM's standardized ``thinking`` param (Anthropic
        format with ``budget_tokens``) and ``reasoning_effort`` into Z.ai's
        format.  ``budget_tokens`` is ignored since Z.ai does not support it.

        ``clear_thinking`` controls Preserved Thinking mode — when false, the
        model retains reasoning across turns instead of re-deriving from
        scratch.  It defaults to false on the Coding Plan endpoint.

        Reference: https://docs.z.ai/guides/capabilities/thinking-mode
        """
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        if thinking_value is not None:
            if isinstance(thinking_value, dict):
                thinking_type = thinking_value.get("type")
                if thinking_type in ("enabled", "disabled"):
                    zai_thinking: dict = {"type": thinking_type}
                    if "clear_thinking" in thinking_value:
                        zai_thinking["clear_thinking"] = thinking_value[
                            "clear_thinking"
                        ]
                    optional_params["thinking"] = zai_thinking
            elif isinstance(thinking_value, bool):
                optional_params["thinking"] = {
                    "type": "enabled" if thinking_value else "disabled"
                }
        elif reasoning_effort is not None:
            if reasoning_effort == "none":
                optional_params["thinking"] = {"type": "disabled"}
            else:
                optional_params["thinking"] = {"type": "enabled"}

        return optional_params
