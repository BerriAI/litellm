from typing import Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"

_REASONING_PARAMS = ("thinking", "reasoning_effort")


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = [
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
                base_params.extend(_REASONING_PARAMS)
        except Exception:
            pass

        return base_params

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param not in supported:
                continue
            if param in _REASONING_PARAMS:
                optional_params.setdefault("extra_body", {})[param] = value
            else:
                optional_params[param] = value
        return optional_params
