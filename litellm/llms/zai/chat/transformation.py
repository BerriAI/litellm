from typing import List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE = "https://api.z.ai/api/paas/v4"

_REASONING_PARAMS = ("thinking", "reasoning_effort")


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
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "thinking",
            "reasoning_effort",
        ]

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in _REASONING_PARAMS:
                optional_params.setdefault("extra_body", {})[param] = value
            elif param in supported:
                optional_params[param] = value
        return optional_params
