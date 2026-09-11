from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"
ZAI_REASONING_PARAMS: Final = frozenset(("thinking", "reasoning_effort"))


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
                return [*base_params, *sorted(ZAI_REASONING_PARAMS)]  # mutable-ok: base class returns a list
        except Exception:
            pass

        return base_params

    def _map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_openai_params: Final = frozenset(self.get_supported_openai_params(model))
        reasoning_params: Final = MappingProxyType(
            {k: v for k, v in non_default_params.items() if k in ZAI_REASONING_PARAMS and k in supported_openai_params}
        )
        passthrough_params: Final = MappingProxyType(
            {
                k: v
                for k, v in non_default_params.items()
                if k in supported_openai_params and k not in ZAI_REASONING_PARAMS
            }
        )
        if not reasoning_params:
            return {**optional_params, **passthrough_params}  # mutable-ok: base class returns a dict
        existing: Final = optional_params.get("extra_body")
        existing_body: Final = existing if isinstance(existing, Mapping) else MappingProxyType({})
        extra_body: Final = {  # mutable-ok: the OpenAI SDK json-encodes extra_body from a plain dict
            **existing_body,
            **reasoning_params,
        }
        return {  # mutable-ok: base class returns a dict
            **optional_params,
            **passthrough_params,
            "extra_body": extra_body,
        }
