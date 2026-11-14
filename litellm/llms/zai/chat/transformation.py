from collections.abc import Mapping
from typing import Final

import litellm
from litellm.router_utils.reasoning_effort_capability import (
    declared_reasoning_efforts_for_model,
    nearest_declared_reasoning_effort,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"


def _accepted_effort(model: str, requested: object, drop_params: bool) -> object:
    declared: Final = declared_reasoning_efforts_for_model(model, "zai")
    if not isinstance(requested, str) or declared is None:
        return requested
    accepted: Final = nearest_declared_reasoning_effort(requested, declared)
    return None if drop_params and accepted not in declared else accepted


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

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

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

    def get_supported_openai_params(self, model: str) -> list[str]:
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

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.extend(("thinking", "reasoning_effort"))
        except Exception:
            pass

        return base_params

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: request pipeline adds parameters to the returned body
        supported_params: Final = self.get_supported_openai_params(model)
        sent: Final = {param: value for param, value in non_default_params.items() if param in supported_params}
        requested_effort: Final = sent.get("reasoning_effort")
        reasoning_effort: Final = (
            None
            if requested_effort is None
            else _accepted_effort(model, requested_effort, litellm.drop_params or drop_params)
        )
        return {
            **optional_params,
            **{param: value for param, value in sent.items() if param != "reasoning_effort"},
            **({} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}),
        }
