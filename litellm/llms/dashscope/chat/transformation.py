"""
Translates from OpenAI's `/v1/chat/completions` to DashScope's `/v1/chat/completions`
"""

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Any, Final, Literal, overload

from typing_extensions import ReadOnly, TypedDict

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

THINKING_OFF_REASONING_EFFORTS: Final = frozenset({"none", "disable"})
DASHSCOPE_THINKING_PARAMS: Final = frozenset({"thinking", "reasoning_effort"})


class DashScopeThinkingBody(TypedDict, total=False):
    enable_thinking: ReadOnly[bool]
    thinking_budget: ReadOnly[int]
    reasoning_effort: ReadOnly[str]


def _dashscope_thinking_body(thinking: object, reasoning_effort: object) -> DashScopeThinkingBody:
    if isinstance(thinking, dict):
        enabled: Final = thinking.get("type") != "disabled"
        budget: Final = thinking.get("budget_tokens")
        if enabled and isinstance(budget, int) and not isinstance(budget, bool):
            with_budget: Final[DashScopeThinkingBody] = {"enable_thinking": enabled, "thinking_budget": budget}
            return with_budget
        toggled: Final[DashScopeThinkingBody] = {"enable_thinking": enabled}
        return toggled
    if isinstance(reasoning_effort, str):
        if reasoning_effort in THINKING_OFF_REASONING_EFFORTS:
            off: Final[DashScopeThinkingBody] = {"enable_thinking": False}
            return off
        with_effort: Final[DashScopeThinkingBody] = {"enable_thinking": True, "reasoning_effort": reasoning_effort}
        return with_effort
    untouched: Final[DashScopeThinkingBody] = {}
    return untouched


class DashScopeChatConfig(OpenAIGPTConfig):
    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for DashScope.
        DashScope supports cache_control - don't strip it.
        """
        return messages, tools

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:
        if is_async:
            return super()._transform_messages(messages=messages, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=messages, model=model, is_async=False)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = (
            api_base or get_secret_str("DASHSCOPE_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        dynamic_api_key: Final = api_key or get_secret_str("DASHSCOPE_API_KEY")
        return api_base, dynamic_api_key

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        If api_base is not provided, use the default DashScope /chat/completions endpoint.
        """
        resolved_api_base: Final = self._resolve_chat_api_base(api_base)
        if resolved_api_base.endswith("/chat/completions"):
            return resolved_api_base
        return f"{resolved_api_base}/chat/completions"

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = super().get_supported_openai_params(model)
        return [*base_params, "thinking", "reasoning_effort"]  # mutable-ok: inherited list contract

    def _map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_openai_params: Final = frozenset(self.get_supported_openai_params(model))
        passthrough_params: Final = MappingProxyType(
            {
                k: v
                for k, v in non_default_params.items()
                if k in supported_openai_params and k not in DASHSCOPE_THINKING_PARAMS
            }
        )
        native: Final = _dashscope_thinking_body(
            non_default_params.get("thinking"), non_default_params.get("reasoning_effort")
        )
        if not native:
            return {**optional_params, **passthrough_params}  # mutable-ok: dict return contract of OpenAIGPTConfig
        existing: Final = optional_params.get("extra_body")
        existing_body: Final = existing if isinstance(existing, Mapping) else MappingProxyType({})
        return {  # mutable-ok: dict return contract of OpenAIGPTConfig
            **optional_params,
            **passthrough_params,
            "extra_body": {**existing_body, **native},  # mutable-ok: the OpenAI SDK json-encodes extra_body from a dict
        }
