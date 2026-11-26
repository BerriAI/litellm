"""
Translates from OpenAI's `/v1/chat/completions` to DashScope's `/v1/chat/completions`
"""

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final, Literal, overload

from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import resolve_dashscope_openai_api_base

THINKING_OFF_REASONING_EFFORTS: Final = frozenset({"none", "disable"})


class DashScopeThinkingBody(TypedDict, total=False):
    enable_thinking: ReadOnly[bool]
    thinking_budget: ReadOnly[int]


def _dashscope_thinking_body(thinking: Mapping[str, object] | None, reasoning_effort: object) -> DashScopeThinkingBody:
    if thinking is not None:
        enabled: Final = thinking.get("type") != "disabled"
        budget: Final = thinking.get("budget_tokens")
        if enabled and isinstance(budget, int) and not isinstance(budget, bool):
            with_budget: Final[DashScopeThinkingBody] = {"enable_thinking": enabled, "thinking_budget": budget}
            return with_budget
        toggled: Final[DashScopeThinkingBody] = {"enable_thinking": enabled}
        return toggled
    if isinstance(reasoning_effort, str):
        from_effort: Final[DashScopeThinkingBody] = {
            "enable_thinking": reasoning_effort not in THINKING_OFF_REASONING_EFFORTS
        }
        return from_effort
    untouched: Final[DashScopeThinkingBody] = {}
    return untouched


class DashScopeChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base class contract returns a list
        return [
            *super().get_supported_openai_params(model=model),
            "thinking",
            "reasoning_effort",
        ]

    def _map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: dict return contract of OpenAIGPTConfig
        thinking: Final = as_str_mapping(non_default_params.get("thinking"))
        consumed: Final = frozenset({"thinking"} if thinking is None else {"thinking", "reasoning_effort"})
        supported_openai_params: Final = frozenset(self.get_supported_openai_params(model))
        passthrough_params: Final = MappingProxyType(
            {k: v for k, v in non_default_params.items() if k in supported_openai_params and k not in consumed}
        )
        native: Final = _dashscope_thinking_body(thinking, non_default_params.get("reasoning_effort"))
        if not native:
            return {**optional_params, **passthrough_params}
        existing_body: Final = as_str_mapping(optional_params.get("extra_body")) or MappingProxyType({})
        return {
            **optional_params,
            **passthrough_params,
            "extra_body": {**existing_body, **native},
        }

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
    ) -> Coroutine[object, object, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
        if is_async:
            return super()._transform_messages(messages=messages, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=messages, model=model, is_async=False)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = resolve_dashscope_openai_api_base(api_base)
        dynamic_api_key: Final = api_key or get_secret_str("DASHSCOPE_API_KEY")
        return resolved_api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return resolve_dashscope_openai_api_base(api_base)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        """
        If api_base is not provided, use the default DashScope /chat/completions endpoint.
        """
        return ensure_api_route(self._resolve_chat_api_base(api_base), "chat/completions")
