import asyncio
import contextvars
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import (
    Final,
    cast,  # noqa: TID251  # casts moved verbatim from litellm/main.py
)

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.chat_completion_agentic_loop import maybe_run_chat_completion_agentic_loop
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.common_utils import get_completion_messages
from litellm.native.chat import OPTIONAL_PARAMS_ADAPTER, execute_chat, resolve_dispatched_chat_response
from litellm.native.request_support import resolve_control_options
from litellm.types.litellm_params import ControlOptions
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    get_llm_provider,
    get_non_default_completion_params,
    validate_and_fix_openai_messages,
    validate_and_fix_openai_tools,
    validate_and_fix_thinking_param,
    validate_chat_completion_tool_choice,
    validate_openai_optional_params,
)

_PLANNED_CHAT_ARGS: Final = frozenset(
    ("kwargs", "non_default_params", "control_options", "prompt_id", "prompt_variables")
)


_EXECUTE_CHAT_CODE: Final = execute_chat.__code__


_EXECUTE_CHAT_PARAMS: Final = (
    frozenset(_EXECUTE_CHAT_CODE.co_varnames[: _EXECUTE_CHAT_CODE.co_kwonlyargcount]) - _PLANNED_CHAT_ARGS
)


_CHAT_SURFACE_KWARGS: Final = (
    "fallbacks",
    "model_list",
    "prompt_id",
    "litellm_system_prompt",
    "ensure_alternating_roles",
    "user_continue_message",
    "assistant_continue_message",
    "mock_delay",
    "mock_timeout",
)


_TEXT_COMPLETION_PROVIDERS: Final = frozenset(
    ("text-completion-openai", "text-completion-codestral", "text-completion-inception")
)


@dataclass(frozen=True, slots=True)
class PlannedChat:
    model: str
    custom_llm_provider: str
    named: Mapping[str, object]
    kwargs: Mapping[str, object]
    non_default_params: Mapping[str, object]
    control_options: ControlOptions


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


_OBJECT_LIST: Final = TypeAdapter(list[object])


def _object_list(value: object) -> list[object] | None:  # mutable-ok: feeds validate_and_fix_openai_* which take a list
    return _OBJECT_LIST.validate_python(value) if isinstance(value, list) else None


def _tool_choice(value: object) -> str | dict[str, object] | None:  # mutable-ok: its validator takes a dict
    if isinstance(value, str):
        return value
    return OPTIONAL_PARAMS_ADAPTER.validate_python(value) if isinstance(value, dict) else None


def _stop(value: object) -> str | list[str] | None:  # mutable-ok: validate_openai_optional_params takes a list
    if isinstance(value, list):
        return [item for item in _OBJECT_LIST.validate_python(value) if isinstance(item, str)]
    return value if isinstance(value, str) else None


def _thinking(value: object) -> AnthropicThinkingParam | bool | None:
    if isinstance(value, dict):
        return cast(AnthropicThinkingParam, value)  # cast-ok: the caller's thinking dict, normalized by the validator
    return value if isinstance(value, bool) else None


def _chat_entry_points_wrapped() -> bool:
    """Tracers such as ddtrace wrap `litellm.completion`/`acompletion`, so bridged calls must go through them."""
    from litellm.chat_completions import dispatch

    return litellm.completion is not dispatch.completion or litellm.acompletion is not dispatch.acompletion


def _uses_chat_surface(request: Mapping[str, object]) -> bool:
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    from litellm.types.llms.openai import ToolParam

    tools: Final = request.get("tools")
    return (
        litellm.model_fallbacks is not None
        or _chat_entry_points_wrapped()
        or any(request.get(key) is not None for key in _CHAT_SURFACE_KWARGS)
        or (
            isinstance(tools, list)
            and LiteLLM_Proxy_MCP_Handler.should_use_litellm_mcp_gateway(
                tools=cast(Iterable[ToolParam], tools)  # cast-ok: OpenAI-format tools from the caller's request
            )
        )
    )


def plan_chat(request: Mapping[str, object], logging_obj: LiteLLMLoggingObj) -> PlannedChat | None:
    model: Final = request.get("model")
    if not isinstance(model, str) or _uses_chat_surface(request):
        return None
    from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

    custom_llm_provider: Final = _optional_str(request.get("custom_llm_provider"))
    api_base: Final = _optional_str(request.get("api_base"))
    _, resolved_provider, _, _ = get_llm_provider(
        model=model, custom_llm_provider=custom_llm_provider, api_base=api_base
    )
    if resolved_provider in _TEXT_COMPLETION_PROVIDERS:
        return None
    kwargs: Final = {key: value for key, value in request.items() if key not in _EXECUTE_CHAT_PARAMS}
    tools: Final = validate_and_fix_openai_tools(tools=_object_list(request.get("tools")))
    messages: Final = get_completion_messages(
        messages=validate_and_fix_openai_messages(messages=_object_list(request.get("messages")) or []),
        ensure_alternating_roles=False,
        user_continue_message=None,
        assistant_continue_message=None,
    )
    non_default_params: Final = get_non_default_completion_params(kwargs=kwargs)
    AnthropicCacheControlHook.maybe_seed_default_injection_points(
        non_default_params=non_default_params,
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        tools=tools,
        enable_prompt_caching=request.get("enable_prompt_caching") is True,
        api_base=api_base,
    )
    if logging_obj.should_run_prompt_management_hooks(non_default_params=non_default_params, tools=tools):
        return None
    named: Final = {key: value for key, value in request.items() if key in _EXECUTE_CHAT_PARAMS}
    tool_choice: Final = validate_chat_completion_tool_choice(
        tool_choice=_tool_choice(request.get("tool_choice")), model=model
    )
    return PlannedChat(
        model=model,
        custom_llm_provider=resolved_provider,
        named=MappingProxyType(
            {
                **named,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "stop": validate_openai_optional_params(stop=_stop(request.get("stop"))),
                "thinking": validate_and_fix_thinking_param(thinking=_thinking(request.get("thinking"))),
            }
        ),
        kwargs=MappingProxyType(kwargs),
        non_default_params=MappingProxyType(non_default_params),
        control_options=resolve_control_options(kwargs, model),
    )


def _run_planned_chat(plan: PlannedChat, is_async: bool) -> ModelResponse | CustomStreamWrapper:
    return execute_chat(
        **plan.named,  # pyright: ignore[reportArgumentType]  # named holds the caller's validated OpenAI-format params
        kwargs={**plan.kwargs, "acompletion": is_async},
        non_default_params=dict(plan.non_default_params),
        control_options=plan.control_options,
        prompt_id=None,
        prompt_variables=None,
    )


def execute_planned_chat(plan: PlannedChat) -> ModelResponse | CustomStreamWrapper:
    return _run_planned_chat(plan, is_async=False)


async def _settled_chat_response(
    plan: PlannedChat, loop: asyncio.AbstractEventLoop
) -> ModelResponse | CustomStreamWrapper:
    pending: Final[object] = await loop.run_in_executor(
        None, partial(contextvars.copy_context().run, partial(_run_planned_chat, plan, True))
    )
    if asyncio.iscoroutine(pending):
        return await resolve_dispatched_chat_response(pending)
    if isinstance(pending, (ModelResponse, CustomStreamWrapper)):
        return pending
    raise TypeError(f"chat provider returned {type(pending).__name__}, expected a ModelResponse or a stream")


async def _with_agentic_loop(plan: PlannedChat, response: ModelResponse) -> ModelResponse | CustomStreamWrapper:
    looped: Final = await maybe_run_chat_completion_agentic_loop(
        response=response,
        model=plan.model,
        messages=_object_list(plan.named["messages"]) or [],
        optional_params={
            key: value
            for key, value in plan.named.items()
            if value is not None and key not in ("model", "messages", "stream", "deployment_id")
        },
        kwargs=dict(plan.kwargs),
        logging_obj=plan.kwargs.get("litellm_logging_obj"),
        custom_llm_provider=plan.custom_llm_provider,
        stream=plan.named.get("stream") is True,
    )
    return response if looped is None else looped


async def aexecute_planned_chat(plan: PlannedChat) -> ModelResponse | CustomStreamWrapper:
    loop: Final = asyncio.get_event_loop()
    response: Final = await _settled_chat_response(plan, loop)
    if isinstance(response, CustomStreamWrapper):
        response.set_logging_event_loop(loop=loop)
        return response
    return await _with_agentic_loop(plan, response)
