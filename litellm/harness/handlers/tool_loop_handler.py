"""In-process tool-calling loop handler."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.harness.context import SessionContext
from litellm.harness.errors import CapabilityUnsupported
from litellm.harness.handlers.base import BaseHarnessHandler
from litellm.harness.types import Approval, Event, Reasoning, Text, ToolCall, ToolResult
from litellm.llms.base_llm.harness.transformation import HarnessTurnError
from litellm.llms.tool_loop.harness.transformation import (
    TOOL_LOOP_MAX_MODEL_CALLS,
    FunctionTool,
    ToolLoopHarnessConfig,
    completion_kwargs,
    function_tool,
)
from litellm.types.completion import ChatCompletionMessageParam
from litellm.types.utils import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
    ModelResponse,
)


@dataclass(frozen=True, slots=True)
class _FunctionToolCall:
    id: str
    name: str
    arguments: str


AsyncCompletion: TypeAlias = Callable[..., Awaitable[ModelResponse]]
_ARGUMENTS_ADAPTER: Final = TypeAdapter(dict[str, object])
_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_MODEL_RESPONSE_ADAPTER: Final = TypeAdapter(ModelResponse)
_RAW_DECODE_ADAPTER: Final = TypeAdapter(tuple[object, int])
_JSON_DECODER: Final = json.JSONDecoder()
_HISTORY_ADAPTER: Final = TypeAdapter(list[dict[str, object]])


class _Usage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


_USAGE_ADAPTER: Final = TypeAdapter(_Usage)


class _AwaitableObject(Protocol):
    def __await__(self) -> Generator[object, None, object]: ...


async def _await_tool_result(result: _AwaitableObject) -> object:
    return await result


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    output: str
    is_error: bool


def _normalize_tool_call(
    call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
) -> _FunctionToolCall:
    if isinstance(call, ChatCompletionMessageToolCall):
        return _FunctionToolCall(
            id=call.id,
            name=call.function.name or "",
            arguments=call.function.arguments,
        )
    return _FunctionToolCall(id=call.id, name=call.custom.name, arguments=call.custom.input)


def _parse_arguments(raw: str) -> tuple[dict[str, object], str | None]:  # mutable-ok: event input uses a dict
    text: Final = raw.strip()
    try:
        raw_decoded: object = _JSON_DECODER.raw_decode(text)
    except json.JSONDecodeError as error:
        return {}, f"{type(error).__name__}: {error}"
    parsed, end = _RAW_DECODE_ADAPTER.validate_python(raw_decoded)
    if text[end:].strip():
        return {}, "JSONDecodeError: Extra data after tool arguments"
    try:
        arguments: Final = _ARGUMENTS_ADAPTER.validate_python(parsed)
    except ValidationError as error:
        if not isinstance(parsed, dict):
            return {}, "ValueError: tool arguments must be a JSON object"
        return {}, f"{type(error).__name__}: {error}"
    return arguments, None


def _cost_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        cost: Final = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) else None


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _response_cost(response: ModelResponse) -> float:
    hidden_params_value: Final[object] = getattr(response, "_hidden_params", {})
    hidden_params: Final = _as_mapping(hidden_params_value)
    if hidden_params is not None:
        additional_headers: Final = _as_mapping(hidden_params.get("additional_headers"))
        if additional_headers is not None:
            header_cost: Final = _cost_value(additional_headers.get("llm_provider-x-litellm-response-cost"))
            if header_cost is not None:
                return header_cost
        hidden_cost: Final = _cost_value(hidden_params.get("response_cost"))
        if hidden_cost is not None:
            return hidden_cost
    try:
        calculated_cost: Final = litellm.completion_cost(completion_response=response)
    except Exception:
        return 0.0
    return _cost_value(calculated_cost) or 0.0


async def _approval_error(approval: Approval | None) -> str | None:
    if approval is None:
        return None
    allowed, reason = await approval.wait()
    return None if allowed else f"denied: {reason}"


async def _tool_outcome(
    tool: FunctionTool | None,
    tool_name: str,
    arguments: dict[str, object],
    parse_error: str | None,
    approval_error: str | None,
) -> _ToolOutcome:
    if parse_error is not None:
        return _ToolOutcome(output=parse_error, is_error=True)
    if approval_error is not None:
        return _ToolOutcome(output=approval_error, is_error=True)
    if tool is None:
        return _ToolOutcome(output=f"ValueError: unknown tool {tool_name!r}", is_error=True)
    try:
        result: Final = await _execute_tool(tool, arguments)
        output: Final = result if isinstance(result, str) else json.dumps(result, default=str)
        return _ToolOutcome(output=output, is_error=False)
    except Exception as error:
        return _ToolOutcome(output=f"{type(error).__name__}: {error}", is_error=True)


def _record_usage(ctx: SessionContext, response: ModelResponse) -> None:
    try:
        usage_value: Final[object] = getattr(response, "usage", None)
        usage: Final = _USAGE_ADAPTER.validate_python(usage_value)
        input_tokens: Final = usage.prompt_tokens or 0
        output_tokens: Final = usage.completion_tokens or 0
        ctx.calls += 1  # rebind-ok: SessionContext is the runtime's per-session usage sink
        ctx.input_tokens += input_tokens  # rebind-ok: SessionContext is the runtime's per-session usage sink
        ctx.output_tokens += output_tokens  # rebind-ok: SessionContext is the runtime's per-session usage sink
        ctx.cost += _response_cost(response)  # rebind-ok: SessionContext is the runtime's per-session usage sink
    except Exception:
        return


async def _execute_tool(tool: FunctionTool, arguments: Mapping[str, object]) -> object:
    validated_model: Final = tool.args_model.model_validate(arguments)
    values_object: Final[object] = validated_model.model_dump()
    validated: Final = _MAPPING_ADAPTER.validate_python(values_object)
    parameters: Final = tuple(inspect.signature(tool.fn).parameters.values())
    positional_args: Final = tuple(
        validated[parameter.name] for parameter in parameters if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    )
    keyword_args: Final[dict[str, object]] = {  # mutable-ok: tool calls need keyword arguments
        parameter.name: validated[parameter.name]
        for parameter in parameters
        if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
    }
    if inspect.iscoroutinefunction(tool.fn):
        async_result: Final[object] = tool.fn(*positional_args, **keyword_args)
        if inspect.isawaitable(async_result):
            return await _await_tool_result(async_result)
        return async_result
    sync_result: Final[object] = await asyncio.to_thread(tool.fn, *positional_args, **keyword_args)
    if inspect.isawaitable(sync_result):
        return await _await_tool_result(sync_result)
    return sync_result


async def _default_acompletion(**kwargs: object) -> ModelResponse:  # kwargs-ok: provider-specific completion options
    response: Final[object] = await litellm.acompletion(**kwargs)
    return _MODEL_RESPONSE_ADAPTER.validate_python(response)


class ToolLoopHandler(BaseHarnessHandler):
    def __init__(
        self,
        config: ToolLoopHarnessConfig,
        acompletion: AsyncCompletion | None = None,
    ) -> None:
        super().__init__(config)  # pyright: ignore[reportUnknownMemberType]  # base handler config is unparameterized
        self._config = config
        self._acompletion = acompletion if acompletion is not None else _default_acompletion
        self._messages: tuple[ChatCompletionMessageParam, ...] = ()
        self._tools: Mapping[str, FunctionTool] = MappingProxyType({})
        self._tool_specs: tuple[ChatCompletionToolParam, ...] = ()
        self._completion_kwargs: Mapping[str, object] = MappingProxyType({})

    async def start(self, ctx: SessionContext) -> None:
        self._config.validate_environment(ctx)
        tools: Final = tuple(function_tool(fn) for fn in ctx.tools)
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Harness.TOOL_LOOP tool names must be unique")
        self._tools = MappingProxyType({tool.name: tool for tool in tools})
        self._tool_specs = tuple(tool.spec for tool in tools)
        self._completion_kwargs = MappingProxyType(completion_kwargs(ctx))
        if not self._messages and ctx.instructions:
            self._messages = ({"role": "system", "content": ctx.instructions},)

    def native_session_id(self) -> str | None:
        return None

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        raise CapabilityUnsupported("Harness.TOOL_LOOP does not support resume")

    async def history(
        self, ctx: SessionContext
    ) -> list[dict[str, object]]:  # mutable-ok: public API returns copied message dictionaries
        history_object: Final[object] = copy.deepcopy(list(self._messages))
        return _HISTORY_ADAPTER.validate_python(history_object)  # pyright: ignore[reportIncompatibleMethodOverride]  # base history uses Any

    async def stop(self, ctx: SessionContext) -> None:
        return None

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        ctx.final_text = ""  # rebind-ok: SessionContext is the runtime's per-turn result sink
        ctx.output_json = None  # rebind-ok: SessionContext is the runtime's per-turn result sink
        user_message: Final[ChatCompletionMessageParam] = {"role": "user", "content": prompt}
        self._messages = (*self._messages, user_message)
        for _ in range(TOOL_LOOP_MAX_MODEL_CALLS):
            messages: list[ChatCompletionMessageParam] = copy.deepcopy(  # mutable-ok: acompletion takes list messages
                list(self._messages)
            )
            tool_specs: list[ChatCompletionToolParam] = copy.deepcopy(  # mutable-ok: acompletion takes tool list
                list(self._tool_specs)
            )
            request_kwargs: dict[str, object] = {  # mutable-ok: acompletion takes keyword arguments
                key: value for key, value in self._completion_kwargs.items() if key not in {"messages", "tools"}
            }
            kwargs: dict[str, object] = {  # mutable-ok: acompletion takes keyword arguments
                **request_kwargs,
                "messages": messages,
                **({"tools": tool_specs} if tool_specs else {}),
            }
            response = await self._acompletion(**kwargs)
            _record_usage(ctx, response)
            message = response.choices[0].message
            reasoning_value: object = getattr(message, "reasoning_content", None)
            reasoning = reasoning_value if isinstance(reasoning_value, str) else None
            content = message.content
            if reasoning:
                yield Reasoning(reasoning)
            if content:
                yield Text(content)
            tool_calls = message.tool_calls or ()
            if not tool_calls:
                final_text = content or ""
                ctx.final_text = final_text  # rebind-ok: SessionContext is the runtime's per-turn result sink
                ctx.output_json = content if ctx.output is not None else None  # rebind-ok: per-turn output sink
                final_message: ChatCompletionMessageParam = {
                    "role": "assistant",
                    "content": content,
                }
                self._messages = (*self._messages, final_message)
                return
            normalized_calls = tuple(_normalize_tool_call(call) for call in tool_calls)
            assistant_message: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in normalized_calls
                ],
            }
            self._messages = (*self._messages, assistant_message)
            for call in normalized_calls:
                arguments, parse_error = _parse_arguments(call.arguments)
                yield ToolCall(
                    id=call.id,
                    name=call.name,
                    native_name=call.name,
                    input=arguments,
                    builtin=False,
                )
                approval = (
                    Approval(tool=call.name, input=arguments)
                    if parse_error is None and ctx.permissions == "ask"
                    else None
                )
                if approval is not None:
                    yield approval
                approval_error = await _approval_error(approval)
                tool = self._tools.get(call.name)
                outcome = await _tool_outcome(
                    tool,
                    call.name,
                    arguments,
                    parse_error,
                    approval_error,
                )
                yield ToolResult(id=call.id, output=outcome.output, is_error=outcome.is_error)
                tool_message: ChatCompletionMessageParam = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": outcome.output,
                }
                self._messages = (*self._messages, tool_message)
        raise HarnessTurnError(f"Harness.TOOL_LOOP exceeded {TOOL_LOOP_MAX_MODEL_CALLS} model calls in one turn")
