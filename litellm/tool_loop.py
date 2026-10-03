"""Client-side tool-calling loop helpers for litellm.completion."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Final, TypeAlias

from typing_extensions import TypedDict, Unpack

from litellm.constants import DEFAULT_TOOL_LOOP_MAX_ROUNDS
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Message, ModelResponse

ToolExecutor: TypeAlias = Callable[[ChatCompletionMessageToolCall], ChatCompletionToolMessage]
AsyncToolExecutor: TypeAlias = Callable[[ChatCompletionMessageToolCall], Awaitable[ChatCompletionToolMessage]]


class _ToolLoopCompletionKwargs(TypedDict, total=False, extra_items=object):
    """Extra keywords forwarded verbatim to ``litellm.completion``, which owns their contract."""


class ToolLoopMaxRoundsExceeded(RuntimeError):
    max_rounds: int

    def __init__(self, max_rounds: int) -> None:
        self.max_rounds = max_rounds
        super().__init__(f"model still requested tool calls on round {max_rounds} of {max_rounds}")


def _validate_tool_loop_args(max_rounds: int, completion_kwargs: _ToolLoopCompletionKwargs) -> None:
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    if completion_kwargs.get("stream"):
        raise ValueError("run_tool_loop requires whole responses; stream=True is not supported")


def _assistant_tool_call(tool_call: ChatCompletionMessageToolCall) -> ChatCompletionAssistantToolCall:
    return ChatCompletionAssistantToolCall(
        id=tool_call.id,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=tool_call.function.name, arguments=tool_call.function.arguments
        ),
    )


def _assistant_message(
    message: Message, tool_calls: tuple[ChatCompletionMessageToolCall, ...]
) -> ChatCompletionAssistantMessage:
    assistant_tool_calls: Final = [_assistant_tool_call(tc) for tc in tool_calls]
    thinking_blocks: Final = getattr(message, "thinking_blocks", None)
    if thinking_blocks is not None:
        return ChatCompletionAssistantMessage(
            role="assistant",
            content=message.content,
            tool_calls=assistant_tool_calls,
            thinking_blocks=thinking_blocks,
        )
    return ChatCompletionAssistantMessage(role="assistant", content=message.content, tool_calls=assistant_tool_calls)


def _function_tool_calls(message: Message) -> tuple[ChatCompletionMessageToolCall, ...]:
    return tuple(
        tool_call for tool_call in message.tool_calls or () if isinstance(tool_call, ChatCompletionMessageToolCall)
    )


def _run_tool_loop_rounds(
    *,
    model: str,
    history: tuple[AllMessageValues, ...],
    tools: Sequence[ChatCompletionToolParam],
    execute_tool: ToolExecutor,
    max_rounds: int,
    rounds_left: int,
    **completion_kwargs: Unpack[_ToolLoopCompletionKwargs],
) -> str | None:
    import litellm

    response: Final = litellm.completion(model=model, messages=list(history), tools=list(tools), **completion_kwargs)
    if not isinstance(response, ModelResponse):
        raise TypeError(f"run_tool_loop requires completion to return a ModelResponse, got {type(response).__name__}")
    message: Final = response.choices[0].message
    if not message.tool_calls:
        return message.content
    if rounds_left == 1:
        raise ToolLoopMaxRoundsExceeded(max_rounds)
    tool_calls: Final = _function_tool_calls(message)
    tool_results: Final = tuple(execute_tool(tool_call) for tool_call in tool_calls)
    return _run_tool_loop_rounds(
        model=model,
        history=(*history, _assistant_message(message, tool_calls), *tool_results),
        tools=tools,
        execute_tool=execute_tool,
        max_rounds=max_rounds,
        rounds_left=rounds_left - 1,
        **completion_kwargs,
    )


async def _arun_tool_loop_rounds(
    *,
    model: str,
    history: tuple[AllMessageValues, ...],
    tools: Sequence[ChatCompletionToolParam],
    execute_tool: AsyncToolExecutor,
    max_rounds: int,
    rounds_left: int,
    **completion_kwargs: Unpack[_ToolLoopCompletionKwargs],
) -> str | None:
    import litellm

    response: Final = await litellm.acompletion(
        model=model, messages=list(history), tools=list(tools), **completion_kwargs
    )
    if not isinstance(response, ModelResponse):
        raise TypeError(f"arun_tool_loop requires acompletion to return a ModelResponse, got {type(response).__name__}")
    message: Final = response.choices[0].message
    if not message.tool_calls:
        return message.content
    if rounds_left == 1:
        raise ToolLoopMaxRoundsExceeded(max_rounds)
    tool_calls: Final = _function_tool_calls(message)
    tool_results: Final = tuple([await execute_tool(tool_call) for tool_call in tool_calls])
    return await _arun_tool_loop_rounds(
        model=model,
        history=(*history, _assistant_message(message, tool_calls), *tool_results),
        tools=tools,
        execute_tool=execute_tool,
        max_rounds=max_rounds,
        rounds_left=rounds_left - 1,
        **completion_kwargs,
    )


def run_tool_loop(
    *,
    model: str,
    messages: Sequence[AllMessageValues],
    tools: Sequence[ChatCompletionToolParam],
    execute_tool: ToolExecutor,
    max_rounds: int = DEFAULT_TOOL_LOOP_MAX_ROUNDS,
    **completion_kwargs: Unpack[_ToolLoopCompletionKwargs],
) -> str | None:
    """Call completion, execute each requested tool, and repeat until the model answers.

    Returns the final assistant message content. Raises ToolLoopMaxRoundsExceeded when the
    model is still requesting tools after max_rounds completions.

    Example:
        execute = functools.partial(run_repo_tool, repository="litellm", revision="main")
        answer = litellm.run_tool_loop(
            model="anthropic/claude-sonnet-5-5", messages=messages, tools=tools, execute_tool=execute
        )
    """
    _validate_tool_loop_args(max_rounds, completion_kwargs)
    return _run_tool_loop_rounds(
        model=model,
        history=tuple(messages),
        tools=tools,
        execute_tool=execute_tool,
        max_rounds=max_rounds,
        rounds_left=max_rounds,
        **completion_kwargs,
    )


async def arun_tool_loop(
    *,
    model: str,
    messages: Sequence[AllMessageValues],
    tools: Sequence[ChatCompletionToolParam],
    execute_tool: AsyncToolExecutor,
    max_rounds: int = DEFAULT_TOOL_LOOP_MAX_ROUNDS,
    **completion_kwargs: Unpack[_ToolLoopCompletionKwargs],
) -> str | None:
    """Async version of run_tool_loop, awaiting each execute_tool call in order.

    Returns the final assistant message content. Raises ToolLoopMaxRoundsExceeded when the
    model is still requesting tools after max_rounds completions.

    Example:
        execute = functools.partial(arun_repo_tool, repository="litellm", revision="main")
        answer = await litellm.arun_tool_loop(
            model="anthropic/claude-sonnet-5-5", messages=messages, tools=tools, execute_tool=execute
        )
    """
    _validate_tool_loop_args(max_rounds, completion_kwargs)
    return await _arun_tool_loop_rounds(
        model=model,
        history=tuple(messages),
        tools=tools,
        execute_tool=execute_tool,
        max_rounds=max_rounds,
        rounds_left=max_rounds,
        **completion_kwargs,
    )
