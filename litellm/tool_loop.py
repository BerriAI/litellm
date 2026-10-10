"""Client-side tool-calling loop helpers for litellm.completion."""

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeAlias, cast

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


def _expect_model_response(response: object) -> ModelResponse:
    if not isinstance(response, ModelResponse):
        raise TypeError(f"run_tool_loop requires completion to return a ModelResponse, got {type(response).__name__}")
    return response


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
    return cast(  # cast-ok: dict literal with thinking_blocks and reasoning_items spread in only when set
        "ChatCompletionAssistantMessage",
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [_assistant_tool_call(tc) for tc in tool_calls],
            **{
                key: value
                for key, value in (
                    ("thinking_blocks", getattr(message, "thinking_blocks", None)),
                    ("reasoning_items", getattr(message, "reasoning_items", None)),
                )
                if value is not None
            },
        },
    )


def _function_tool_call(tool_call: object) -> ChatCompletionMessageToolCall:
    if not isinstance(tool_call, ChatCompletionMessageToolCall):
        raise TypeError(
            f"run_tool_loop only executes function tool calls, got custom tool call {getattr(tool_call, 'id', None)}"
        )
    return tool_call


def _function_tool_calls(message: Message) -> tuple[ChatCompletionMessageToolCall, ...]:
    return tuple(_function_tool_call(tool_call) for tool_call in message.tool_calls or ())


def run_tool_loop(
    *,
    model: str,
    messages: Sequence[AllMessageValues],
    tools: Sequence[ChatCompletionToolParam],
    execute_tool: ToolExecutor,
    max_rounds: int = DEFAULT_TOOL_LOOP_MAX_ROUNDS,
    **completion_kwargs: Unpack[_ToolLoopCompletionKwargs],  # kwargs-ok: forwarded verbatim to litellm.completion
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
    import litellm

    _validate_tool_loop_args(max_rounds, completion_kwargs)
    history: tuple[AllMessageValues, ...] = tuple(messages)  # rebind-ok: rounds append new turns
    for round_number in range(1, max_rounds + 1):
        message = (
            _expect_model_response(
                litellm.completion(model=model, messages=list(history), tools=list(tools), **completion_kwargs)
            )
            .choices[0]
            .message
        )
        if not message.tool_calls:
            return message.content
        tool_calls = _function_tool_calls(message)
        if round_number == max_rounds:
            break
        tool_results = tuple(execute_tool(tool_call) for tool_call in tool_calls)
        history = (*history, _assistant_message(message, tool_calls), *tool_results)
    raise ToolLoopMaxRoundsExceeded(max_rounds)


async def arun_tool_loop(
    *,
    model: str,
    messages: Sequence[AllMessageValues],
    tools: Sequence[ChatCompletionToolParam],
    execute_tool: AsyncToolExecutor,
    max_rounds: int = DEFAULT_TOOL_LOOP_MAX_ROUNDS,
    **completion_kwargs: Unpack[_ToolLoopCompletionKwargs],  # kwargs-ok: forwarded verbatim to litellm.acompletion
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
    import litellm

    _validate_tool_loop_args(max_rounds, completion_kwargs)
    history: tuple[AllMessageValues, ...] = tuple(messages)  # rebind-ok: rounds append new turns
    for round_number in range(1, max_rounds + 1):
        message = (
            _expect_model_response(
                await litellm.acompletion(model=model, messages=list(history), tools=list(tools), **completion_kwargs)
            )
            .choices[0]
            .message
        )
        if not message.tool_calls:
            return message.content
        tool_calls = _function_tool_calls(message)
        if round_number == max_rounds:
            break
        tool_results = tuple([await execute_tool(tool_call) for tool_call in tool_calls])
        history = (*history, _assistant_message(message, tool_calls), *tool_results)
    raise ToolLoopMaxRoundsExceeded(max_rounds)
