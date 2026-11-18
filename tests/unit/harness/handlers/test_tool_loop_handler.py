"""Tests for the in-process Tool Loop handler."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final, Literal

import litellm
import pytest
from pydantic import BaseModel

from litellm import sandbox
from litellm.harness.context import GatewayTarget, SessionContext
from litellm.harness.handlers.tool_loop_handler import ToolLoopHandler
from litellm.harness.options import ToolLoopOptions
from litellm.harness.types import Approval, Event, Harness, Text, ToolCall, ToolResult
from litellm.llms.base_llm.harness.transformation import HarnessTurnError
from litellm.llms.tool_loop.harness.transformation import ToolLoopHarnessConfig
from litellm.types.utils import ModelResponse


class ScriptedCompletion:
    def __init__(self, responses: tuple[ModelResponse, ...]) -> None:
        self.responses: Iterator[ModelResponse] = iter(responses)
        self.calls: list[dict[str, object]] = []  # mutable-ok: captures injected completion requests

    async def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls.append(dict(kwargs))
        return next(self.responses)


def model_response(
    *,
    content: str | None = None,
    tool_calls: tuple[dict[str, object], ...] = (),
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    hidden_params: dict[str, object] | None = None,
) -> ModelResponse:
    response: Final = ModelResponse(
        model="gpt-test",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": list(tool_calls) if tool_calls else None,
                }
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
    if hidden_params is not None:
        response._hidden_params = hidden_params
    return response


def function_call(name: str, arguments: str, call_id: str = "call-1") -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def make_context(
    tmp_path: Path,
    *,
    model: str | None = "gpt-4o-mini",
    gateway: GatewayTarget | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    tools: tuple[Callable[..., object], ...] = (),
    permissions: Literal["ask", "full"] = "full",
    output: type[BaseModel] | None = None,
    metadata: Mapping[str, object] | None = None,
    options: ToolLoopOptions | None = None,
) -> SessionContext:
    return SessionContext(
        harness=Harness.TOOL_LOOP,
        sandbox=sandbox.local(tmp_path),
        session_id="tool-loop-session",
        model=model,
        gateway=gateway,
        api_key=api_key,
        api_base=api_base,
        instructions=instructions,
        tools=tools,
        permissions=permissions,
        output=output,
        metadata={} if metadata is None else metadata,
        options=options,
    )


def make_handler(completion: ScriptedCompletion) -> ToolLoopHandler:
    return ToolLoopHandler(ToolLoopHarnessConfig(), acompletion=completion)


async def run_turn(
    handler: ToolLoopHandler,
    ctx: SessionContext,
    prompt: str,
    allow: bool = True,
) -> tuple[Event, ...]:
    events: list[Event] = []  # mutable-ok: gathers this async turn for assertions
    async for event in handler.turn(ctx, prompt):
        events.append(event)
        if isinstance(event, Approval):
            event.allow() if allow else event.deny("not approved")
    return tuple(events)


def add(a: int, b: int) -> int:
    return a + b


def duplicate_add() -> Callable[..., object]:
    def add(a: int, b: int) -> int:
        return a + b

    return add


async def test_tool_round_trip_appends_assistant_and_tool_messages(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("add", '{"a": 2, "b": 3}'),)),
            model_response(content="The sum is 5"),
        )
    )
    ctx: Final = make_context(tmp_path, instructions="Use tools when needed", tools=(add,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Add two and three")

    assert events == (
        ToolCall(
            id="call-1",
            name="add",
            native_name="add",
            input={"a": 2, "b": 3},
            builtin=False,
        ),
        ToolResult(id="call-1", output="5", is_error=False),
        Text(delta="The sum is 5"),
    )
    assert ctx.final_text == "The sum is 5"
    assert completion.calls[1]["messages"] == [
        {"role": "system", "content": "Use tools when needed"},
        {"role": "user", "content": "Add two and three"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "5"},
    ]


async def test_multiple_tool_calls_run_in_order(tmp_path: Path) -> None:
    values: list[int] = []  # mutable-ok: records the order of calls from the injected model response

    def record(value: int) -> int:
        values.append(value)
        return value

    completion: Final = ScriptedCompletion(
        (
            model_response(
                tool_calls=(
                    function_call("record", '{"value": 1}', "call-1"),
                    function_call("record", '{"value": 2}', "call-2"),
                )
            ),
            model_response(content="Recorded"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(record,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Record both")

    assert values == [1, 2]
    assert tuple(event for event in events if isinstance(event, ToolResult)) == (
        ToolResult(id="call-1", output="1", is_error=False),
        ToolResult(id="call-2", output="2", is_error=False),
    )


async def test_tool_exception_is_returned_to_model_and_loop_continues(tmp_path: Path) -> None:
    def fail() -> str:
        raise RuntimeError("tool failed")

    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("fail", "{}"),)),
            model_response(content="Recovered"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(fail,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Run fail")

    assert ToolResult(id="call-1", output="RuntimeError: tool failed", is_error=True) in events
    assert completion.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "RuntimeError: tool failed",
    }
    assert ctx.final_text == "Recovered"


async def test_unknown_tool_is_returned_and_tools_are_omitted_when_empty(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("missing", "{}"),)),
            model_response(content="Unknown tool handled"),
        )
    )
    ctx: Final = make_context(tmp_path)
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Call a missing tool")

    assert "tools" not in completion.calls[0]
    assert (
        ToolResult(
            id="call-1",
            output="ValueError: unknown tool 'missing'",
            is_error=True,
        )
        in events
    )
    assert ctx.final_text == "Unknown tool handled"


@pytest.mark.parametrize(
    "arguments",
    ['{"a": 2}', '{"a": 2, "b": 3, "extra": 4}'],
)
async def test_invalid_tool_arguments_return_validation_error(
    tmp_path: Path,
    arguments: str,
) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("add", arguments),)),
            model_response(content="Arguments were invalid"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(add,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Add")
    result: Final = next(event for event in events if isinstance(event, ToolResult))

    assert result.is_error
    assert result.output.startswith("ValidationError:")
    assert completion.calls[1]["messages"][-1]["content"] == result.output


async def test_malformed_tool_arguments_return_error_without_calling_tool(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("add", "{"),)),
            model_response(content="Malformed arguments"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(add,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Add")
    result: Final = next(event for event in events if isinstance(event, ToolResult))

    assert result.is_error
    assert result.output.startswith("JSONDecodeError:")
    assert completion.calls[1]["messages"][-1]["content"] == result.output


async def test_ask_permission_denial_skips_tool_and_returns_reason(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("add", '{"a": 2, "b": 3}'),)),
            model_response(content="Denied"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(add,), permissions="ask")
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Add", allow=False)

    assert any(isinstance(event, Approval) for event in events)
    assert ToolResult(id="call-1", output="denied: not approved", is_error=True) in events
    assert completion.calls[1]["messages"][-1]["content"] == "denied: not approved"


async def test_ask_permission_allow_runs_tool(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("add", '{"a": 2, "b": 3}'),)),
            model_response(content="Allowed"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(add,), permissions="ask")
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Add")

    assert any(isinstance(event, Approval) for event in events)
    assert ToolResult(id="call-1", output="5", is_error=False) in events


async def test_async_tool_is_awaited(tmp_path: Path) -> None:
    async def multiply(a: int, b: int) -> int:
        return a * b

    completion: Final = ScriptedCompletion(
        (
            model_response(tool_calls=(function_call("multiply", '{"a": 3, "b": 4}'),)),
            model_response(content="12"),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(multiply,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    events: Final = await run_turn(handler, ctx, "Multiply")

    assert ToolResult(id="call-1", output="12", is_error=False) in events


class Answer(BaseModel):
    value: int


async def test_structured_output_is_forwarded_and_retained(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion((model_response(content='{"value": 7}'),))
    ctx: Final = make_context(tmp_path, output=Answer)
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    await run_turn(handler, ctx, "Return a value")

    assert completion.calls[0]["response_format"] is Answer
    assert ctx.output_json == '{"value": 7}'
    assert ctx.final_text == '{"value": 7}'


async def test_gateway_routing_uses_proxy_model_and_tool_loop_tag(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion((model_response(content="done"),))
    gateway: Final = GatewayTarget(api_base="http://gateway", api_key="sk-virtual")
    ctx: Final = make_context(tmp_path, gateway=gateway, metadata={"team": "test"})
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    await run_turn(handler, ctx, "Hi")

    assert completion.calls[0]["model"] == "litellm_proxy/gpt-4o-mini"
    assert completion.calls[0]["api_base"] == "http://gateway"
    assert completion.calls[0]["api_key"] == "sk-virtual"
    headers: Final = completion.calls[0]["extra_headers"]
    assert isinstance(headers, dict)
    assert headers["x-litellm-tags"] == "harness,tool_loop"
    assert '"team": "test"' in headers["x-litellm-spend-logs-metadata"]


async def test_usage_and_cost_accumulate_across_model_calls(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion(
        (
            model_response(
                tool_calls=(function_call("add", '{"a": 2, "b": 3}'),),
                prompt_tokens=10,
                completion_tokens=4,
                hidden_params={
                    "additional_headers": {"llm_provider-x-litellm-response-cost": "0.4"},
                    "response_cost": 0.1,
                },
            ),
            model_response(
                content="done",
                prompt_tokens=20,
                completion_tokens=5,
                hidden_params={"response_cost": 0.2},
            ),
        )
    )
    ctx: Final = make_context(tmp_path, tools=(add,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    await run_turn(handler, ctx, "Add")

    assert ctx.calls == 2
    assert ctx.input_tokens == 30
    assert ctx.output_tokens == 9
    assert ctx.cost == pytest.approx(0.6)


async def test_history_survives_stop_and_start(tmp_path: Path) -> None:
    completion: Final = ScriptedCompletion((model_response(content="first"), model_response(content="second")))
    ctx: Final = make_context(tmp_path, instructions="Keep answers concise")
    handler: Final = make_handler(completion)
    await handler.start(ctx)
    await run_turn(handler, ctx, "first prompt")
    await handler.stop(ctx)
    await handler.start(ctx)

    await run_turn(handler, ctx, "second prompt")

    assert completion.calls[1]["messages"] == [
        {"role": "system", "content": "Keep answers concise"},
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "second prompt"},
    ]
    history: Final = await handler.history(ctx)
    history[0]["content"] = "changed"
    assert (await handler.history(ctx))[0]["content"] == "Keep answers concise"


async def test_duplicate_tool_names_are_rejected(tmp_path: Path) -> None:
    ctx: Final = make_context(tmp_path, tools=(add, duplicate_add()))
    handler: Final = make_handler(ScriptedCompletion(()))

    with pytest.raises(ValueError, match="tool names must be unique"):
        await handler.start(ctx)


async def test_model_call_limit_raises_harness_turn_error(tmp_path: Path) -> None:
    repeating_response: Final = model_response(tool_calls=(function_call("ping", "{}"),))
    completion: Final = ScriptedCompletion((repeating_response,) * 100)

    def ping() -> str:
        return "pong"

    ctx: Final = make_context(tmp_path, tools=(ping,))
    handler: Final = make_handler(completion)
    await handler.start(ctx)

    with pytest.raises(HarnessTurnError, match="exceeded 100 model calls"):
        await run_turn(handler, ctx, "Ping repeatedly")


async def test_public_aagent_uses_tool_loop_with_mock_response(tmp_path: Path) -> None:
    result: Final = await litellm.aagent(
        Harness.TOOL_LOOP,
        "Say done",
        sandbox=sandbox.local(tmp_path),
        model="gpt-4o-mini",
        options=ToolLoopOptions(completion_kwargs={"mock_response": "done"}),
    )

    assert result.text == "done"
    assert result.stop_reason == "done"
