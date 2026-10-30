from __future__ import annotations

from dataclasses import replace
from typing import Final

import anyio
import pytest
from pydantic import JsonValue

from litellm.proxy.mcp_server.code_runtime import CodeExecutionLimits, execute_code


async def _echo(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"structuredContent": arguments, "content": [], "isError": tool_id == "error"}


@pytest.mark.asyncio
async def test_code_roundtrips_full_results_and_chains_calls() -> None:
    result = await execute_code(
        """
        const first = await tools.call_tool({tool_id: 'echo', arguments: {id: '9007199254740993'}});
        const results = await Promise.all(['echo', 'error'].map(tool_id =>
            tools.call_tool({tool_id, arguments: first.structuredContent})));
        text(results);
        """,
        _echo,
    )
    assert result.error is None
    assert result.calls_started == 3
    assert result.output == (
        [
            {"structuredContent": {"id": "9007199254740993"}, "content": [], "isError": False},
            {"structuredContent": {"id": "9007199254740993"}, "content": [], "isError": True},
        ],
    )


@pytest.mark.asyncio
async def test_script_near_byte_limit_survives_json_escaping() -> None:
    source = "/*" + '\\"' * ((CodeExecutionLimits().script_bytes - 20) // 2) + "*/text(42)"
    result = await execute_code(source, _echo)
    assert result.error is None
    assert result.output == (42,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("syntax error !", "SyntaxError"),
        ("text(missing)", "ReferenceError"),
        ("throw new Error('repair me')", "repair me"),
    ],
)
async def test_script_errors_help_the_caller_repair_the_code(code: str, message: str) -> None:
    result = await execute_code(code, _echo)
    assert result.error is not None and message in result.error
    assert result.calls_started == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["while (true) {}", "await new Promise(() => {})"])
async def test_code_deadline_releases_worker_for_next_request(code: str) -> None:
    limits = replace(CodeExecutionLimits(), timeout_seconds=1)
    result = await execute_code(code, _echo, limits)
    assert result.error is not None
    next_result = await execute_code("text(42)", _echo)
    assert next_result.output == (42,)
    assert next_result.error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "text(9007199254740993)",
        "text(NaN)",
        "text(undefined)",
        "text(1n)",
        "const a = {}; a.a = a; text(a)",
        "text('x'.repeat(10000))",
        "syntax error !",
    ],
)
async def test_code_rejects_unserializable_or_oversized_output(code: str) -> None:
    result = await execute_code(code, _echo, replace(CodeExecutionLimits(), output_bytes=1024))
    assert result.error is not None
    assert result.output == ()


@pytest.mark.asyncio
async def test_code_rejects_unsafe_upstream_integer_without_silent_rounding() -> None:
    async def large_integer(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {"value": 9007199254740993}

    result = await execute_code("text(await tools.call_tool({tool_id:'echo', arguments:{}}))", large_integer)
    assert result.error is not None
    assert "JSON numbers" in result.error
    assert result.calls_started == 1
    assert result.output == ()


@pytest.mark.asyncio
async def test_code_bounds_calls_even_when_javascript_catches_errors() -> None:
    result = await execute_code(
        "for(let i=0;i<5;i++) { try { await tools.call_tool({tool_id:'echo',arguments:{}}); } catch (_) {} }",
        _echo,
        replace(CodeExecutionLimits(), max_calls=2),
    )
    assert result.calls_started == 2
    assert result.error is not None


@pytest.mark.asyncio
async def test_code_cancellation_cleans_pending_callbacks_and_releases_capacity() -> None:
    started: Final = anyio.Event()
    stopped: Final = anyio.Event()
    scope: Final = anyio.CancelScope()

    async def blocking(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            stopped.set()
        return {}

    async def run() -> None:
        with scope:
            await execute_code("await tools.call_tool({tool_id:'echo', arguments:{}})", blocking)

    async with anyio.create_task_group() as group:
        group.start_soon(run)
        with anyio.fail_after(10):
            await started.wait()
            scope.cancel()
            await stopped.wait()
    result = await execute_code("text('ok')", _echo)
    assert result.error is None
    assert result.output == ("ok",)


@pytest.mark.asyncio
async def test_code_has_no_system_apis_and_does_not_keep_globals() -> None:
    first = await execute_code(
        "globalThis.secret='temporary'; text([typeof process, typeof require, typeof fetch]);", _echo
    )
    second = await execute_code("text(typeof globalThis.secret)", _echo)
    assert first.output == (["undefined", "undefined", "undefined"],)
    assert second.output == ("undefined",)


@pytest.mark.asyncio
async def test_code_does_not_retry_a_completed_action_after_failure() -> None:
    seen: Final[list[str]] = []

    async def action(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        seen.append(tool_id)
        return {}

    result = await execute_code(
        "await tools.call_tool({tool_id:'write', arguments:{}}); throw new Error('failed');", action
    )
    assert result.error is not None
    assert result.calls_started == 1
    assert seen == ["write"]


@pytest.mark.asyncio
async def test_code_memory_failure_leaves_next_execution_healthy() -> None:
    result = await execute_code(
        "const data = []; while(true) data.push(new Array(65536).fill('value'));",
        _echo,
        replace(CodeExecutionLimits(), memory_bytes=16 * 1024 * 1024, timeout_seconds=5),
    )
    assert result.error is not None
    next_result = await execute_code("text('healthy')", _echo)
    assert next_result.output == ("healthy",)


@pytest.mark.asyncio
async def test_code_unawaited_callbacks_are_cancelled() -> None:
    async def blocking(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        await anyio.sleep_forever()
        return {}

    result = await execute_code("tools.call_tool({tool_id:'echo', arguments:{}})", blocking)
    assert result.error is not None
    assert "unawaited" in result.error


@pytest.mark.asyncio
async def test_code_timeout_retains_completed_call_count() -> None:
    result = await execute_code(
        "await tools.call_tool({tool_id:'echo',arguments:{}}); await new Promise(()=>{});",
        _echo,
        replace(CodeExecutionLimits(), timeout_seconds=1),
    )
    assert result.error is not None
    assert result.calls_started == 1
