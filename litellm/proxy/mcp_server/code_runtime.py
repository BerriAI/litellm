from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, TypeAlias

import anyio
from anyio.abc import Process
from anyio.lowlevel import RunVar
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic import JsonValue

from litellm.proxy.mcp_server.code_worker import (
    WORKER_MESSAGE,
    Call,
    CodeExecutionLimits,
    CodeExecutionResult,
    Done,
    Reply,
    Start,
    encode_message,
    load_json,
)

ToolCallback: TypeAlias = Callable[[str, dict[str, JsonValue]], Awaitable[dict[str, JsonValue]]]
_WORKERS: Final = RunVar[anyio.CapacityLimiter]("mcp_code_workers")
_DEFAULT_LIMITS: Final = CodeExecutionLimits()


class _Calls:
    def __init__(self) -> None:
        self.started: frozenset[int] = frozenset()
        self.active: frozenset[int] = frozenset()

    def start(self, call_id: int) -> None:
        self.started = self.started | {call_id}
        self.active = self.active | {call_id}

    def finish(self, call_id: int) -> None:
        self.active = self.active - {call_id}


def check_runtime() -> None:
    try:
        metadata.distribution("quickjs-ng")
    except metadata.PackageNotFoundError as exc:
        raise ValueError(
            "MCP code mode requires quickjs-ng. Install 'litellm[proxy,mcp-code-mode]' on a supported platform."
        ) from exc
    try:
        metadata.distribution("quickjs")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("MCP code mode requires removing the conflicting quickjs package and reinstalling quickjs-ng")
    if importlib.util.find_spec("_quickjs") is None:
        raise ValueError(
            "MCP code mode requires quickjs-ng. Install 'litellm[proxy,mcp-code-mode]' on a supported platform."
        )


async def _build_reply(call: Call, callback: ToolCallback, maximum: int) -> bytes:
    try:
        value: Final = await callback(call.tool_id, dict(call.arguments))
        reply: Final = Reply(id=call.id, value=value)
        try:
            return encode_message(reply, maximum)
        except (ValueError, RecursionError):
            return encode_message(
                Reply(id=call.id, error="Tool result is too large or contains unsupported JSON numbers"), maximum
            )
    except Exception:  # noqa: BLE001 - Untrusted upstream failures must not disclose secrets to scripts.
        return encode_message(Reply(id=call.id, error="Tool execution failed"), maximum)


async def _exchange(
    process: Process,
    code: str,
    callback: ToolCallback,
    settings: CodeExecutionLimits,
    calls: _Calls,
) -> CodeExecutionResult:
    assert process.stdin is not None and process.stdout is not None
    output: Final = BufferedByteReceiveStream(process.stdout)
    writer: Final = anyio.Lock()

    async def dispatch(call: Call) -> None:
        data: Final = await _build_reply(call, callback, settings.message_bytes)
        async with writer:
            assert process.stdin is not None
            calls.finish(call.id)
            await process.stdin.send(data)

    await process.stdin.send(encode_message(Start(code=code, limits=settings), settings.message_bytes))
    async with anyio.create_task_group() as group:
        try:
            async for message in _messages(output, settings.message_bytes):
                match message:
                    case Call():
                        if message.id in calls.started or len(calls.started) >= settings.max_calls:
                            return CodeExecutionResult(
                                error="Code execution tool-call limit exceeded", calls_started=len(calls.started)
                            )
                        if len(calls.active) >= settings.max_concurrent_calls:
                            return CodeExecutionResult(
                                error="Code execution concurrent-call limit exceeded", calls_started=len(calls.started)
                            )
                        calls.start(message.id)
                        group.start_soon(dispatch, message)
                    case Done():
                        if len(json.dumps(message.output, ensure_ascii=False).encode("utf-8")) > settings.output_bytes:
                            return CodeExecutionResult(
                                error="Code output exceeds its byte limit", calls_started=len(calls.started)
                            )
                        return CodeExecutionResult(
                            output=message.output,
                            error="Script returned with unawaited tool calls" if calls.active else message.error,
                            calls_started=len(calls.started),
                        )
        finally:
            group.cancel_scope.cancel()
    return CodeExecutionResult(error="Code execution worker closed without a result", calls_started=len(calls.started))


async def _messages(stream: BufferedByteReceiveStream, maximum: int) -> AsyncIterator[Call | Done]:
    while True:
        yield WORKER_MESSAGE.validate_python(load_json((await stream.receive_until(b"\n", maximum)).decode("utf-8")))


async def execute_code(
    code: str, callback: ToolCallback, settings: CodeExecutionLimits = _DEFAULT_LIMITS
) -> CodeExecutionResult:
    try:
        size: Final = len(code.encode("utf-8"))
    except UnicodeError:
        return CodeExecutionResult(error="Code must contain valid Unicode")
    if size > settings.script_bytes:
        return CodeExecutionResult(error="Code exceeds its byte limit")
    check_runtime()
    existing: Final = _WORKERS.get(None)
    limiter: Final = existing if existing is not None else anyio.CapacityLimiter(settings.max_workers)
    if existing is None:
        _WORKERS.set(limiter)
    try:
        limiter.acquire_nowait()
    except anyio.WouldBlock:
        return CodeExecutionResult(error="Code execution is busy; retry when a worker is available")
    calls: Final = _Calls()
    try:
        with TemporaryDirectory(prefix="litellm-code-") as directory:
            environment: Final = {"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {}
            with anyio.fail_after(settings.timeout_seconds):
                async with await anyio.open_process(
                    [sys.executable, "-I", "-u", str(Path(__file__).with_name("code_worker.py"))],
                    cwd=directory,
                    env=environment,
                    stderr=subprocess.DEVNULL,
                ) as process:
                    try:
                        return await _exchange(process, code, callback, settings, calls)
                    finally:
                        with anyio.CancelScope(shield=True):
                            if process.returncode is None:
                                try:
                                    process.kill()
                                except ProcessLookupError:
                                    pass
                            await process.wait()
    except TimeoutError:
        return CodeExecutionResult(
            error="Code execution timed out; upstream actions may have completed", calls_started=len(calls.started)
        )
    except Exception:  # noqa: BLE001 - Process/protocol failures are isolated; cancellation still propagates.
        return CodeExecutionResult(
            error="Code execution worker failed; upstream actions may have completed", calls_started=len(calls.started)
        )
    finally:
        limiter.release()
