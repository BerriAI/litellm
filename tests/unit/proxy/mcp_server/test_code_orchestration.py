import asyncio
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

import anyio
import pytest
from anyio.abc import ByteReceiveStream, ByteSendStream, Process
from mcp.types import TextContent, Tool
from pydantic import JsonValue

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3, get_request_stash
from litellm.proxy.mcp_server.code_execution import CODE_TOOL_NAME, run_code_tool
from litellm.proxy.mcp_server.code_worker import Call, Done, Reply, Start, encode_message, load_json
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
from litellm.proxy.mcp_server.result_conversion import INPUT_REQUIRED_UNSUPPORTED_MESSAGE
from litellm.proxy.mcp_server.tool_registry import global_mcp_tool_registry
from litellm.proxy.mcp_server.tool_search import mcp_proxy_tool_id, with_mcp_proxy_identity
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp import MCPPostCallResponseObject, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.utils import CallTypesLiteral

pytestmark: Final = pytest.mark.usefixtures("mcp_code_runtime")


class ScriptedProcess(Process, ByteReceiveStream, ByteSendStream):
    """Only the external process transport is replaced; each reply comes from LiteLLM."""

    def __init__(self, calls: tuple[Call, ...]) -> None:
        self.calls = calls
        self.started: Start | None = None
        self.replies: tuple[Reply, ...] = ()
        self.stopped = False
        self.sender, self.receiver = anyio.create_memory_object_stream[bytes](1)

    async def send(self, item: bytes) -> None:
        if self.started is None:
            self.started = Start.model_validate_json(item)
        else:
            reply: Final = Reply.model_validate_json(item)
            assert reply.id == self.calls[len(self.replies)].id
            self.replies = (*self.replies, reply)
        message: Final = (
            self.calls[len(self.replies)]
            if len(self.replies) < len(self.calls)
            else Done(
                output=tuple(
                    dict(reply.value) if reply.value is not None else {"error": reply.error} for reply in self.replies
                )
            )
        )
        await self.sender.send(encode_message(message, 2 * 1024 * 1024))

    async def receive(self, max_bytes: int = 65536) -> bytes:
        data: Final = await self.receiver.receive()
        assert len(data) <= max_bytes
        return data

    async def aclose(self) -> None:
        await self.sender.aclose()
        await self.receiver.aclose()

    async def wait(self) -> int:
        assert self.stopped
        return -9

    def kill(self) -> None:
        self.stopped = True

    def terminate(self) -> None:
        self.kill()

    def send_signal(self, signal: int) -> None:
        self.kill()

    @property
    def pid(self) -> int:
        return 0

    @property
    def returncode(self) -> int | None:
        return -9 if self.stopped else None

    @property
    def stdin(self) -> ByteSendStream:
        return self

    @property
    def stdout(self) -> ByteReceiveStream:
        return self

    @property
    def stderr(self) -> None:
        return None


def install_transport(monkeypatch: pytest.MonkeyPatch, calls: tuple[Call, ...]) -> ScriptedProcess:
    process: Final = ScriptedProcess(calls)

    async def open_process(command: Sequence[str], *, cwd: str, env: Mapping[str, str], stderr: int) -> Process:
        assert "-I" in command and "-u" in command
        assert set(env) <= {"SYSTEMROOT"}
        assert stderr == subprocess.DEVNULL
        assert cwd
        return process

    monkeypatch.setattr(anyio, "open_process", open_process)
    return process


class ObservedCalls(CustomLogger):
    def __init__(self) -> None:
        self.names: tuple[str, ...] = ()
        self.traces: tuple[object, ...] = ()
        self.responses: tuple[MCPPostCallResponseObject, ...] = ()
        self.log_traces: tuple[object, ...] = ()
        self.handler_started = asyncio.Event()

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> dict[str, object]:
        name: Final = data.get("name") or data.get("mcp_tool_name")
        if isinstance(name, str):
            self.names = (*self.names, name)
        self.traces = (*self.traces, data.get("litellm_trace_id"))
        return data

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict[str, object],
        response_obj: MCPPostCallResponseObject,
        start_time: datetime,
        end_time: datetime,
    ) -> MCPPostCallResponseObject:
        self.responses = (*self.responses, response_obj)
        self.log_traces = (*self.log_traces, kwargs.get("litellm_trace_id"))
        return response_obj


@pytest.fixture
def registered_gateway(monkeypatch: pytest.MonkeyPatch) -> tuple[OperationContext, ObservedCalls, list[str]]:
    observed: Final = ObservedCalls()
    logger: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(logger.internal_usage_cache)
    monkeypatch.setattr(
        litellm, "mcp_code_mode", {"enabled": True, "excluded_tools": {"allowed-blocked": "requires manual approval"}}
    )
    monkeypatch.setattr(litellm, "callbacks", [limiter, observed])
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logger)
    manager: Final = global_mcp_server_manager
    monkeypatch.setattr(manager, "registry", {})
    monkeypatch.setattr(manager, "tool_name_to_mcp_server_name_mapping", {})
    monkeypatch.setattr(global_mcp_tool_registry, "tools", {})
    received: Final[list[str]] = []

    async def echo(message: str) -> str:
        stash: Final = get_request_stash()
        assert stash is not None and stash.borrowed_parallel_keys
        received.append(message)
        if message == "wait":
            observed.handler_started.set()
            await anyio.sleep_forever()
        if message == "continuation documentation":
            return INPUT_REQUIRED_UNSUPPORTED_MESSAGE
        if message == "upstream failure":
            raise ValueError("upstream-specific detail")
        return message

    for server in (
        MCPServer(
            server_id=alias,
            name=alias,
            alias=alias,
            transport=MCPTransport.http,
            url="https://example.invalid/mcp",
            spec_path="registered-openapi.json",
        )
        for alias in ("allowed", "denied")
    ):
        manager.registry[server.name] = server
        manager.tool_name_to_mcp_server_name_mapping[server.name + "-echo"] = server.name
        global_mcp_tool_registry.register_tool(  # pyright: ignore[reportUnknownMemberType]  # Legacy registry leaves handler Callable unparameterized.
            name=server.name + "-echo",
            description="Echo",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            handler=echo,
            server_id=server.server_id,
        )
    global_mcp_tool_registry.register_tool(  # pyright: ignore[reportUnknownMemberType]  # Legacy registry leaves handler Callable unparameterized.
        name="allowed-blocked",
        description="Blocked",
        input_schema={"type": "object"},
        handler=echo,
        server_id="allowed",
    )
    manager.tool_name_to_mcp_server_name_mapping["allowed-blocked"] = "allowed"
    auth: Final = UserAPIKeyAuth(
        api_key="code-mode-component",
        max_parallel_requests=1,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="code-mode-component", mcp_servers=["allowed"]
        ),
    )
    return (
        OperationContext(_caller=auth, raw_headers={"x-litellm-trace-id": "code-component-trace"}),
        observed,
        received,
    )


def error_text(reply: Reply) -> str:
    assert reply.value is not None and reply.value["isError"] is True
    content: Final = reply.value["content"]
    assert isinstance(content, list) and isinstance(content[0], dict)
    return str(content[0]["text"])


@pytest.mark.asyncio
@pytest.mark.parametrize("opaque", [False, True])
async def test_authenticated_coordinator_preserves_results_and_rejects_unauthorized_tools(
    monkeypatch: pytest.MonkeyPatch,
    registered_gateway: tuple[OperationContext, ObservedCalls, list[str]],
    opaque: bool,
) -> None:
    context, observed, received = registered_gateway
    reference: Final = (
        mcp_proxy_tool_id(with_mcp_proxy_identity(Tool(name="allowed-echo", input_schema={}), "allowed"))
        if opaque
        else "allowed-echo"
    )
    process: Final = install_transport(
        monkeypatch,
        (
            Call(id=0, tool_id=reference, arguments={"message": "kept"}),
            Call(id=1, tool_id="denied-echo", arguments={"message": "must not run"}),
            Call(id=2, tool_id=CODE_TOOL_NAME, arguments={}),
            Call(id=3, tool_id="allowed-blocked", arguments={"message": "must not run"}),
            Call(id=4, tool_id=reference, arguments={"message": "upstream failure"}),
        ),
    )
    result: Final = await run_code_tool({"code": "/* protocol fixture */"}, context)
    assert not result.is_error
    assert process.stopped
    assert process.started is not None and process.started.code == "/* protocol fixture */"
    assert received == ["kept", "upstream failure"]
    assert process.replies[0].value == {
        "content": [{"type": "text", "text": "kept"}],
        "isError": False,
        "resultType": "complete",
    }
    for reply, reason in zip(process.replies[1:4], ("Unknown", "Nested", "requires manual approval"), strict=True):
        assert reason in error_text(reply)
    assert process.replies[4].value == {
        "content": [{"type": "text", "text": "Error: upstream-specific detail"}],
        "isError": True,
        "resultType": "complete",
    }
    assert isinstance(result.content[0], TextContent)
    body: Final = load_json(result.content[0].text)
    assert isinstance(body, dict) and body["calls_started"] == 5
    assert body["output"] == [
        dict(reply.value) if reply.value is not None else {"error": reply.error} for reply in process.replies
    ]
    assert observed.responses
    assert observed.log_traces == ("code-component-trace",) * len(observed.responses)
    assert "code-component-trace" in observed.traces


@pytest.mark.asyncio
async def test_invalid_script_releases_the_outer_admission(
    monkeypatch: pytest.MonkeyPatch, registered_gateway: tuple[OperationContext, ObservedCalls, list[str]]
) -> None:
    context, observed, received = registered_gateway
    process: Final = install_transport(monkeypatch, ())
    invalid: Final = await run_code_tool({"code": 42}, context)
    assert invalid.is_error and isinstance(invalid.content[0], TextContent)
    assert "Invalid code request" in invalid.content[0].text
    assert process.started is None and received == []
    result: Final = await run_code_tool({"code": "/* protocol fixture */"}, context)
    assert not result.is_error
    assert CODE_TOOL_NAME in observed.names
    assert process.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "unsafe number"])
async def test_runtime_sanitizes_callback_failures(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    from litellm.proxy.mcp_server.code_runtime import execute_code

    async def upstream(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        assert tool_id == "echo" and arguments == {"n": 1}
        if failure == "exception":
            raise RuntimeError("upstream-specific detail")
        return {"large": 2**53}

    process: Final = install_transport(monkeypatch, (Call(id=0, tool_id="echo", arguments={"n": 1}),))
    result: Final = await execute_code("/* protocol fixture */", upstream)
    expected: Final = (
        "Tool execution failed"
        if failure == "exception"
        else "Tool result is too large or contains unsupported JSON numbers"
    )
    assert process.replies == (Reply(id=0, error=expected),)
    assert result.output == ({"error": expected},)
    assert result.calls_started == 1 and process.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["call count", "output bytes"])
async def test_runtime_enforces_limits_on_worker_messages(monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    from dataclasses import replace

    from litellm.proxy.mcp_server.code_runtime import CodeExecutionLimits, execute_code

    limits: Final = (
        replace(CodeExecutionLimits(), max_calls=1)
        if boundary == "call count"
        else replace(CodeExecutionLimits(), output_bytes=8)
    )
    first: Final = Call(id=0, tool_id="echo", arguments={"n": 1})
    calls: Final = (first, Call(id=1, tool_id="echo", arguments={"n": 2})) if boundary == "call count" else (first,)
    process: Final = install_transport(monkeypatch, calls)
    received: Final[list[dict[str, JsonValue]]] = []

    async def upstream(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        assert tool_id == "echo"
        received.append(arguments)
        return {"text": "value exceeds eight bytes"}

    result: Final = await execute_code("/* protocol fixture */", upstream, limits)
    assert received == [{"n": 1}]
    expected: Final = "tool-call limit" if boundary == "call count" else "output exceeds its byte limit"
    assert result.error is not None and expected in result.error
    assert result.calls_started == 1 and result.output == ()
    assert process.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame", [b"not JSON\n", b'{"kind":"unknown"}\n', b'{"kind":"done","output":[9007199254740993]}\n']
)
async def test_runtime_rejects_invalid_worker_protocol(monkeypatch: pytest.MonkeyPatch, frame: bytes) -> None:
    from litellm.proxy.mcp_server.code_runtime import execute_code

    process: Final = install_transport(monkeypatch, ())

    async def receive(max_bytes: int = 65536) -> bytes:
        assert len(frame) <= max_bytes
        return frame

    async def upstream(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        pytest.fail("Malformed worker messages must not dispatch tools")

    monkeypatch.setattr(process, "receive", receive)
    result: Final = await execute_code("/* protocol fixture */", upstream)
    assert result.error == "Code execution worker failed; upstream actions may have completed"
    assert result.calls_started == 0 and result.output == ()
    assert process.stopped


@pytest.mark.asyncio
async def test_cancelling_coordinator_releases_parent_and_child_admission(
    monkeypatch: pytest.MonkeyPatch, registered_gateway: tuple[OperationContext, ObservedCalls, list[str]]
) -> None:
    context, observed, received = registered_gateway
    process: Final = install_transport(
        monkeypatch, (Call(id=0, tool_id="allowed-echo", arguments={"message": "wait"}),)
    )
    request: Final = asyncio.create_task(run_code_tool({"code": "/* protocol fixture */"}, context))
    entered: Final = asyncio.create_task(observed.handler_started.wait())
    try:
        completed, _ = await asyncio.wait((request, entered), return_when=asyncio.FIRST_COMPLETED)
        assert entered in completed, "Request exited before reaching the registered handler"
        assert entered.result()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    finally:
        for task in (request, entered):
            if not task.done():
                task.cancel()
        await asyncio.gather(request, entered, return_exceptions=True)
    assert process.stopped and process.replies == ()
    assert received == ["wait"]
    replacement: Final = install_transport(
        monkeypatch, (Call(id=0, tool_id="allowed-echo", arguments={"message": "after cancellation"}),)
    )
    result: Final = await run_code_tool({"code": "/* protocol fixture */"}, context)
    assert not result.is_error
    assert received == ["wait", "after cancellation"]
    assert replacement.replies[0].value is not None and replacement.replies[0].value["isError"] is False
    assert replacement.stopped


@pytest.mark.asyncio
async def test_successful_continuation_text_is_not_reclassified_as_an_error(
    monkeypatch: pytest.MonkeyPatch, registered_gateway: tuple[OperationContext, ObservedCalls, list[str]]
) -> None:
    context, _, received = registered_gateway
    process: Final = install_transport(
        monkeypatch, (Call(id=0, tool_id="allowed-echo", arguments={"message": "continuation documentation"}),)
    )
    result: Final = await run_code_tool({"code": "/* protocol fixture */"}, context)
    assert not result.is_error and received == ["continuation documentation"]
    assert process.replies[0].value == {
        "content": [{"type": "text", "text": INPUT_REQUIRED_UNSUPPORTED_MESSAGE}],
        "isError": False,
        "resultType": "complete",
    }
