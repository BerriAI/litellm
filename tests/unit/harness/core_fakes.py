"""Fake handler/config, sandbox and endpoint shared by the core runtime tests."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from litellm.harness import runtime
from litellm.harness.context import SessionContext
from litellm.harness.handlers.base import BaseHarnessHandler
from litellm.llms.base_llm.harness.transformation import BaseHarnessConfig
from litellm.harness.options import ClaudeCodeOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.sandbox.snapshot import snapshot_local
from litellm.harness.types import (
    Approval,
    Capabilities,
    Event,
    Harness,
    Text,
    ToolCall,
    ToolResult,
)

ALL_MODES = frozenset({"read-only", "ask", "edit", "full"})
FULL_CAPS = Capabilities(
    structured_output=True,
    tool_approval=True,
    tool_filtering=True,
    history=True,
    custom_tools=True,
    skills=True,
    resume=True,
    permission_modes=ALL_MODES,
)
NARROW_CAPS = Capabilities(
    structured_output=False,
    tool_approval=False,
    tool_filtering=False,
    history=False,
    custom_tools=False,
    skills=False,
    resume=False,
    permission_modes=frozenset({"read-only", "full"}),
)


class FakeConfig(BaseHarnessConfig):
    """Declares the fake harness; per-test subclasses override capabilities."""

    harness: ClassVar[Harness] = Harness.CLAUDE_CODE
    options_type: ClassVar[type] = ClaudeCodeOptions
    capabilities: ClassVar[Capabilities] = FULL_CAPS
    uses_model_endpoint: ClassVar[bool] = True


Script = Callable[["FakeAdapter", SessionContext, str], AsyncIterator[Event]]


class FakeSandbox:
    """A LocalSandbox-like object over a temp dir; no subprocesses."""

    def __init__(self, workdir: str) -> None:
        self.workdir = workdir
        self.closed = False

    def _path(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.workdir, path)

    async def exec(self, cmd: list[str], *, env: Any = None, cwd: Any = None) -> Any:
        raise NotImplementedError

    async def run(
        self, cmd: list[str], *, env: Any = None, cwd: Any = None, timeout: Any = None
    ) -> CompletedRun:
        return CompletedRun(stdout="", stderr="", exit_code=0)

    async def read(self, path: str) -> bytes:
        with open(self._path(path), "rb") as fh:
            return fh.read()

    async def write(self, path: str, data: bytes) -> None:
        full = self._path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(data)

    def host_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    async def which(self, binary: str) -> str | None:
        return None

    async def snapshot(self) -> dict[str, str]:
        return await snapshot_local(self.workdir)

    async def close(self) -> None:
        self.closed = True


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost
        self.calls += 1


class FakeEndpoint:
    """Stands in for ModelEndpoint; records every instance."""

    instances: ClassVar[list[FakeEndpoint]] = []

    def __init__(self, harness: Harness, model: Any, gateway: Any, **kwargs: Any):
        self.harness = harness
        self.model = model
        self.gateway = gateway
        self.kwargs = kwargs
        self.usage = FakeUsage()
        self.url = "http://127.0.0.1:1"
        self.token = "tok"
        self.entered = False
        self.exited = False
        FakeEndpoint.instances.append(self)

    async def __aenter__(self) -> FakeEndpoint:
        self.entered = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True


async def script_hello(
    adapter: FakeAdapter, ctx: SessionContext, prompt: str
) -> AsyncIterator[Event]:
    yield Text("hello ")
    yield ToolCall(id="t1", name="bash", native_name="Bash", input={"cmd": "ls"})
    yield ToolResult(id="t1", output="a.txt")
    yield Text("world")
    if ctx.endpoint is not None:
        ctx.endpoint.usage.add(10, 5, 0.25)
    else:
        ctx.input_tokens += 10
        ctx.output_tokens += 5
        ctx.cost += 0.25
        ctx.calls += 1


class FakeAdapter(BaseHarnessHandler):
    """Configurable adapter; subclass per test and set `script` / `caps`."""

    harness: ClassVar[Harness] = Harness.CLAUDE_CODE
    options_type: ClassVar[type] = ClaudeCodeOptions
    capabilities: ClassVar[Capabilities] = FULL_CAPS
    uses_endpoint: ClassVar[bool] = True
    script: ClassVar[Script] = script_hello
    instances: ClassVar[list[FakeAdapter]] = []

    def __init__(self, config: BaseHarnessConfig | None = None) -> None:
        self.config = config if config is not None else FakeConfig()
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.resumed_with: str | None = None
        self.approvals: list[tuple[bool, str]] = []
        type(self).instances.append(self)

    async def start(self, ctx: SessionContext) -> None:
        self.calls.append("start")

    async def turn(self, ctx: SessionContext, prompt: str) -> AsyncIterator[Event]:
        self.calls.append("turn")
        self.prompts.append(prompt)
        async for event in type(self).script(self, ctx, prompt):
            yield event

    async def stop(self, ctx: SessionContext) -> None:
        self.calls.append("stop")

    def native_session_id(self) -> str | None:
        return "native-123"

    async def resume(self, ctx: SessionContext, native_session_id: str) -> None:
        self.calls.append("resume")
        self.resumed_with = native_session_id

    async def history(self, ctx: SessionContext) -> list[dict[str, Any]]:
        return [{"role": "user", "content": p} for p in self.prompts]


async def script_approval(
    adapter: FakeAdapter, ctx: SessionContext, prompt: str
) -> AsyncIterator[Event]:
    approval = Approval(tool="bash", input={"cmd": "rm"})
    yield approval
    decision = await approval.wait()
    adapter.approvals.append(decision)
    yield Text("allowed" if decision[0] else "denied")


def install_adapter(
    monkeypatch: pytest.MonkeyPatch,
    script: Script = script_hello,
    caps: Capabilities = FULL_CAPS,
    uses_endpoint: bool = True,
) -> type[FakeAdapter]:
    """Register a FakeAdapter subclass for every harness and fake the endpoint."""
    adapter_cls = type(
        "TestAdapter",
        (FakeAdapter,),
        {
            "script": staticmethod(script),
            "capabilities": caps,
            "uses_endpoint": uses_endpoint,
            "instances": [],
        },
    )
    config_cls = type(
        "TestConfig",
        (FakeConfig,),
        {"capabilities": caps, "uses_model_endpoint": uses_endpoint},
    )
    monkeypatch.setattr(runtime, "get_harness_config", lambda harness: config_cls())
    monkeypatch.setattr(
        runtime, "get_harness_handler", lambda config: adapter_cls(config)
    )
    monkeypatch.setattr(runtime, "ModelEndpoint", FakeEndpoint)
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    FakeEndpoint.instances = []
    return adapter_cls


async def wait_forever() -> None:
    await asyncio.Event().wait()
