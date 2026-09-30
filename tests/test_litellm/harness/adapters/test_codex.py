import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import pytest
from pydantic import BaseModel

from litellm.harness.adapters.base import SessionContext
from litellm.harness.adapters.codex import (
    CodexAdapter,
    CodexParseState,
    parse_codex_event,
    strict_json_schema,
    toml_value,
    validate_config,
)
from litellm.harness.errors import HarnessInstallFailed, OptionsMismatch
from litellm.harness.options import ClaudeCodeOptions, CodexOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.sandbox.docker import DockerSandbox
from litellm.harness.types import Harness, Reasoning, Text, ToolCall, ToolResult

FIXTURES = Path(__file__).parent / "fixtures" / "codex"
TOKEN = "tok-secret-123"


def load_fixture(name: str) -> list[dict]:
    return [
        json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line
    ]


def parse_all(name: str, state: Optional[CodexParseState] = None):
    state = state or CodexParseState()
    events = []
    for obj in load_fixture(name):
        events.extend(parse_codex_event(obj, state))
    return events, state


# --------------------------------------------------------------------------- fakes


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", exit_code: int = 0):
        self.stdin = FakeStdin()
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self._exit_code = exit_code
        self.killed = False

    async def wait(self) -> int:
        return self._exit_code

    async def kill(self) -> None:
        self.killed = True


class FakeStdin:
    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeSandbox:
    workdir: str = "/work"
    has_codex: bool = True
    outputs: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    execs: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    processes: list = field(default_factory=list)

    async def exec(self, cmd, *, env=None, cwd=None):
        self.execs.append({"cmd": cmd, "env": dict(env or {}), "cwd": cwd})
        proc = self.outputs.pop(0)
        self.processes.append(proc)
        return proc

    async def run(self, cmd, *, env=None, cwd=None, timeout=None):
        self.runs.append(cmd)
        return CompletedRun(stdout="", stderr="", exit_code=0)

    async def read(self, path):
        return self.files[path]

    async def write(self, path, data):
        self.files[path] = data

    def host_url(self, port):
        return f"http://127.0.0.1:{port}"

    async def which(self, binary):
        return f"/usr/bin/{binary}" if self.has_codex else None

    async def tempdir(self):
        return "/tmp/codex-home"

    async def snapshot(self):
        return {}

    async def close(self):
        return None


@dataclass
class FakeEndpoint:
    port: int = 4555
    token: str = TOKEN


class Answer(BaseModel):
    file: str
    content: str


class Nested(BaseModel):
    answer: Answer
    tags: list[str] = []
    note: Optional[str] = None


def make_ctx(sandbox, **kwargs) -> SessionContext:
    return SessionContext(
        harness=Harness.CODEX,
        sandbox=sandbox,
        session_id="s1",
        model=kwargs.pop("model", "gpt-5.4"),
        endpoint=kwargs.pop("endpoint", FakeEndpoint()),
        **kwargs,
    )


def fixture_proc(name: str, **kwargs) -> FakeProcess:
    return FakeProcess((FIXTURES / name).read_bytes(), **kwargs)


def config_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]


async def collect(adapter, ctx, prompt):
    return [e async for e in adapter.turn(ctx, prompt)]


# --------------------------------------------------------------------------- parsing


def test_parse_bash_turn():
    events, state = parse_all("turn1_bash.jsonl")
    assert state.thread_id == "01a0f341-fe37-7072-93b3-055358e8147f"
    assert [type(e) for e in events] == [Text, ToolCall, ToolResult, Text]
    call, result = events[1], events[2]
    assert call.name == "bash" and call.native_name == "command_execution"
    assert call.builtin is True
    assert "hello.txt" in call.input["command"]
    assert result.id == call.id == "item_1"
    assert result.output == "hi\n" and result.is_error is False
    assert state.final_text.startswith("Done")
    assert not state.failed


def test_parse_reasoning():
    events, state = parse_all("reasoning.jsonl")
    assert isinstance(events[0], Reasoning) and "391" in events[0].delta
    assert events[1] == Text(delta="391")
    assert state.final_text == "391"


def test_parse_turn_failed():
    events, state = parse_all("turn_failed.jsonl")
    assert events == []
    assert state.failed
    assert "no healthy deployments" in state.error


def test_parse_file_change_and_mcp_and_web_search():
    state = CodexParseState()
    change = {
        "id": "i1",
        "type": "file_change",
        "changes": [{"path": "a.txt", "kind": "add"}],
        "status": "completed",
    }
    events = parse_codex_event({"type": "item.completed", "item": change}, state)
    assert events[0] == ToolCall(
        id="i1",
        name="edit",
        native_name="apply_patch",
        input={"changes": [{"path": "a.txt", "kind": "add"}]},
    )
    assert events[1] == ToolResult(id="i1", output="add a.txt", is_error=False)

    mcp = {
        "id": "i2",
        "type": "mcp_tool_call",
        "server": "docs",
        "tool": "search",
        "arguments": {"q": "x"},
        "status": "in_progress",
    }
    started = parse_codex_event({"type": "item.started", "item": mcp}, state)
    assert started == [
        ToolCall(
            id="i2",
            name="docs.search",
            native_name="search",
            input={"q": "x"},
            builtin=False,
        )
    ]
    done = {**mcp, "status": "failed", "error": {"message": "boom"}}
    assert parse_codex_event({"type": "item.completed", "item": done}, state) == [
        ToolResult(id="i2", output="boom", is_error=True)
    ]

    web = {"id": "i3", "type": "web_search", "query": "litellm"}
    events = parse_codex_event({"type": "item.completed", "item": web}, state)
    assert events[0].name == "web_search" and events[0].input == {"query": "litellm"}


def test_parse_failed_command_is_error_and_unknown_events_ignored():
    state = CodexParseState()
    item = {
        "id": "c",
        "type": "command_execution",
        "command": "false",
        "aggregated_output": "",
        "exit_code": 1,
        "status": "failed",
    }
    events = parse_codex_event({"type": "item.completed", "item": item}, state)
    assert events[1].is_error is True
    assert (
        parse_codex_event(
            {"type": "turn.completed", "usage": {"input_tokens": 5}}, state
        )
        == []
    )
    assert (
        parse_codex_event(
            {"type": "item.completed", "item": {"type": "todo_list"}}, state
        )
        == []
    )


# --------------------------------------------------------------------------- helpers


def test_strict_json_schema_recursive():
    schema = strict_json_schema(Nested.model_json_schema())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["answer", "tags", "note"]
    assert schema["properties"]["answer"] == {"$ref": "#/$defs/Answer"}
    assert "default" not in schema["properties"]["tags"]
    answer = schema["$defs"]["Answer"]
    assert answer["additionalProperties"] is False
    assert answer["required"] == ["file", "content"]


def test_validate_config_rejects_managed_keys():
    for key in (
        "model_provider",
        "model_providers.x.base_url",
        "approval_policy",
        "sandbox_mode",
        "mcp_servers.a",
    ):
        with pytest.raises(OptionsMismatch):
            validate_config({key: "x"})
    assert validate_config(
        {
            "sandbox_workspace_write.network_access": True,
            "notice": {"a b": 1},
            "x": ["y"],
        }
    ) == [
        "sandbox_workspace_write.network_access=true",
        'notice={"a b" = 1}',
        'x=["y"]',
    ]
    assert toml_value('say "hi"') == '"say \\"hi\\""'


# --------------------------------------------------------------------------- adapter


async def test_start_missing_binary():
    ctx = make_ctx(FakeSandbox(has_codex=False))
    with pytest.raises(HarnessInstallFailed, match="codex"):
        await CodexAdapter().start(ctx)


async def test_start_rejects_managed_config_and_wrong_options():
    with pytest.raises(OptionsMismatch):
        await CodexAdapter().start(
            make_ctx(
                FakeSandbox(), options=CodexOptions(config={"model_provider": "openai"})
            )
        )
    with pytest.raises(OptionsMismatch):
        await CodexAdapter().start(make_ctx(FakeSandbox(), options=ClaudeCodeOptions()))


async def test_start_writes_skills_and_schema(tmp_path):
    skill = tmp_path / "my-skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")
    (skill / "scripts" / "run.sh").write_text("echo hi")
    sbx = FakeSandbox()
    await CodexAdapter().start(make_ctx(sbx, skills=[str(skill)], output=Answer))
    assert sbx.files["/tmp/codex-home/skills/my-skill/SKILL.md"].startswith(b"---")
    assert sbx.files["/tmp/codex-home/skills/my-skill/scripts/run.sh"] == b"echo hi"
    schema = json.loads(sbx.files["/tmp/codex-home/output_schema.json"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["file", "content"]


async def test_first_turn_then_resume_argv_env():
    sbx = FakeSandbox(
        outputs=[
            fixture_proc("turn1_bash.jsonl"),
            fixture_proc("turn2_resume_apply_patch.jsonl"),
        ]
    )
    adapter = CodexAdapter()
    ctx = make_ctx(
        sbx,
        instructions="Be terse.",
        options=CodexOptions(
            reasoning_effort="low",
            config={"sandbox_workspace_write.network_access": True},
        ),
    )
    await adapter.start(ctx)
    events = await collect(adapter, ctx, "create hello.txt")

    first = sbx.execs[0]
    argv, env = first["cmd"], first["env"]
    assert argv[:4] == ["codex", "exec", "--json", "--skip-git-repo-check"]
    assert argv[-1] == "-" and argv[argv.index("-C") + 1] == "/work"
    assert argv[argv.index("-m") + 1] == "gpt-5.4"
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    cfg = config_values(argv)
    assert "model_provider=litellm" in cfg
    assert 'model_providers.litellm.base_url="http://127.0.0.1:4555/v1"' in cfg
    assert "model_providers.litellm.env_key=LITELLM_HARNESS_TOKEN" in cfg
    assert "model_providers.litellm.wire_api=responses" in cfg
    assert "approval_policy=never" in cfg
    assert "model_reasoning_effort=low" in cfg
    assert "web_search=disabled" in cfg
    assert 'developer_instructions="Be terse."' in cfg
    assert "sandbox_workspace_write.network_access=true" in cfg
    assert not any(TOKEN in a for a in argv)
    assert env["LITELLM_HARNESS_TOKEN"] == TOKEN
    assert env["CODEX_HOME"] == "/tmp/codex-home"
    assert sbx.processes[0].stdin.data == b"create hello.txt"
    assert sbx.processes[0].stdin.closed

    assert isinstance(events[-1], Text)
    assert ctx.final_text.startswith("Done")
    assert adapter.native_session_id() == "01a0f341-fe37-7072-93b3-055358e8147f"

    await collect(adapter, ctx, "edit it")
    argv2 = sbx.execs[1]["cmd"]
    assert argv2[:4] == [
        "codex",
        "exec",
        "resume",
        "01a0f341-fe37-7072-93b3-055358e8147f",
    ]
    assert "--sandbox" not in argv2 and "-C" not in argv2
    assert 'sandbox_mode="workspace-write"' in config_values(argv2)
    assert ctx.final_text == "done"


async def test_permission_modes():
    adapter = CodexAdapter()
    ro = make_ctx(FakeSandbox(), permissions="read-only")
    await adapter.start(ro)
    argv = adapter.build_argv(ro)
    assert argv[argv.index("--sandbox") + 1] == "read-only"

    docker = DockerSandbox.__new__(DockerSandbox)
    docker.workdir = "/workspace"
    full = make_ctx(docker, permissions="full")
    adapter._thread_id = None
    argv = adapter.build_argv(full)
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert "--sandbox" not in argv

    web = make_ctx(FakeSandbox(), options=CodexOptions(web_search=True))
    assert "web_search=live" in config_values(adapter.build_argv(web))


async def test_resume_sets_thread_id():
    adapter = CodexAdapter()
    ctx = make_ctx(FakeSandbox())
    await adapter.start(ctx)
    await adapter.resume(ctx, "thread-9")
    assert adapter.native_session_id() == "thread-9"
    assert adapter.build_argv(ctx)[:4] == ["codex", "exec", "resume", "thread-9"]


async def test_structured_output_sets_output_json():
    sbx = FakeSandbox(outputs=[fixture_proc("structured_output.jsonl")])
    adapter = CodexAdapter()
    ctx = make_ctx(sbx, output=Answer, permissions="read-only")
    await adapter.start(ctx)
    await collect(adapter, ctx, "read hello.txt")
    argv = sbx.execs[0]["cmd"]
    assert (
        argv[argv.index("--output-schema") + 1] == "/tmp/codex-home/output_schema.json"
    )
    assert Answer.model_validate_json(ctx.output_json).file == "hello.txt"


async def test_turn_failed_raises():
    sbx = FakeSandbox(outputs=[fixture_proc("turn_failed.jsonl", exit_code=1)])
    adapter = CodexAdapter()
    ctx = make_ctx(sbx)
    await adapter.start(ctx)
    with pytest.raises(RuntimeError, match="no healthy deployments"):
        await collect(adapter, ctx, "hi")


async def test_nonzero_exit_raises_with_stderr_tail():
    sbx = FakeSandbox(
        outputs=[
            FakeProcess(b"", stderr=b"Error loading config.toml: bad\n", exit_code=1)
        ]
    )
    adapter = CodexAdapter()
    ctx = make_ctx(sbx)
    await adapter.start(ctx)
    with pytest.raises(RuntimeError, match=r"code 1: Error loading config\.toml"):
        await collect(adapter, ctx, "hi")


async def test_early_close_kills_process_and_stop_is_idempotent():
    sbx = FakeSandbox(outputs=[fixture_proc("turn1_bash.jsonl")])
    adapter = CodexAdapter()
    ctx = make_ctx(sbx)
    await adapter.start(ctx)
    gen = adapter.turn(ctx, "hi")
    await gen.__anext__()
    await gen.aclose()
    assert sbx.processes[0].killed
    await adapter.stop(ctx)
    await adapter.stop(ctx)


async def test_long_jsonl_line_is_parsed():
    text = "x" * 200_000
    line = json.dumps(
        {
            "type": "item.completed",
            "item": {"id": "a", "type": "agent_message", "text": text},
        }
    )
    sbx = FakeSandbox(outputs=[FakeProcess(line.encode() + b"\n")])
    adapter = CodexAdapter()
    ctx = make_ctx(sbx)
    await adapter.start(ctx)
    events = await collect(adapter, ctx, "hi")
    assert events == [Text(delta=text)]
