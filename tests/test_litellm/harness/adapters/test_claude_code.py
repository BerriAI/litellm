"""Unit tests for the Claude Code adapter. No network, no real CLI.

Fixtures under fixtures/claude_code/ are sanitized stream-json recorded from
Claude Code 2.1.285 through a LiteLLM gateway.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from litellm.harness.adapters.base import SessionContext
from litellm.harness.adapters.claude_code import (
    ClaudeCodeAdapter,
    StreamState,
    build_command,
    extract_last_json_object,
    native_disallowed_tools,
    parse_stream_json_line,
    stringify_tool_output,
)
from litellm.harness.errors import (
    HarnessError,
    HarnessInstallFailed,
    OptionsMismatch,
)
from litellm.harness.options import ClaudeCodeOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.types import (
    Compaction,
    Harness,
    Reasoning,
    Text,
    ToolCall,
    ToolResult,
)

FIXTURES = Path(__file__).parent / "fixtures" / "claude_code"
SESSION_ID = "5ef64ff1-d2af-4c38-a7ca-17b4a9d07d34"
TOKEN = "per-session-token-abc"
PORT = 53211


def fixture_lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


def parse_fixture(name: str) -> tuple[list[Any], StreamState]:
    state = StreamState()
    events: list[Any] = []
    for line in fixture_lines(name):
        events.extend(parse_stream_json_line(line, state))
    return events, state


class FakeEndpoint:
    port = PORT
    token = TOKEN


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, exit_code: int) -> None:
        self.stdin_data = bytearray()
        self.stdin_closed = False
        self.killed = False
        self._exit_code = exit_code
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self.stdin = FakeStdin(self)

    async def wait(self) -> int:
        return self._exit_code

    async def kill(self) -> None:
        self.killed = True


class FakeStdin:
    def __init__(self, proc: FakeProcess) -> None:
        self._proc = proc

    def write(self, data: bytes) -> None:
        self._proc.stdin_data.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._proc.stdin_closed = True


class FakeSandbox:
    def __init__(
        self,
        workdir: str,
        outputs: list[tuple[str, bytes, int]],
        binary: str | None = "/usr/bin/claude",
        tempdir: str | None = None,
    ) -> None:
        self.workdir = workdir
        self.binary = binary
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []
        self.procs: list[FakeProcess] = []
        self.written: dict[str, bytes] = {}
        self._tempdir = tempdir or os.path.join(workdir, "_cfg")

    async def exec(
        self,
        cmd: list[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> FakeProcess:
        self.calls.append({"cmd": cmd, "env": dict(env or {}), "cwd": cwd})
        fixture, stderr, code = self.outputs.pop(0)
        stdout = (FIXTURES / fixture).read_bytes() if fixture else b""
        proc = FakeProcess(stdout, stderr, code)
        self.procs.append(proc)
        return proc

    async def run(self, cmd: list[str], **kwargs: Any) -> CompletedRun:
        return CompletedRun(stdout="", stderr="", exit_code=0)

    async def read(self, path: str) -> bytes:
        return self.written[path]

    async def write(self, path: str, data: bytes) -> None:
        self.written[path] = data

    def host_url(self, port: int) -> str:
        return f"http://host.docker.internal:{port}"

    async def which(self, binary: str) -> str | None:
        return self.binary

    async def tempdir(self) -> str:
        return self._tempdir

    async def snapshot(self) -> dict[str, str]:
        return {}

    async def close(self) -> None:
        return None


class Answer(BaseModel):
    answer: int
    word: str


def make_ctx(sandbox: FakeSandbox, **overrides: Any) -> SessionContext:
    values: dict[str, Any] = {
        "harness": Harness.CLAUDE_CODE,
        "sandbox": sandbox,
        "session_id": "hs_1",
        "model": "claude-haiku-4-5-20251001",
        "endpoint": FakeEndpoint(),
        **overrides,
    }
    return SessionContext(**values)


async def run_turn(adapter: ClaudeCodeAdapter, ctx: SessionContext, prompt: str):
    return [event async for event in adapter.turn(ctx, prompt)]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_success_fixture_events():
    events, state = parse_fixture("success_tools.jsonl")
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "Reasoning",
        "ToolCall",
        "ToolResult",
        "Reasoning",
        "ToolCall",
        "ToolResult",
        "Reasoning",
        "Text",
    ]
    write_call, read_call = events[1], events[4]
    assert write_call == ToolCall(
        id="toolu_01DFhmKzT5x1NzxuestG2Hkj",
        name="write",
        native_name="Write",
        input={"file_path": "/workspace/hello.txt", "content": "hi"},
        builtin=True,
    )
    assert read_call.name == "read" and read_call.native_name == "Read"
    assert events[2].id == write_call.id and events[2].is_error is False
    assert events[5].output == "1\thi"
    assert state.session_id == SESSION_ID
    assert state.result_seen and not state.is_error
    assert state.final_text.startswith("Done. Created `hello.txt`")


def test_parse_api_error_fixture_skips_synthetic_text():
    events, state = parse_fixture("api_error.jsonl")
    assert events == []
    assert state.is_error
    assert "no healthy deployments" in (state.result_text or "")


def test_parse_max_turns_fixture():
    events, state = parse_fixture("max_turns.jsonl")
    assert [e.native_name for e in events if isinstance(e, ToolCall)] == [
        "Write",
        "Write",
        "Write",
    ]
    assert state.is_error and state.result_text is None
    assert state.errors == ["Reached maximum number of turns (1)"]


def test_parse_structured_output_fixture():
    _, state = parse_fixture("structured_output.jsonl")
    assert state.structured_output == {"answer": 5, "word": "sum"}


def test_parse_compaction_and_garbage():
    state = StreamState()
    line = json.dumps(
        {
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": {"trigger": "auto", "pre_tokens": 1234},
        }
    )
    assert parse_stream_json_line(line, state) == [
        Compaction(tokens_before=1234, tokens_after=None)
    ]
    assert parse_stream_json_line("not json", state) == []
    assert parse_stream_json_line("", state) == []
    assert parse_stream_json_line("[1,2]", state) == []


def test_parse_skips_subagent_messages_and_maps_errors():
    state = StreamState()
    sub = {
        "type": "assistant",
        "parent_tool_use_id": "toolu_parent",
        "message": {"content": [{"type": "text", "text": "inner"}]},
    }
    assert parse_stream_json_line(json.dumps(sub), state) == []
    err = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "is_error": True,
                    "content": [{"type": "text", "text": "boom"}],
                }
            ]
        },
    }
    assert parse_stream_json_line(json.dumps(err), state) == [
        ToolResult(id="t1", output="boom", is_error=True)
    ]


def test_parse_thinking_and_mcp_tools():
    state = StreamState()
    msg = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "tool_use", "id": "t", "name": "mcp__x__y", "input": {}},
                {"type": "tool_use", "id": "u", "name": "MultiEdit", "input": {}},
            ]
        },
    }
    events = parse_stream_json_line(json.dumps(msg), state)
    assert events[0] == Reasoning(delta="hmm")
    assert events[1].name == "mcp__x__y" and events[1].builtin is False
    assert events[2].name == "edit"


def test_stringify_tool_output_variants():
    assert stringify_tool_output(None) == ""
    assert stringify_tool_output("x") == "x"
    assert stringify_tool_output([{"type": "text", "text": "a"}, "b"]) == "a\nb"
    assert stringify_tool_output({"k": 1}) == '{"k": 1}'


def test_extract_last_json_object():
    text = 'first {"a": 1} then {not json} and finally {"b": {"c": 2}}'
    assert json.loads(extract_last_json_object(text) or "") == {"b": {"c": 2}}
    assert extract_last_json_object("no json here") is None


# ---------------------------------------------------------------------------
# Command / env
# ---------------------------------------------------------------------------


def test_native_disallowed_tools_mapping():
    assert native_disallowed_tools(["edit", "bash", "Task", "edit"]) == [
        "Edit",
        "MultiEdit",
        "Bash",
        "Task",
    ]


@pytest.mark.parametrize(
    "permissions,native",
    [
        ("read-only", "plan"),
        ("edit", "acceptEdits"),
        ("full", "bypassPermissions"),
    ],
)
def test_build_command_permission_modes(permissions, native):
    cmd = build_command(
        "claude",
        model="m",
        permissions=permissions,
        system_prompt=None,
        max_turns=None,
        disable_tools=(),
        resume_session_id=None,
        isolated_config=False,
    )
    assert cmd[cmd.index("--permission-mode") + 1] == native
    assert "--resume" not in cmd and "--setting-sources" not in cmd


@pytest.mark.asyncio
async def test_start_and_turn_env_and_command(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("success_tools.jsonl", b"", 0)])
    ctx = make_ctx(
        sandbox,
        instructions="Be terse.",
        disable_tools=["bash", "web_search"],
        options=ClaudeCodeOptions(max_turns=7, small_model="small-m", env={"X": "1"}),
    )
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    events = await run_turn(adapter, ctx, "do the thing")

    call = sandbox.calls[0]
    env, cmd = call["env"], call["cmd"]
    assert env["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_BASE_URL"] == f"http://host.docker.internal:{PORT}"
    assert env["ANTHROPIC_MODEL"] == "claude-haiku-4-5-20251001"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "small-m"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "_cfg")
    assert env["DISABLE_TELEMETRY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["X"] == "1"
    assert cmd[:7] == [
        "/usr/bin/claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--input-format",
        "text",
    ]
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-4-5-20251001"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    assert cmd[cmd.index("--setting-sources") + 1] == "user"
    assert cmd[cmd.index("--append-system-prompt") + 1] == "Be terse."
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    assert cmd[cmd.index("--disallowedTools") + 1] == "Bash,WebSearch"
    assert "--resume" not in cmd

    proc = sandbox.procs[0]
    assert bytes(proc.stdin_data) == b"do the thing" and proc.stdin_closed
    assert any(isinstance(e, Text) for e in events)
    assert ctx.final_text.startswith("Done.")
    assert adapter.native_session_id() == SESSION_ID


@pytest.mark.asyncio
async def test_small_model_defaults_to_model(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("success_tools.jsonl", b"", 0)])
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    await run_turn(adapter, ctx, "hi")
    env = sandbox.calls[0]["env"]
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_second_turn_resumes_session(tmp_path):
    sandbox = FakeSandbox(
        str(tmp_path),
        [("success_tools.jsonl", b"", 0), ("resume_turn.jsonl", b"", 0)],
    )
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    await run_turn(adapter, ctx, "one")
    await run_turn(adapter, ctx, "two")
    cmd = sandbox.calls[1]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == SESSION_ID
    assert ctx.final_text == "hello.txt"


@pytest.mark.asyncio
async def test_resume_sets_native_session_id(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("resume_turn.jsonl", b"", 0)])
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    await adapter.resume(ctx, "prior-session")
    assert adapter.native_session_id() == "prior-session"
    await run_turn(adapter, ctx, "again")
    cmd = sandbox.calls[0]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == "prior-session"


@pytest.mark.asyncio
async def test_missing_binary_raises_install_failed(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [], binary=None)
    with pytest.raises(HarnessInstallFailed, match="claude"):
        await ClaudeCodeAdapter().start(make_ctx(sandbox))


@pytest.mark.asyncio
async def test_missing_endpoint_raises(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [])
    with pytest.raises(HarnessError):
        await ClaudeCodeAdapter().start(make_ctx(sandbox, endpoint=None))


@pytest.mark.asyncio
async def test_options_env_cannot_override_credentials(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [])
    options = ClaudeCodeOptions(env={"ANTHROPIC_API_KEY": "sk-real"})
    with pytest.raises(OptionsMismatch, match="ANTHROPIC_API_KEY"):
        await ClaudeCodeAdapter().start(make_ctx(sandbox, options=options))


@pytest.mark.asyncio
async def test_api_error_raises_runtime_error_with_stderr(tmp_path):
    stderr = b"[claude-code:unrecognized_model] bad model\n"
    sandbox = FakeSandbox(str(tmp_path), [("api_error.jsonl", stderr, 1)])
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    with pytest.raises(RuntimeError) as info:
        await run_turn(adapter, ctx, "hi")
    assert "no healthy deployments" in str(info.value)
    assert "unrecognized_model" in str(info.value)


@pytest.mark.asyncio
async def test_max_turns_raises_runtime_error(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("max_turns.jsonl", b"", 1)])
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    with pytest.raises(RuntimeError, match="maximum number of turns"):
        await run_turn(adapter, ctx, "hi")


@pytest.mark.asyncio
async def test_nonzero_exit_without_result_raises(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("", b"segfault\n", 139)])
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    with pytest.raises(RuntimeError, match="code 139.*no result event") as info:
        await run_turn(adapter, ctx, "hi")
    assert "segfault" in str(info.value)


@pytest.mark.asyncio
async def test_structured_output_prompt_and_json(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("structured_output.jsonl", b"", 0)])
    ctx = make_ctx(sandbox, output=Answer, instructions="Base.")
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    await run_turn(adapter, ctx, "2+3?")
    cmd = sandbox.calls[0]["cmd"]
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert prompt.startswith("Base.\n\n")
    assert json.dumps(Answer.model_json_schema()) in prompt
    assert json.loads(ctx.output_json or "") == {"answer": 5, "word": "sum"}


@pytest.mark.asyncio
async def test_structured_output_falls_back_to_final_text(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("resume_turn.jsonl", b"", 0)])
    ctx = make_ctx(sandbox, output=Answer)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    await run_turn(adapter, ctx, "hi")
    assert ctx.output_json is None  # "hello.txt" holds no JSON object


@pytest.mark.asyncio
async def test_skills_copied_into_private_config(tmp_path):
    skill = tmp_path / "skills_src" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nSay DEMO.\n")
    (skill / "scripts" / "run.sh").write_text("echo hi\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    sandbox = FakeSandbox(str(workdir), [], tempdir="/cfg")
    await ClaudeCodeAdapter().start(make_ctx(sandbox, skills=[str(skill)]))
    assert sandbox.written == {
        "/cfg/skills/demo/SKILL.md": b"---\nname: demo\n---\nSay DEMO.\n",
        f"/cfg/skills/demo/{os.path.join('scripts', 'run.sh')}": b"echo hi\n",
    }


@pytest.mark.asyncio
async def test_skill_without_manifest_rejected(tmp_path):
    skill = tmp_path / "bad"
    skill.mkdir()
    sandbox = FakeSandbox(str(tmp_path), [])
    with pytest.raises(HarnessError, match="SKILL.md"):
        await ClaudeCodeAdapter().start(make_ctx(sandbox, skills=[str(skill)]))


@pytest.mark.asyncio
async def test_stop_kills_live_process(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("success_tools.jsonl", b"", 0)])
    ctx = make_ctx(sandbox)
    adapter = ClaudeCodeAdapter()
    await adapter.start(ctx)
    stream = adapter.turn(ctx, "hi")
    await stream.__anext__()
    proc = sandbox.procs[0]
    await adapter.stop(ctx)
    assert proc.killed
    await stream.aclose()
    await adapter.stop(ctx)  # safe twice


def test_capabilities_match_spec():
    caps = ClaudeCodeAdapter.capabilities
    assert ClaudeCodeAdapter.harness is Harness.CLAUDE_CODE
    assert ClaudeCodeAdapter.options_type is ClaudeCodeOptions
    assert caps.structured_output and caps.tool_filtering and caps.skills
    assert caps.resume
    assert not (caps.tool_approval or caps.custom_tools or caps.history)
    assert caps.permission_modes == frozenset({"read-only", "edit", "full"})
