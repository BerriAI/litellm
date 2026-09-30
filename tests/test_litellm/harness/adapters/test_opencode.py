import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel

from litellm.harness.adapters.base import SessionContext
from litellm.harness.adapters.opencode import (
    OPENCODE_SESSION_TITLE,
    OpenCodeAdapter,
    OpenCodeParseState,
    build_opencode_config,
    parse_opencode_event,
    permission_rules,
    validate_user_config,
)
from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessInstallFailed,
    OptionsMismatch,
)
from litellm.harness.options import CodexOptions, OpenCodeOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.types import Harness, Reasoning, Text, ToolCall, ToolResult

FIXTURES = Path(__file__).parent / "fixtures" / "opencode"
TOKEN = "tok-secret-123"
XDG_ROOT = "/home/u/.cache/litellm-harness/opencode"
SESSION = "ses_f0cb977cdffeoCMeplOiw1KY25"


def load_fixture(name: str) -> list[dict]:
    return [
        json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line
    ]


def parse_all(name: str, state: OpenCodeParseState | None = None):
    state = state or OpenCodeParseState()
    events = []
    for obj in load_fixture(name):
        events.extend(parse_opencode_event(obj, state))
    return events, state


# --------------------------------------------------------------------------- fakes


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


@dataclass
class FakeSandbox:
    workdir: str = "/work"
    has_binary: bool = True
    xdg_ok: bool = True
    outputs: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    execs: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    tempdirs: int = 0

    async def exec(self, cmd, *, env=None, cwd=None):
        self.execs.append({"cmd": cmd, "env": dict(env or {}), "cwd": cwd})
        return self.outputs.pop(0)

    async def run(self, cmd, *, env=None, cwd=None, timeout=None):
        self.runs.append(cmd)
        if self.xdg_ok:
            return CompletedRun(stdout=XDG_ROOT, stderr="", exit_code=0)
        return CompletedRun(stdout="", stderr="read-only fs", exit_code=1)

    async def read(self, path):
        return self.files[path]

    async def write(self, path, data):
        self.files[path] = data

    def host_url(self, port):
        return f"http://host.docker.internal:{port}"

    async def which(self, binary):
        return f"/usr/bin/{binary}" if self.has_binary else None

    async def tempdir(self):
        self.tempdirs += 1
        return f"/tmp/oc-{self.tempdirs}"

    async def snapshot(self):
        return {}

    async def close(self):
        return None


@dataclass
class FakeEndpoint:
    port: int = 4555
    token: str = TOKEN
    model: str | None = None


class Answer(BaseModel):
    file: str
    content: str


def make_ctx(sandbox, **kwargs) -> SessionContext:
    return SessionContext(
        harness=Harness.OPENCODE,
        sandbox=sandbox,
        session_id="s1",
        model=kwargs.pop("model", "claude-haiku-4-5-20251001"),
        endpoint=kwargs.pop("endpoint", FakeEndpoint()),
        **kwargs,
    )


def fixture_proc(name: str, **kwargs) -> FakeProcess:
    return FakeProcess((FIXTURES / name).read_bytes(), **kwargs)


async def collect(adapter, ctx, prompt):
    return [e async for e in adapter.turn(ctx, prompt)]


async def started(sandbox=None, **kwargs):
    sandbox = sandbox or FakeSandbox()
    adapter = OpenCodeAdapter()
    ctx = make_ctx(sandbox, **kwargs)
    await adapter.start(ctx)
    return adapter, ctx, sandbox


def exec_config(sandbox, index=0) -> dict:
    return json.loads(sandbox.execs[index]["env"]["OPENCODE_CONFIG_CONTENT"])


# --------------------------------------------------------------------------- parsing


def test_parse_write_read_turn():
    events, state = parse_all("turn1_write_read.jsonl")
    assert state.session_id == SESSION
    assert [type(e) for e in events] == [
        Text,
        ToolCall,
        ToolResult,
        Text,
        ToolCall,
        ToolResult,
        Text,
    ]
    write, write_result = events[1], events[2]
    assert write.name == "write" and write.native_name == "write"
    assert write.builtin is True
    assert write.input == {"filePath": "/workspace/hello.txt", "content": "hi"}
    assert write_result.id == write.id == "toolu_015FFUwEf2dazoWfCrMbMCnm"
    assert write_result.is_error is False
    read, read_result = events[4], events[5]
    assert read.name == "read" and "1: hi" in read_result.output
    assert state.final_text.startswith("Done!")
    assert state.error is None


def test_parse_skill_tool_on_continued_session():
    events, state = parse_all("turn2_session_skill.jsonl")
    assert state.session_id == SESSION
    skill = next(e for e in events if isinstance(e, ToolCall))
    assert skill.name == "skill" and skill.input == {"name": "greeter"}
    assert skill.builtin is True
    assert "PINEAPPLE" in state.final_text


def test_parse_denied_tool_is_error_result():
    events, state = parse_all("readonly_denied_bash.jsonl")
    call = next(e for e in events if isinstance(e, ToolCall))
    result = next(e for e in events if isinstance(e, ToolResult))
    assert call.native_name == "invalid" and call.input["tool"] == "bash"
    assert result.is_error is True
    assert state.final_text.startswith("FAILED")


def test_parse_api_error_records_error():
    events, state = parse_all("api_error.jsonl")
    assert events == []
    assert "no healthy deployments" in state.error


def test_parse_reasoning_and_tool_error_and_name_mapping():
    state = OpenCodeParseState()
    reasoning = parse_opencode_event(
        {"type": "reasoning", "sessionID": "s", "part": {"text": "thinking hard"}},
        state,
    )
    assert reasoning == [Reasoning(delta="thinking hard")]
    failed = parse_opencode_event(
        {
            "type": "tool_use",
            "part": {
                "tool": "bash",
                "callID": "c1",
                "state": {
                    "status": "error",
                    "input": {"command": "x"},
                    "error": "boom",
                },
            },
        },
        state,
    )
    assert failed[1] == ToolResult(id="c1", output="boom", is_error=True)
    for native, normalized in [
        ("list", "ls"),
        ("webfetch", "web_search"),
        ("glob", "glob"),
        ("grep", "grep"),
        ("apply_patch", "edit"),
    ]:
        call = parse_opencode_event(
            {
                "type": "tool_use",
                "part": {
                    "tool": native,
                    "callID": "x",
                    "state": {"status": "completed", "input": {}, "output": ""},
                },
            },
            state,
        )[0]
        assert call.name == normalized
    mcp = parse_opencode_event(
        {
            "type": "tool_use",
            "part": {
                "tool": "github_search",
                "callID": "m",
                "state": {"status": "completed", "input": {}, "output": {"a": 1}},
            },
        },
        state,
    )
    assert mcp[0].builtin is False and mcp[1].output == '{"a": 1}'
    assert state.session_id == "s"


def test_final_text_is_last_step_text():
    state = OpenCodeParseState()
    parse_opencode_event({"type": "step_start"}, state)
    parse_opencode_event({"type": "text", "part": {"text": "working"}}, state)
    parse_opencode_event({"type": "step_start"}, state)
    parse_opencode_event({"type": "text", "part": {"text": "a"}}, state)
    parse_opencode_event({"type": "text", "part": {"text": "b"}}, state)
    assert state.final_text == "a\n\nb"


# --------------------------------------------------------------------------- config


def test_permission_mapping():
    assert permission_rules("full", ()) == {"*": "allow"}
    assert permission_rules("read-only", ()) == {
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
    }
    edit = permission_rules("edit", ())
    assert edit["edit"] == "allow" and edit["bash"] == "deny"
    with pytest.raises(CapabilityUnsupported):
        permission_rules("ask", ())


def test_disable_tools_map_to_native_denies_after_wildcard():
    rules = permission_rules("full", ["bash", "web_search", "ls", "write"])
    assert list(rules)[0] == "*"
    assert rules["bash"] == "deny"
    assert rules["webfetch"] == rules["websearch"] == "deny"
    assert rules["list"] == "deny"
    assert rules["edit"] == "deny"


def test_build_config_merges_user_config_under_managed_keys():
    config = build_opencode_config(
        model="m1",
        base_url="http://h:1/v1",
        token_path="/tmp/p/token",
        permissions="full",
        user_config={"instructions": ["RULES.md"], "compaction": {"auto": False}},
        instructions_path="/tmp/p/instructions.md",
        skills_path="/tmp/p/skills",
    )
    provider = config["provider"]["litellm"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {
        "baseURL": "http://h:1/v1",
        "apiKey": "{file:/tmp/p/token}",
    }
    assert provider["models"] == {"m1": {}}
    assert config["model"] == config["small_model"] == "litellm/m1"
    assert config["enabled_providers"] == ["litellm"]
    assert config["instructions"] == ["RULES.md", "/tmp/p/instructions.md"]
    assert config["skills"] == {"paths": ["/tmp/p/skills"]}
    assert config["compaction"] == {"auto": False}


@pytest.mark.parametrize(
    "config",
    [
        {"provider": {}},
        {"model": "openai/gpt-5"},
        {"permission": {"*": "allow"}},
        {"tools": {"bash": True}},
        {"agent": {"build": {"permission": {"bash": "allow"}}}},
        {"mode": {"x": {"model": "a/b"}}},
    ],
)
def test_managed_keys_rejected(config):
    with pytest.raises(OptionsMismatch):
        validate_user_config(config)


# --------------------------------------------------------------------------- adapter


async def test_start_writes_token_only_in_private_file():
    adapter, ctx, sandbox = await started()
    token_path = "/tmp/oc-1/token"
    assert sandbox.files[token_path] == TOKEN.encode()
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(adapter, ctx, "create hello.txt containing hi then read it")
    call = sandbox.execs[0]
    assert TOKEN not in json.dumps(call["cmd"])
    assert TOKEN not in json.dumps(call["env"])
    config = exec_config(sandbox)
    assert config["provider"]["litellm"]["options"] == {
        "baseURL": "http://host.docker.internal:4555/v1",
        "apiKey": "{file:/tmp/oc-1/token}",
    }
    assert config["permission"] == {"*": "allow"}


async def test_turn_argv_env_and_session_continuation():
    adapter, ctx, sandbox = await started(
        options=OpenCodeOptions(agent="build", env={"FOO": "1"})
    )
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    events = await collect(adapter, ctx, "create hello.txt containing hi then read it")
    first = sandbox.execs[0]
    assert first["cmd"] == [
        "opencode",
        "run",
        "--format",
        "json",
        "--thinking",
        "-m",
        "litellm/claude-haiku-4-5-20251001",
        "--agent",
        "build",
        "--title",
        OPENCODE_SESSION_TITLE,
    ]
    assert first["cwd"] == "/work"
    env = first["env"]
    assert env["XDG_CONFIG_HOME"] == f"{XDG_ROOT}/config"
    assert env["XDG_DATA_HOME"] == f"{XDG_ROOT}/data"
    assert env["XDG_STATE_HOME"] == f"{XDG_ROOT}/state"
    assert env["XDG_CACHE_HOME"] == f"{XDG_ROOT}/cache"
    assert env["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    assert env["OPENCODE_CONFIG"] == "" and env["OPENCODE_PERMISSION"] == ""
    assert env["FOO"] == "1"
    assert any(isinstance(e, ToolCall) for e in events)
    assert ctx.final_text.startswith("Done!")
    assert adapter.native_session_id() == SESSION

    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(adapter, ctx, "what file did you create?")
    second = sandbox.execs[1]["cmd"]
    assert second[second.index("--session") + 1] == SESSION
    assert "--title" not in second
    assert "what file" not in " ".join(second)


async def test_prompt_is_sent_on_stdin_not_argv():
    adapter, ctx, sandbox = await started()
    proc = fixture_proc("turn1_write_read.jsonl")
    sandbox.outputs.append(proc)
    await collect(adapter, ctx, "secret prompt text")
    assert proc.stdin.data == b"secret prompt text" and proc.stdin.closed
    assert "secret prompt text" not in sandbox.execs[0]["cmd"]


async def test_resume_sets_session():
    adapter, ctx, sandbox = await started()
    await adapter.resume(ctx, "ses_prev")
    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(adapter, ctx, "hi")
    cmd = sandbox.execs[0]["cmd"]
    assert cmd[cmd.index("--session") + 1] == "ses_prev"


async def test_read_only_and_disable_tools_config():
    adapter, ctx, sandbox = await started(
        permissions="read-only", disable_tools=["grep"]
    )
    sandbox.outputs.append(fixture_proc("readonly_denied_bash.jsonl"))
    await collect(adapter, ctx, "x")
    permission = exec_config(sandbox)["permission"]
    assert permission == {
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
        "grep": "deny",
    }


async def test_instructions_and_structured_output():
    adapter, ctx, sandbox = await started(instructions="Be terse.", output=Answer)
    written = sandbox.files["/tmp/oc-1/instructions.md"].decode()
    assert written.startswith("Be terse.")
    assert '"file"' in written and "single JSON object" in written
    lines = [
        {"type": "step_start", "sessionID": "s"},
        {
            "type": "text",
            "sessionID": "s",
            "part": {"text": 'Here: {"file": "a", "content": "hi"}'},
        },
    ]
    stdout = "\n".join(json.dumps(line) for line in lines).encode()
    sandbox.outputs.append(FakeProcess(stdout))
    await collect(adapter, ctx, "x")
    assert exec_config(sandbox)["instructions"] == ["/tmp/oc-1/instructions.md"]
    assert json.loads(ctx.output_json) == {"file": "a", "content": "hi"}


async def test_skills_copied_to_private_skills_path(tmp_path):
    skill = tmp_path / "greeter"
    (skill / "ref").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: greeter\ndescription: d\n---\nbody")
    (skill / "ref" / "notes.txt").write_text("n")
    adapter, ctx, sandbox = await started(skills=[str(skill)])
    assert sandbox.files["/tmp/oc-1/skills/greeter/SKILL.md"].startswith(b"---")
    assert sandbox.files["/tmp/oc-1/skills/greeter/ref/notes.txt"] == b"n"
    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(adapter, ctx, "x")
    assert exec_config(sandbox)["skills"] == {"paths": ["/tmp/oc-1/skills"]}


async def test_xdg_root_falls_back_to_tempdir():
    adapter, ctx, sandbox = await started(FakeSandbox(xdg_ok=False))
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(adapter, ctx, "x")
    assert sandbox.execs[0]["env"]["XDG_DATA_HOME"] == "/tmp/oc-2/data"


async def test_missing_binary():
    with pytest.raises(HarnessInstallFailed, match="opencode"):
        await started(FakeSandbox(has_binary=False))


async def test_wrong_options_and_managed_config_rejected():
    with pytest.raises(OptionsMismatch):
        await started(options=CodexOptions())
    with pytest.raises(OptionsMismatch):
        await started(options=OpenCodeOptions(config={"model": "openai/x"}))


async def test_api_error_event_raises_even_on_exit_zero():
    adapter, ctx, sandbox = await started()
    sandbox.outputs.append(fixture_proc("api_error.jsonl"))
    with pytest.raises(RuntimeError, match="no healthy deployments"):
        await collect(adapter, ctx, "x")


async def test_nonzero_exit_raises_with_stderr_tail():
    adapter, ctx, sandbox = await started()
    sandbox.outputs.append(
        FakeProcess(b"", stderr=b"line1\nfatal: bad config\n", exit_code=2)
    )
    with pytest.raises(RuntimeError, match="code 2: line1\nfatal: bad config"):
        await collect(adapter, ctx, "x")


async def test_early_close_kills_process_and_stop_is_idempotent():
    adapter, ctx, sandbox = await started()
    proc = fixture_proc("turn1_write_read.jsonl")
    sandbox.outputs.append(proc)
    gen = adapter.turn(ctx, "x")
    await gen.__anext__()
    await gen.aclose()
    assert proc.killed
    await adapter.stop(ctx)
    await adapter.stop(ctx)


async def test_model_falls_back_to_endpoint_model():
    adapter, ctx, sandbox = await started(
        model=None, endpoint=FakeEndpoint(model="gw-model")
    )
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(adapter, ctx, "x")
    assert "litellm/gw-model" in sandbox.execs[0]["cmd"]
    assert exec_config(sandbox)["provider"]["litellm"]["models"] == {"gw-model": {}}


def test_endpoint_request_fixture_documents_contract():
    requests = load_fixture("endpoint_requests.jsonl")
    assert {r["path"] for r in requests} == {"/v1/chat/completions"}
    assert all(r["stream"] is True for r in requests)
    assert all(r["stream_options"] == {"include_usage": True} for r in requests)
