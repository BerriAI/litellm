"""Unit tests for the Codex harness config. No network, no real CLI.

Fixtures under fixtures/ are sanitized `codex exec --json` output recorded from
codex-cli through a LiteLLM gateway.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError

from litellm.harness.context import SessionContext
from litellm.harness.errors import HarnessError, HarnessInstallFailed, OptionsMismatch
from litellm.harness.handlers.cli_handler import CLIHarnessHandler
from litellm.harness.options import ClaudeCodeOptions, CodexOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.sandbox.docker import DockerSandbox
from litellm.harness.types import Harness, Reasoning, Text, ToolCall, ToolResult
from litellm.llms.base_llm.harness.transformation import (
    HarnessTurnError,
    HarnessTurnRequest,
)
from litellm.llms.base_llm.harness.utils import strict_json_schema
from litellm.llms.codex.harness.transformation import (
    CODEX_SCHEMA_FILENAME,
    CODEX_TOKEN_ENV,
    MANAGED_CONFIG_KEYS,
    CodexHarnessConfig,
    CodexStreamState,
    config_overrides,
    toml_key,
    toml_value,
)

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "tok-secret-123"
HOME = "/tmp/codex-home"
THREAD_ID = "01a0f341-fe37-7072-93b3-055358e8147f"


def load_fixture(name: str) -> list[dict]:
    return [
        json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line
    ]


def parse_event(obj: dict, state: CodexStreamState) -> list:
    return CodexHarnessConfig().transform_stream_line(obj, state)


def parse_all(name: str, state: Optional[CodexStreamState] = None):
    state = state or CodexHarnessConfig().create_stream_state()
    events = []
    for obj in load_fixture(name):
        events.extend(parse_event(obj, state))
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
        return CompletedRun("", "", 0)

    async def read(self, path):
        return self.files[path]

    async def write(self, path, data):
        self.files[path] = data

    def host_url(self, port):
        return f"http://127.0.0.1:{port}"

    async def which(self, binary):
        return f"/usr/bin/{binary}" if self.has_codex else None

    async def tempdir(self):
        return HOME

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


def request_for(
    ctx: SessionContext, native_session_id: Optional[str] = None, prompt: str = "hi"
) -> HarnessTurnRequest:
    cfg = CodexHarnessConfig()
    setup = cfg.transform_session_setup(ctx, HOME)
    return cfg.transform_turn_request(ctx, setup, HOME, prompt, native_session_id)


def argv_for(ctx: SessionContext, native_session_id: Optional[str] = None) -> list:
    return list(request_for(ctx, native_session_id).argv)


def fixture_proc(name: str, **kwargs) -> FakeProcess:
    return FakeProcess((FIXTURES / name).read_bytes(), **kwargs)


def config_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]


def make_handler() -> CLIHarnessHandler:
    return CLIHarnessHandler(CodexHarnessConfig())


async def collect(handler, ctx, prompt):
    return [e async for e in handler.turn(ctx, prompt)]


# --------------------------------------------------------------------------- parsing


def test_parse_bash_turn():
    events, state = parse_all("turn1_bash.jsonl")
    assert state.thread_id == THREAD_ID
    assert CodexHarnessConfig().get_native_session_id(state) == THREAD_ID
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
    state = CodexStreamState()
    change = {
        "id": "i1",
        "type": "file_change",
        "changes": [{"path": "a.txt", "kind": "add"}],
        "status": "completed",
    }
    events = parse_event({"type": "item.completed", "item": change}, state)
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
    started = parse_event({"type": "item.started", "item": mcp}, state)
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
    assert parse_event({"type": "item.completed", "item": done}, state) == [
        ToolResult(id="i2", output="boom", is_error=True)
    ]

    web = {"id": "i3", "type": "web_search", "query": "litellm"}
    events = parse_event({"type": "item.completed", "item": web}, state)
    assert events[0].name == "web_search" and events[0].input == {"query": "litellm"}


@pytest.mark.parametrize(
    ("changes", "output"),
    [
        ([{"path": "a.txt", "kind": "add"}, {"path": "b.txt", "kind": "update", "diff": "@@"}], "add a.txt\nupdate b.txt"),
        ([{"path": "only-path.txt"}, {"kind": "delete"}, {}], "only-path.txt\ndelete\n"),
        ([], ""),
        (None, ""),
    ],
)
def test_completed_file_change_lists_each_change(changes: object, output: str):
    state = CodexStreamState(started={"i1"})
    item = {"id": "i1", "type": "file_change", "changes": changes, "status": "completed"}

    assert parse_event({"type": "item.completed", "item": item}, state) == [
        ToolResult(id="i1", output=output, is_error=False)
    ]


@pytest.mark.parametrize(
    "changes",
    [["a.txt"], [{"path": "a.txt"}, None], "a.txt", {"a.txt": {"kind": "add"}}, 7],
)
def test_completed_file_change_rejects_changes_that_are_not_a_list_of_objects(changes: object):
    state = CodexStreamState(started={"i1"})
    item = {"id": "i1", "type": "file_change", "changes": changes, "status": "completed"}

    with pytest.raises(ValidationError):
        parse_event({"type": "item.completed", "item": item}, state)


def test_parse_failed_command_is_error_and_unknown_events_ignored():
    state = CodexStreamState()
    item = {
        "id": "c",
        "type": "command_execution",
        "command": "false",
        "aggregated_output": "",
        "exit_code": 1,
        "status": "failed",
    }
    events = parse_event({"type": "item.completed", "item": item}, state)
    assert events[1].is_error is True
    usage = {"type": "turn.completed", "usage": {"input_tokens": 5}}
    assert parse_event(usage, state) == []
    todo = {"type": "item.completed", "item": {"type": "todo_list"}}
    assert parse_event(todo, state) == []
    assert parse_event({"type": "item.completed", "item": "nope"}, state) == []


def test_parse_error_event_then_turn_failed():
    state = CodexStreamState()
    assert parse_event({"type": "error", "message": "reconnecting"}, state) == []
    assert state.error == "reconnecting" and not state.failed
    assert parse_event({"type": "turn.failed", "error": {"message": "x"}}, state) == []
    assert state.failed and state.error == "x"


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


def test_config_overrides_rejects_managed_keys():
    for key in (
        "model_provider",
        "model_providers.x.base_url",
        "approval_policy",
        "sandbox_mode",
        "mcp_servers.a",
    ):
        with pytest.raises(OptionsMismatch):
            config_overrides({key: "x"})
    for key in sorted(MANAGED_CONFIG_KEYS):
        with pytest.raises(OptionsMismatch, match="managed by LiteLLM"):
            config_overrides({key: "x"})
    for bad in ("", "a=b"):
        with pytest.raises(OptionsMismatch, match="Invalid"):
            config_overrides({bad: "x"})
    assert config_overrides(
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


def test_toml_value_and_key():
    assert toml_value('say "hi"') == '"say \\"hi\\""'
    assert toml_value(False) == "false"
    assert toml_value(1.5) == "1.5"
    assert toml_value(("a", 2)) == '["a", 2]'
    assert toml_key("plain_key-1") == "plain_key-1"
    assert toml_key("a b") == '"a b"'
    with pytest.raises(OptionsMismatch):
        toml_value(object())


# --------------------------------------------------------------------------- session setup / turn request


def test_session_setup_env_and_schema():
    ctx = make_ctx(FakeSandbox(), output=Answer, options=CodexOptions(env={"X": "1"}))
    setup = CodexHarnessConfig().transform_session_setup(ctx, HOME)
    assert setup.env == {"X": "1", CODEX_TOKEN_ENV: TOKEN, "CODEX_HOME": HOME}
    assert setup.persisted_dirs == [("sessions", "codex/sessions")]
    assert setup.skills_dir == "skills"
    schema = json.loads(setup.files[CODEX_SCHEMA_FILENAME])
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["file", "content"]
    no_schema = CodexHarnessConfig().transform_session_setup(
        make_ctx(FakeSandbox()), HOME
    )
    assert no_schema.files == {}


def test_missing_endpoint_raises():
    ctx = make_ctx(FakeSandbox(), endpoint=None)
    with pytest.raises(HarnessError, match="endpoint"):
        CodexHarnessConfig().transform_session_setup(ctx, HOME)


def test_validate_environment_rejects_managed_config_and_wrong_options():
    cfg = CodexHarnessConfig()
    with pytest.raises(OptionsMismatch):
        cfg.validate_environment(
            make_ctx(
                FakeSandbox(), options=CodexOptions(config={"model_provider": "openai"})
            )
        )
    with pytest.raises(OptionsMismatch):
        cfg.validate_environment(make_ctx(FakeSandbox(), options=ClaudeCodeOptions()))


def test_first_turn_argv_env():
    ctx = make_ctx(
        FakeSandbox(),
        instructions="Be terse.",
        options=CodexOptions(
            reasoning_effort="low",
            config={"sandbox_workspace_write.network_access": True},
        ),
    )
    request = request_for(ctx, prompt="create hello.txt")
    argv, env = list(request.argv), request.env
    assert request.stdin == "create hello.txt"
    assert request.cwd == "/work"
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
    assert "model_reasoning_summary=auto" in cfg
    assert "web_search=disabled" in cfg
    assert 'developer_instructions="Be terse."' in cfg
    assert "sandbox_workspace_write.network_access=true" in cfg
    assert "--output-schema" not in argv
    assert not any(TOKEN in a for a in argv)
    assert env["LITELLM_HARNESS_TOKEN"] == TOKEN
    assert env["CODEX_HOME"] == HOME


def test_resume_argv():
    argv = argv_for(make_ctx(FakeSandbox()), THREAD_ID)
    assert argv[:4] == ["codex", "exec", "resume", THREAD_ID]
    assert "--sandbox" not in argv and "-C" not in argv
    assert 'sandbox_mode="workspace-write"' in config_values(argv)
    assert argv[-1] == "-"


def test_permission_modes():
    ro = make_ctx(FakeSandbox(), permissions="read-only")
    argv = argv_for(ro)
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert 'sandbox_mode="read-only"' in config_values(argv_for(ro, "t"))

    # The container is the boundary: DockerSandbox opts out of codex's own sandbox.
    assert DockerSandbox.is_container is True
    container = FakeSandbox(workdir="/workspace")
    container.is_container = True
    argv = argv_for(make_ctx(container, permissions="full"))
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert "--sandbox" not in argv
    assert argv[argv.index("-C") + 1] == "/workspace"
    resumed = argv_for(make_ctx(container, permissions="full"), "t")
    assert "--dangerously-bypass-approvals-and-sandbox" in resumed
    assert not any(v.startswith("sandbox_mode=") for v in config_values(resumed))

    # read-only wins even inside a container
    argv = argv_for(make_ctx(container, permissions="read-only"))
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv

    web = make_ctx(FakeSandbox(), options=CodexOptions(web_search=True))
    assert "web_search=live" in config_values(argv_for(web))


def test_structured_output_argv():
    argv = argv_for(make_ctx(FakeSandbox(), output=Answer))
    assert argv[argv.index("--output-schema") + 1] == f"{HOME}/{CODEX_SCHEMA_FILENAME}"


def test_no_model_omits_flag():
    assert "-m" not in argv_for(make_ctx(FakeSandbox(), model=None))


# --------------------------------------------------------------------------- turn response


def test_turn_response_failed_raises():
    _, state = parse_all("turn_failed.jsonl")
    with pytest.raises(HarnessTurnError, match="no healthy deployments"):
        CodexHarnessConfig().transform_turn_response(
            make_ctx(FakeSandbox()), state, 1, []
        )


def test_turn_response_nonzero_exit_uses_stderr_tail():
    with pytest.raises(HarnessTurnError, match=r"code 1: Error loading config\.toml"):
        CodexHarnessConfig().transform_turn_response(
            make_ctx(FakeSandbox()),
            CodexStreamState(),
            1,
            ["Error loading config.toml: bad", ""],
        )
    with pytest.raises(HarnessTurnError, match="code 2: no output"):
        CodexHarnessConfig().transform_turn_response(
            make_ctx(FakeSandbox()), CodexStreamState(), 2, []
        )


def test_turn_response_output_json_only_with_output():
    state = CodexStreamState(final_text='{"file": "a", "content": "b"}')
    cfg = CodexHarnessConfig()
    with_out = cfg.transform_turn_response(
        make_ctx(FakeSandbox(), output=Answer), state, 0, []
    )
    assert with_out.output_json == state.final_text
    without = cfg.transform_turn_response(make_ctx(FakeSandbox()), state, 0, [])
    assert without.output_json is None and without.final_text == state.final_text


# --------------------------------------------------------------------------- handler


async def test_start_missing_binary():
    ctx = make_ctx(FakeSandbox(has_codex=False))
    with pytest.raises(HarnessInstallFailed, match="codex"):
        await make_handler().start(ctx)


async def test_start_rejects_managed_config_and_wrong_options():
    with pytest.raises(OptionsMismatch):
        await make_handler().start(
            make_ctx(
                FakeSandbox(), options=CodexOptions(config={"model_provider": "openai"})
            )
        )
    with pytest.raises(OptionsMismatch):
        await make_handler().start(make_ctx(FakeSandbox(), options=ClaudeCodeOptions()))


async def test_start_writes_skills_and_schema(tmp_path):
    skill = tmp_path / "my-skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: my-skill\n---\nbody")
    (skill / "scripts" / "run.sh").write_text("echo hi")
    sbx = FakeSandbox()
    await make_handler().start(make_ctx(sbx, skills=[str(skill)], output=Answer))
    assert sbx.files[f"{HOME}/skills/my-skill/SKILL.md"].startswith(b"---")
    assert sbx.files[f"{HOME}/skills/my-skill/scripts/run.sh"] == b"echo hi"
    schema = json.loads(sbx.files[f"{HOME}/output_schema.json"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["file", "content"]
    assert sbx.runs[0][:2] == ["sh", "-c"]
    assert sbx.runs[0][-2:] == [f"{HOME}/sessions", "codex/sessions"]


async def test_first_turn_then_resume_argv_env():
    sbx = FakeSandbox(
        outputs=[
            fixture_proc("turn1_bash.jsonl"),
            fixture_proc("turn2_resume_apply_patch.jsonl"),
        ]
    )
    handler = make_handler()
    ctx = make_ctx(
        sbx,
        instructions="Be terse.",
        options=CodexOptions(
            reasoning_effort="low",
            config={"sandbox_workspace_write.network_access": True},
        ),
    )
    await handler.start(ctx)
    events = await collect(handler, ctx, "create hello.txt")

    first = sbx.execs[0]
    argv, env = first["cmd"], first["env"]
    assert argv[:4] == ["codex", "exec", "--json", "--skip-git-repo-check"]
    assert argv[-1] == "-" and argv[argv.index("-C") + 1] == "/work"
    assert first["cwd"] == "/work"
    assert "model_provider=litellm" in config_values(argv)
    assert not any(TOKEN in a for a in argv)
    assert env["LITELLM_HARNESS_TOKEN"] == TOKEN
    assert env["CODEX_HOME"] == HOME
    assert sbx.processes[0].stdin.data == b"create hello.txt"
    assert sbx.processes[0].stdin.closed

    assert isinstance(events[-1], Text)
    assert ctx.final_text.startswith("Done")
    assert handler.native_session_id() == THREAD_ID

    await collect(handler, ctx, "edit it")
    argv2 = sbx.execs[1]["cmd"]
    assert argv2[:4] == ["codex", "exec", "resume", THREAD_ID]
    assert "--sandbox" not in argv2 and "-C" not in argv2
    assert 'sandbox_mode="workspace-write"' in config_values(argv2)
    assert ctx.final_text == "done"


async def test_resume_sets_thread_id():
    sbx = FakeSandbox(outputs=[fixture_proc("reasoning.jsonl")])
    handler = make_handler()
    ctx = make_ctx(sbx)
    await handler.start(ctx)
    await handler.resume(ctx, "thread-9")
    assert handler.native_session_id() == "thread-9"
    await collect(handler, ctx, "again")
    assert sbx.execs[0]["cmd"][:4] == ["codex", "exec", "resume", "thread-9"]


async def test_structured_output_sets_output_json():
    sbx = FakeSandbox(outputs=[fixture_proc("structured_output.jsonl")])
    handler = make_handler()
    ctx = make_ctx(sbx, output=Answer, permissions="read-only")
    await handler.start(ctx)
    await collect(handler, ctx, "read hello.txt")
    argv = sbx.execs[0]["cmd"]
    assert argv[argv.index("--output-schema") + 1] == f"{HOME}/output_schema.json"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert Answer.model_validate_json(ctx.output_json).file == "hello.txt"


async def test_turn_failed_raises():
    sbx = FakeSandbox(outputs=[fixture_proc("turn_failed.jsonl", exit_code=1)])
    handler = make_handler()
    ctx = make_ctx(sbx)
    await handler.start(ctx)
    with pytest.raises(HarnessTurnError, match="no healthy deployments"):
        await collect(handler, ctx, "hi")


async def test_nonzero_exit_raises_with_stderr_tail():
    sbx = FakeSandbox(
        outputs=[
            FakeProcess(b"", stderr=b"Error loading config.toml: bad\n", exit_code=1)
        ]
    )
    handler = make_handler()
    ctx = make_ctx(sbx)
    await handler.start(ctx)
    with pytest.raises(HarnessTurnError, match=r"code 1: Error loading config\.toml"):
        await collect(handler, ctx, "hi")


async def test_early_close_kills_process_and_stop_is_idempotent():
    sbx = FakeSandbox(outputs=[fixture_proc("turn1_bash.jsonl")])
    handler = make_handler()
    ctx = make_ctx(sbx)
    await handler.start(ctx)
    gen = handler.turn(ctx, "hi")
    await gen.__anext__()
    await gen.aclose()
    assert sbx.processes[0].killed
    await handler.stop(ctx)
    await handler.stop(ctx)


async def test_long_jsonl_line_is_parsed():
    text = "x" * 200_000
    line = json.dumps(
        {
            "type": "item.completed",
            "item": {"id": "a", "type": "agent_message", "text": text},
        }
    )
    sbx = FakeSandbox(outputs=[FakeProcess(line.encode() + b"\n")])
    handler = make_handler()
    ctx = make_ctx(sbx)
    await handler.start(ctx)
    events = await collect(handler, ctx, "hi")
    assert events == [Text(delta=text)]


def test_capabilities():
    cfg = CodexHarnessConfig()
    caps = cfg.capabilities
    assert cfg.harness is Harness.CODEX
    assert cfg.options_type is CodexOptions
    assert cfg.get_binary() == "codex"
    assert "@openai/codex" in cfg.get_install_hint()
    assert caps.structured_output and caps.skills and caps.resume
    assert not (
        caps.tool_approval or caps.tool_filtering or caps.custom_tools or caps.history
    )
    assert caps.permission_modes == frozenset({"read-only", "full"})
