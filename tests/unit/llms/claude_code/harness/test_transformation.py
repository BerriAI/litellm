"""Unit tests for the Claude Code harness config. No network, no real CLI.

Fixtures under fixtures/ are sanitized stream-json recorded from Claude Code
2.1.285 through a LiteLLM gateway.
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

from litellm.harness.context import SessionContext
from litellm.harness.errors import (
    HarnessError,
    HarnessInstallFailed,
    OptionsMismatch,
)
from litellm.harness.handlers.cli_handler import CLIHarnessHandler
from litellm.harness.options import ClaudeCodeOptions, CodexOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.types import (
    Compaction,
    Harness,
    Reasoning,
    Text,
    ToolCall,
    ToolResult,
)
from litellm.llms.base_llm.harness.transformation import (
    HarnessTurnError,
    HarnessTurnRequest,
)
from litellm.llms.base_llm.harness.utils import (
    decode_json_line,
    last_json_object,
    native_tool_names,
)
from litellm.llms.claude_code.harness.transformation import (
    MANAGED_CONFIG_KEYS,
    MANAGED_ENV_KEYS,
    NORMALIZED_TO_NATIVE,
    PERMISSION_MODES,
    ClaudeCodeHarnessConfig,
    ClaudeCodeStreamState,
    build_system_prompt,
    stringify_tool_output,
    turn_error_message,
)

FIXTURES = Path(__file__).parent / "fixtures"
SESSION_ID = "5ef64ff1-d2af-4c38-a7ca-17b4a9d07d34"
TOKEN = "per-session-token-abc"
PORT = 53211
PRIV = "/priv"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text().splitlines()


def parse_line(line: str, state: ClaudeCodeStreamState) -> list[Any]:
    decoded = decode_json_line(line)
    if decoded is None:
        return []
    return ClaudeCodeHarnessConfig().transform_stream_line(decoded, state)


def parse_fixture(name: str) -> tuple[list[Any], ClaudeCodeStreamState]:
    state = ClaudeCodeHarnessConfig().create_stream_state()
    events: list[Any] = []
    for line in fixture_lines(name):
        events.extend(parse_line(line, state))
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
        self.runs: list[list[str]] = []
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
        self.runs.append(cmd)
        return CompletedRun("", "", 0)

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


def pure_ctx(tmp_path: Path, **overrides: Any) -> SessionContext:
    return make_ctx(FakeSandbox(str(tmp_path), []), **overrides)


def make_handler() -> CLIHarnessHandler:
    return CLIHarnessHandler(ClaudeCodeHarnessConfig())


def request_for(
    ctx: SessionContext, native_session_id: str | None = None, prompt: str = "hi"
) -> HarnessTurnRequest:
    cfg = ClaudeCodeHarnessConfig()
    setup = cfg.transform_session_setup(ctx, PRIV)
    return cfg.transform_turn_request(ctx, setup, PRIV, prompt, native_session_id)


async def run_turn(handler: CLIHarnessHandler, ctx: SessionContext, prompt: str):
    return [event async for event in handler.turn(ctx, prompt)]


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
    assert ClaudeCodeHarnessConfig().get_native_session_id(state) == SESSION_ID
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
    state = ClaudeCodeStreamState()
    line = json.dumps(
        {
            "type": "system",
            "subtype": "compact_boundary",
            "compact_metadata": {"trigger": "auto", "pre_tokens": 1234},
        }
    )
    assert parse_line(line, state) == [
        Compaction(tokens_before=1234, tokens_after=None)
    ]
    assert parse_line("not json", state) == []
    assert parse_line("", state) == []
    assert parse_line("[1,2]", state) == []
    cfg = ClaudeCodeHarnessConfig()
    assert cfg.transform_stream_line({"type": "unknown"}, state) == []


def test_parse_skips_subagent_messages_and_maps_errors():
    cfg = ClaudeCodeHarnessConfig()
    state = ClaudeCodeStreamState()
    sub = {
        "type": "assistant",
        "parent_tool_use_id": "toolu_parent",
        "message": {"content": [{"type": "text", "text": "inner"}]},
    }
    assert cfg.transform_stream_line(sub, state) == []
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
    assert cfg.transform_stream_line(err, state) == [
        ToolResult(id="t1", output="boom", is_error=True)
    ]


def test_parse_thinking_and_mcp_tools():
    state = ClaudeCodeStreamState()
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
    events = ClaudeCodeHarnessConfig().transform_stream_line(msg, state)
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
    assert json.loads(last_json_object(text) or "") == {"b": {"c": 2}}
    assert last_json_object("no json here") is None


# ---------------------------------------------------------------------------
# Session setup / turn request (argv + env)
# ---------------------------------------------------------------------------


def test_native_disallowed_tools_mapping():
    natives = native_tool_names(["edit", "bash", "Task", "edit"], NORMALIZED_TO_NATIVE)
    assert natives == ["Edit", "MultiEdit", "Bash", "Task"]


@pytest.mark.parametrize(
    "permissions,native",
    [
        ("read-only", "plan"),
        ("edit", "acceptEdits"),
        ("full", "bypassPermissions"),
    ],
)
def test_turn_request_permission_modes(tmp_path, permissions, native):
    assert PERMISSION_MODES[permissions] == native
    argv = list(request_for(pure_ctx(tmp_path, permissions=permissions)).argv)
    assert argv[argv.index("--permission-mode") + 1] == native
    assert "--resume" not in argv
    assert argv[argv.index("--setting-sources") + 1] == "user"


def test_session_setup_and_turn_request_env_and_command(tmp_path):
    ctx = pure_ctx(
        tmp_path,
        instructions="Be terse.",
        disable_tools=["bash", "web_search"],
        max_turns=7,
        options=ClaudeCodeOptions(config={"cleanupPeriodDays": 1}, env={"X": "1"}),
    )
    cfg = ClaudeCodeHarnessConfig()
    setup = cfg.transform_session_setup(ctx, PRIV)
    assert setup.persisted_dirs == [("projects", "claude_code/projects")]
    assert setup.skills_dir == "skills"
    request = cfg.transform_turn_request(ctx, setup, PRIV, "do the thing", None)
    env, cmd = request.env, list(request.argv)
    assert request.stdin == "do the thing"
    assert env["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_BASE_URL"] == f"http://host.docker.internal:{PORT}"
    assert env["ANTHROPIC_MODEL"] == "claude-haiku-4-5-20251001"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-4-5-20251001"
    assert env["CLAUDE_CONFIG_DIR"] == PRIV
    assert env["DISABLE_TELEMETRY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["X"] == "1"
    assert not any(TOKEN in a for a in cmd)
    assert cmd[:7] == [
        "claude",
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
    assert json.loads(cmd[cmd.index("--settings") + 1]) == {"cleanupPeriodDays": 1}
    assert cmd[cmd.index("--disallowedTools") + 1] == "Bash,WebSearch"
    assert "--resume" not in cmd


def test_background_model_is_the_session_model(tmp_path):
    env = (
        ClaudeCodeHarnessConfig().transform_session_setup(pure_ctx(tmp_path), PRIV).env
    )
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-haiku-4-5-20251001"


def test_resume_argv(tmp_path):
    argv = list(request_for(pure_ctx(tmp_path), "prior-session").argv)
    assert argv[argv.index("--resume") + 1] == "prior-session"


def test_missing_endpoint_raises(tmp_path):
    with pytest.raises(HarnessError, match="endpoint"):
        ClaudeCodeHarnessConfig().transform_session_setup(
            pure_ctx(tmp_path, endpoint=None), PRIV
        )


@pytest.mark.parametrize("key", sorted(MANAGED_ENV_KEYS))
def test_options_env_cannot_override_managed_keys(tmp_path, key):
    ctx = pure_ctx(tmp_path, options=ClaudeCodeOptions(env={key: "sk-real"}))
    with pytest.raises(OptionsMismatch, match=key):
        ClaudeCodeHarnessConfig().validate_environment(ctx)


def test_wrong_options_type_rejected(tmp_path):
    with pytest.raises(OptionsMismatch):
        ClaudeCodeHarnessConfig().validate_environment(
            pure_ctx(tmp_path, options=CodexOptions())
        )


def test_structured_output_system_prompt(tmp_path):
    argv = list(
        request_for(pure_ctx(tmp_path, output=Answer, instructions="Base.")).argv
    )
    prompt = argv[argv.index("--append-system-prompt") + 1]
    assert prompt.startswith("Base.\n\n")
    assert json.dumps(Answer.model_json_schema()) in prompt
    assert build_system_prompt(None, None) is None


# ---------------------------------------------------------------------------
# Turn response
# ---------------------------------------------------------------------------


def test_turn_response_api_error_includes_stderr(tmp_path):
    _, state = parse_fixture("api_error.jsonl")
    with pytest.raises(HarnessTurnError) as info:
        ClaudeCodeHarnessConfig().transform_turn_response(
            pure_ctx(tmp_path), state, 1, ["[claude-code:unrecognized_model] bad"]
        )
    assert "no healthy deployments" in str(info.value)
    assert "unrecognized_model" in str(info.value)


def test_turn_response_max_turns(tmp_path):
    _, state = parse_fixture("max_turns.jsonl")
    with pytest.raises(HarnessTurnError, match="maximum number of turns"):
        ClaudeCodeHarnessConfig().transform_turn_response(
            pure_ctx(tmp_path), state, 1, []
        )


def test_turn_error_message_no_result():
    message = turn_error_message(ClaudeCodeStreamState(), 139, ["segfault", ""])
    assert message is not None
    assert "code 139: no result event" in message and "segfault" in message
    _, ok = parse_fixture("success_tools.jsonl")
    assert turn_error_message(ok, 0, []) is None


def test_turn_response_structured_output_and_fallback(tmp_path):
    cfg = ClaudeCodeHarnessConfig()
    ctx = pure_ctx(tmp_path, output=Answer)
    _, state = parse_fixture("structured_output.jsonl")
    response = cfg.transform_turn_response(ctx, state, 0, [])
    assert json.loads(response.output_json or "") == {"answer": 5, "word": "sum"}

    _, plain = parse_fixture("resume_turn.jsonl")
    response = cfg.transform_turn_response(ctx, plain, 0, [])
    assert response.final_text == "hello.txt"
    assert response.output_json is None  # "hello.txt" holds no JSON object

    text_json = ClaudeCodeStreamState(
        result_seen=True, result_text='answer: {"answer": 1, "word": "x"}'
    )
    response = cfg.transform_turn_response(ctx, text_json, 0, [])
    assert json.loads(response.output_json or "") == {"answer": 1, "word": "x"}

    no_output = cfg.transform_turn_response(pure_ctx(tmp_path), state, 0, [])
    assert no_output.output_json is None


# ---------------------------------------------------------------------------
# Through CLIHarnessHandler (start + turn)
# ---------------------------------------------------------------------------


async def test_start_and_turn_env_and_command(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("success_tools.jsonl", b"", 0)])
    ctx = make_ctx(sandbox, options=ClaudeCodeOptions(env={"X": "1"}))
    handler = make_handler()
    await handler.start(ctx)
    assert len(sandbox.runs) == 1
    assert sandbox.runs[0][:2] == ["sh", "-c"]
    assert sandbox.runs[0][-2:] == [
        f"{tmp_path / '_cfg'}/projects",
        "claude_code/projects",
    ]
    events = await run_turn(handler, ctx, "do the thing")

    call = sandbox.calls[0]
    env, cmd = call["env"], call["cmd"]
    assert env["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "_cfg")
    assert env["X"] == "1"
    assert not any(TOKEN in a for a in cmd)
    assert cmd[cmd.index("--setting-sources") + 1] == "user"

    proc = sandbox.procs[0]
    assert bytes(proc.stdin_data) == b"do the thing" and proc.stdin_closed
    assert any(isinstance(e, Text) for e in events)
    assert ctx.final_text.startswith("Done.")
    assert handler.native_session_id() == SESSION_ID


async def test_second_turn_resumes_session(tmp_path):
    sandbox = FakeSandbox(
        str(tmp_path),
        [("success_tools.jsonl", b"", 0), ("resume_turn.jsonl", b"", 0)],
    )
    ctx = make_ctx(sandbox)
    handler = make_handler()
    await handler.start(ctx)
    await run_turn(handler, ctx, "one")
    await run_turn(handler, ctx, "two")
    cmd = sandbox.calls[1]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == SESSION_ID
    assert ctx.final_text == "hello.txt"


async def test_resume_sets_native_session_id(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("resume_turn.jsonl", b"", 0)])
    ctx = make_ctx(sandbox)
    handler = make_handler()
    await handler.start(ctx)
    await handler.resume(ctx, "prior-session")
    assert handler.native_session_id() == "prior-session"
    await run_turn(handler, ctx, "again")
    cmd = sandbox.calls[0]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == "prior-session"


async def test_missing_binary_raises_install_failed(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [], binary=None)
    with pytest.raises(HarnessInstallFailed, match="claude"):
        await make_handler().start(make_ctx(sandbox))


async def test_start_missing_endpoint_raises(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [])
    with pytest.raises(HarnessError, match="endpoint"):
        await make_handler().start(make_ctx(sandbox, endpoint=None))


async def test_start_rejects_managed_env(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [])
    options = ClaudeCodeOptions(env={"ANTHROPIC_API_KEY": "sk-real"})
    with pytest.raises(OptionsMismatch, match="ANTHROPIC_API_KEY"):
        await make_handler().start(make_ctx(sandbox, options=options))
    assert sandbox.runs == [] and sandbox.written == {}


async def test_api_error_raises_turn_error_with_stderr(tmp_path):
    stderr = b"[claude-code:unrecognized_model] bad model\n"
    sandbox = FakeSandbox(str(tmp_path), [("api_error.jsonl", stderr, 1)])
    ctx = make_ctx(sandbox)
    handler = make_handler()
    await handler.start(ctx)
    with pytest.raises(HarnessTurnError) as info:
        await run_turn(handler, ctx, "hi")
    assert "no healthy deployments" in str(info.value)
    assert "unrecognized_model" in str(info.value)


async def test_max_turns_raises_turn_error(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("max_turns.jsonl", b"", 1)])
    ctx = make_ctx(sandbox)
    handler = make_handler()
    await handler.start(ctx)
    with pytest.raises(HarnessTurnError, match="maximum number of turns"):
        await run_turn(handler, ctx, "hi")


async def test_nonzero_exit_without_result_raises(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("", b"segfault\n", 139)])
    ctx = make_ctx(sandbox)
    handler = make_handler()
    await handler.start(ctx)
    with pytest.raises(HarnessTurnError, match=r"code 139.*no result event") as info:
        await run_turn(handler, ctx, "hi")
    assert "segfault" in str(info.value)


async def test_structured_output_prompt_and_json(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("structured_output.jsonl", b"", 0)])
    ctx = make_ctx(sandbox, output=Answer, instructions="Base.")
    handler = make_handler()
    await handler.start(ctx)
    await run_turn(handler, ctx, "2+3?")
    cmd = sandbox.calls[0]["cmd"]
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert prompt.startswith("Base.\n\n")
    assert json.loads(ctx.output_json or "") == {"answer": 5, "word": "sum"}


async def test_structured_output_falls_back_to_final_text(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("resume_turn.jsonl", b"", 0)])
    ctx = make_ctx(sandbox, output=Answer)
    handler = make_handler()
    await handler.start(ctx)
    await run_turn(handler, ctx, "hi")
    assert ctx.output_json is None  # "hello.txt" holds no JSON object


async def test_skills_copied_into_private_config(tmp_path):
    skill = tmp_path / "skills_src" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nSay DEMO.\n")
    (skill / "scripts" / "run.sh").write_text("echo hi\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    sandbox = FakeSandbox(str(workdir), [], tempdir="/cfg")
    await make_handler().start(make_ctx(sandbox, skills=[str(skill)]))
    assert sandbox.written == {
        "/cfg/skills/demo/SKILL.md": b"---\nname: demo\n---\nSay DEMO.\n",
        "/cfg/skills/demo/scripts/run.sh": b"echo hi\n",
    }


async def test_skill_without_manifest_rejected(tmp_path):
    skill = tmp_path / "bad"
    skill.mkdir()
    sandbox = FakeSandbox(str(tmp_path), [])
    with pytest.raises(ValueError, match=r"SKILL\.md"):
        await make_handler().start(make_ctx(sandbox, skills=[str(skill)]))


async def test_stop_kills_live_process(tmp_path):
    sandbox = FakeSandbox(str(tmp_path), [("success_tools.jsonl", b"", 0)])
    ctx = make_ctx(sandbox)
    handler = make_handler()
    await handler.start(ctx)
    stream = handler.turn(ctx, "hi")
    await stream.__anext__()
    proc = sandbox.procs[0]
    await handler.stop(ctx)
    assert proc.killed
    await stream.aclose()
    await handler.stop(ctx)  # safe twice


def test_capabilities_match_spec():
    cfg = ClaudeCodeHarnessConfig()
    caps = cfg.capabilities
    assert cfg.harness is Harness.CLAUDE_CODE
    assert cfg.options_type is ClaudeCodeOptions
    assert cfg.get_binary() == "claude"
    assert "@anthropic-ai/claude-code" in cfg.get_install_hint()
    assert caps.structured_output and caps.tool_filtering and caps.skills
    assert caps.resume
    assert not (caps.tool_approval or caps.custom_tools or caps.history)
    assert caps.permission_modes == frozenset({"read-only", "edit", "full"})


@pytest.mark.parametrize("key", sorted(MANAGED_CONFIG_KEYS))
def test_options_config_cannot_set_managed_keys(tmp_path, key):
    ctx = pure_ctx(tmp_path, options=ClaudeCodeOptions(config={key: "x"}))
    with pytest.raises(OptionsMismatch, match=key):
        ClaudeCodeHarnessConfig().validate_environment(ctx)


def test_no_settings_flag_without_config(tmp_path):
    assert "--settings" not in list(request_for(pure_ctx(tmp_path), None).argv)
