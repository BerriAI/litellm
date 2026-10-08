"""Unit tests for the pi harness config. No network, no real CLI.

Fixtures under fixtures/ are `pi --mode json` output recorded from
@earendil-works/pi-coding-agent 1.1.0 against a scripted OpenAI-compatible endpoint,
with local paths replaced.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel

from litellm.harness.context import SessionContext
from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessError,
    HarnessInstallFailed,
    OptionsMismatch,
)
from litellm.harness.handlers.cli_handler import PERSIST_DIR_SCRIPT, CLIHarnessHandler
from litellm.harness.options import OpenCodeOptions, PiOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.types import Harness, Reasoning, Text, ToolCall, ToolResult
from litellm.llms.base_llm.harness.transformation import HarnessTurnError
from litellm.llms.pi.harness.transformation import (
    PI_ISOLATION_ENV,
    PI_TOKEN_ENV,
    PiHarnessConfig,
    PiStreamState,
    tool_args,
)
from litellm.utils import ProviderConfigManager

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "tok-secret-123"
SESSION = "01a11ca4-d40a-77eb-bbca-100787a0d1a2"
PRIVATE = "/tmp/pi-1"
MODEL = "claude-haiku-4-5-20251001"
CONFIG = PiHarnessConfig()


def load_fixture(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line]


def parse_all(name: str):
    state = CONFIG.create_stream_state()
    events = [event for obj in load_fixture(name) for event in CONFIG.transform_stream_line(obj, state)]
    return events, state


def parse_lines(lines: list[dict]):
    state = CONFIG.create_stream_state()
    events = [event for obj in lines for event in CONFIG.transform_stream_line(obj, state)]
    return events, state


def assistant_end(text: str, stop_reason: str, error: str | None = None) -> dict:
    message = {"role": "assistant", "content": [{"type": "text", "text": text}], "stopReason": stop_reason}
    return {"type": "message_end", "message": {**message, **({"errorMessage": error} if error else {})}}


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
    outputs: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    execs: list = field(default_factory=list)
    runs: list = field(default_factory=list)

    async def exec(self, cmd, *, env=None, cwd=None):
        self.execs.append({"cmd": cmd, "env": dict(env or {}), "cwd": cwd})
        return self.outputs.pop(0)

    async def run(self, cmd, *, env=None, cwd=None, timeout=None):
        self.runs.append(cmd)
        return CompletedRun("", "", 0)

    async def write(self, path, data):
        self.files[path] = data

    def host_url(self, port):
        return f"http://host.docker.internal:{port}"

    async def which(self, binary):
        return f"/usr/bin/{binary}" if self.has_binary else None

    async def tempdir(self):
        return PRIVATE


@dataclass
class FakeEndpoint:
    port: int = 4555
    token: str = TOKEN
    model: str | None = None


class Answer(BaseModel):
    city: str
    country: str


def make_ctx(sandbox=None, **kwargs) -> SessionContext:
    return SessionContext(
        harness=Harness.PI,
        sandbox=sandbox or FakeSandbox(),
        session_id="s1",
        model=kwargs.pop("model", MODEL),
        endpoint=kwargs.pop("endpoint", FakeEndpoint()),
        **kwargs,
    )


def fixture_proc(name: str, **kwargs) -> FakeProcess:
    return FakeProcess((FIXTURES / name).read_bytes(), **kwargs)


async def started(sandbox=None, **kwargs):
    sandbox = sandbox or FakeSandbox()
    handler = CLIHarnessHandler(PiHarnessConfig())
    ctx = make_ctx(sandbox, **kwargs)
    await handler.start(ctx)
    return handler, ctx, sandbox


async def collect(handler, ctx, prompt):
    return [e async for e in handler.turn(ctx, prompt)]


def flag(argv, name):
    argv = list(argv)
    return argv[argv.index(name) + 1]


# --------------------------------------------------------------------------- parsing


def test_parse_write_read_turn():
    events, state = parse_all("turn1_write_read.jsonl")
    assert CONFIG.get_native_session_id(state) == SESSION
    assert [type(e) for e in events[:5]] == [Reasoning, ToolCall, ToolResult, ToolCall, ToolResult]
    assert all(isinstance(e, Text) for e in events[5:])
    assert events[0] == Reasoning(delta="I should write the file first.")
    write, write_result, read, read_result = events[1:5]
    assert write == ToolCall(
        id="call_w1",
        name="write",
        native_name="write",
        input={"path": "hello.txt", "content": "hi"},
        builtin=True,
    )
    assert write_result == ToolResult(id="call_w1", output="Successfully wrote to hello.txt", is_error=False)
    assert read.name == "read" and read.id == "call_r1"
    assert read_result == ToolResult(id="call_r1", output="hi", is_error=False)
    assert "".join(e.delta for e in events[5:]) == state.final_text == "Done! hello.txt contains: hi"
    assert state.stop_reason == "stop" and state.error is None


def test_parse_resumed_turn_keeps_session_id():
    events, state = parse_all("turn2_resume.jsonl")
    assert state.session_id == SESSION
    assert state.final_text == "You asked me to create hello.txt."
    assert all(isinstance(e, Text) for e in events)


def test_parse_tool_unavailable_in_read_only_is_error_result():
    events, state = parse_all("readonly_denied_bash.jsonl")
    call = next(e for e in events if isinstance(e, ToolCall))
    result = next(e for e in events if isinstance(e, ToolResult))
    assert call.name == "bash" and call.input == {"command": "echo x > blocked.txt"}
    assert result == ToolResult(id="call_b1", output="Tool bash not found", is_error=True)
    assert state.final_text.startswith("FAILED")


def test_parse_api_error_records_last_assistant_failure():
    events, state = parse_all("api_error.jsonl")
    assert events == []
    assert state.stop_reason == "error"
    assert "no healthy deployments" in state.error


def test_successful_retry_after_error_is_not_a_failure():
    _, state = parse_lines([assistant_end("", "error", "529 overloaded"), assistant_end("recovered", "stop")])
    response = CONFIG.transform_turn_response(make_ctx(), state, 0, [])
    assert response.final_text == "recovered"


def test_user_and_system_message_end_do_not_replace_final_text():
    _, state = parse_lines(
        [
            assistant_end("answer", "stop"),
            {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "q"}]}},
            {"type": "message_end", "message": {"role": "system", "content": ""}},
        ]
    )
    assert state.final_text == "answer"


def test_final_text_joins_text_blocks_and_skips_thinking_and_tool_calls():
    _, state = parse_lines(
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "a"},
                        {"type": "toolCall", "id": "x", "name": "read", "arguments": {}},
                        {"type": "text", "text": "b"},
                    ],
                },
            }
        ]
    )
    assert state.final_text == "ab"


def test_aborted_run_is_a_failure():
    _, state = parse_lines([assistant_end("partial", "stop"), {"type": "agent_settled", "aborted": True}])
    with pytest.raises(HarnessTurnError, match="aborted"):
        CONFIG.transform_turn_response(make_ctx(), state, 0, [])
    _, settled = parse_lines([assistant_end("done", "stop"), {"type": "agent_settled", "aborted": False}])
    assert CONFIG.transform_turn_response(make_ctx(), settled, 0, []).final_text == "done"


def test_message_update_ignores_non_delta_events():
    events, _ = parse_lines(
        [
            {"type": "message_update", "assistantMessageEvent": {"type": "text_start", "contentIndex": 0}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_end", "content": "full"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "toolcall_delta", "delta": "{"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": ""}},
            {"type": "message_update"},
        ]
    )
    assert events == []


@pytest.mark.parametrize(
    "native,normalized,builtin",
    [
        ("find", "glob", True),
        ("powershell", "bash", True),
        ("ls", "ls", True),
        ("mcp__gh__search", "mcp__gh__search", False),
    ],
)
def test_tool_name_mapping(native, normalized, builtin):
    events, _ = parse_lines(
        [
            {"type": "tool_execution_start", "toolCallId": "c", "toolName": native, "args": {"q": 1}},
            {"type": "tool_execution_end", "toolCallId": "c", "toolName": native, "result": "plain", "isError": False},
        ]
    )
    assert events == [
        ToolCall(id="c", name=normalized, native_name=native, input={"q": 1}, builtin=builtin),
        ToolResult(id="c", output="plain", is_error=False),
    ]


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "mode,tools",
    [
        ("read-only", "read,grep,find,ls"),
        ("edit", "read,edit,write,grep,find,ls"),
        ("full", "read,bash,edit,write,grep,find,ls"),
    ],
)
def test_permission_modes_map_to_tool_allowlist(mode, tools):
    assert tool_args(mode, ()) == ("--tools", tools)


def test_disable_tools_map_to_native_excludes():
    args = tool_args("full", ["bash", "glob", "mcp__x"])
    assert flag(args, "--exclude-tools") == "bash,powershell,find,mcp__x"


def test_ask_mode_unsupported():
    with pytest.raises(CapabilityUnsupported, match="permissions='ask'"):
        tool_args("ask", ())
    with pytest.raises(CapabilityUnsupported):
        CONFIG.validate_environment(make_ctx(permissions="ask"))


def test_capabilities_match_supported_modes():
    assert CONFIG.capabilities.permission_modes == {"read-only", "edit", "full"}
    assert CONFIG.capabilities.resume and CONFIG.capabilities.tool_filtering


def test_wrong_options_rejected():
    with pytest.raises(OptionsMismatch, match="PiOptions"):
        CONFIG.validate_environment(make_ctx(options=OpenCodeOptions()))


def test_harness_config_dispatch():
    assert isinstance(ProviderConfigManager.get_provider_harness_config(Harness.PI), PiHarnessConfig)


def test_session_setup_models_json_points_at_endpoint_without_token():
    setup = CONFIG.transform_session_setup(make_ctx(), PRIVATE)
    models = json.loads(setup.files["agent/models.json"])
    assert models == {
        "providers": {
            "litellm": {
                "baseUrl": "http://host.docker.internal:4555/v1",
                "api": "openai-completions",
                "apiKey": f"${PI_TOKEN_ENV}",
                "models": [{"id": MODEL}],
            }
        }
    }
    assert all(TOKEN.encode() not in data for data in setup.files.values())
    assert setup.env[PI_TOKEN_ENV] == TOKEN


def test_session_setup_env_isolates_pi_and_options_cannot_override_managed_keys():
    options = PiOptions(env={"FOO": "1", PI_TOKEN_ENV: "evil", "PI_CODING_AGENT_DIR": "/home/u/.pi/agent"})
    setup = CONFIG.transform_session_setup(make_ctx(options=options), PRIVATE)
    assert setup.env["PI_CODING_AGENT_DIR"] == f"{PRIVATE}/agent"
    assert setup.env[PI_TOKEN_ENV] == TOKEN
    assert setup.env["FOO"] == "1"
    for key, value in PI_ISOLATION_ENV.items():
        assert setup.env[key] == value
    assert list(setup.persisted_dirs) == [("sessions", "pi/sessions")]


def test_session_setup_errors():
    with pytest.raises(HarnessError):
        CONFIG.transform_session_setup(make_ctx(endpoint=None), PRIVATE)
    with pytest.raises(ValueError, match="needs model="):
        CONFIG.transform_session_setup(make_ctx(model=None, endpoint=FakeEndpoint(model=None)), PRIVATE)


def test_first_turn_argv():
    ctx = make_ctx()
    request = CONFIG.transform_turn_request(ctx, CONFIG.transform_session_setup(ctx, PRIVATE), PRIVATE, "hi", None)
    assert list(request.argv) == [
        "pi",
        "--mode",
        "json",
        "--no-approve",
        "--no-skills",
        "--provider",
        "litellm",
        "--model",
        MODEL,
        "--session-dir",
        f"{PRIVATE}/sessions",
        "--tools",
        "read,bash,edit,write,grep,find,ls",
    ]
    assert request.stdin == "hi" and request.cwd == "/work"


def test_turn_argv_optional_flags():
    ctx = make_ctx(
        options=PiOptions(thinking="high"),
        skills=["/s/greeter"],
        instructions="Be terse.",
        output=Answer,
    )
    setup = CONFIG.transform_session_setup(ctx, PRIVATE)
    argv = CONFIG.transform_turn_request(ctx, setup, PRIVATE, "hi", SESSION).argv
    assert flag(argv, "--session") == SESSION
    assert flag(argv, "--thinking") == "high"
    assert flag(argv, "--skill") == f"{PRIVATE}/skills"
    assert flag(argv, "--append-system-prompt") == f"{PRIVATE}/instructions.md"
    instructions = setup.files["instructions.md"].decode()
    assert instructions.startswith("Be terse.") and '"city"' in instructions


def test_turn_response_structured_output_and_exit_code():
    _, state = parse_all("structured_output.jsonl")
    ok = CONFIG.transform_turn_response(make_ctx(output=Answer), state, 0, [])
    assert json.loads(ok.output_json) == {"city": "Paris", "country": "France"}
    assert CONFIG.transform_turn_response(make_ctx(), state, 0, []).output_json is None
    with pytest.raises(HarnessTurnError, match="code 3: no output"):
        CONFIG.transform_turn_response(make_ctx(), PiStreamState(), 3, [])


# --------------------------------------------------------------------------- handler


async def test_session_start_writes_config_and_persists_sessions():
    _, _, sandbox = await started()
    assert (
        json.loads(sandbox.files[f"{PRIVATE}/agent/models.json"])["providers"]["litellm"]["apiKey"]
        == "$LITELLM_HARNESS_TOKEN"
    )
    assert sandbox.runs == [["sh", "-c", PERSIST_DIR_SCRIPT, "sh", f"{PRIVATE}/sessions", "pi/sessions"]]


async def test_turns_continue_native_session_and_send_prompt_on_stdin():
    handler, ctx, sandbox = await started()
    first_proc = fixture_proc("turn1_write_read.jsonl")
    sandbox.outputs.append(first_proc)
    events = await collect(handler, ctx, "create hello.txt")
    first = sandbox.execs[0]
    assert "--session" not in first["cmd"]
    assert TOKEN not in json.dumps(first["cmd"])
    assert first["env"][PI_TOKEN_ENV] == TOKEN
    assert first_proc.stdin.data == b"create hello.txt" and first_proc.stdin.closed
    assert any(isinstance(e, ToolCall) for e in events)
    assert ctx.final_text == "Done! hello.txt contains: hi"
    assert handler.native_session_id() == SESSION

    sandbox.outputs.append(fixture_proc("turn2_resume.jsonl"))
    await collect(handler, ctx, "what did you create?")
    assert flag(sandbox.execs[1]["cmd"], "--session") == SESSION
    assert ctx.final_text == "You asked me to create hello.txt."


async def test_resume_uses_stored_session():
    handler, ctx, sandbox = await started()
    await handler.resume(ctx, "prev-session")
    sandbox.outputs.append(fixture_proc("turn2_resume.jsonl"))
    await collect(handler, ctx, "hi")
    assert flag(sandbox.execs[0]["cmd"], "--session") == "prev-session"


async def test_provider_error_raises_even_though_pi_exits_zero():
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(fixture_proc("api_error.jsonl", exit_code=0))
    with pytest.raises(HarnessTurnError, match="no healthy deployments"):
        await collect(handler, ctx, "x")


async def test_nonzero_exit_raises_with_stderr_tail():
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(FakeProcess(b"", stderr=b"line1\nfatal: bad flag\n", exit_code=2))
    with pytest.raises(HarnessTurnError, match="code 2: line1\nfatal: bad flag"):
        await collect(handler, ctx, "x")


async def test_structured_output_reaches_context():
    handler, ctx, sandbox = await started(output=Answer)
    proc = fixture_proc("structured_output.jsonl")
    sandbox.outputs.append(proc)
    await collect(handler, ctx, "capital of France?")
    assert '"city"' in proc.stdin.data.decode()
    assert json.loads(ctx.output_json) == {"city": "Paris", "country": "France"}


async def test_skills_copied_where_skill_flag_points(tmp_path):
    skill = tmp_path / "greeter"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: greeter\ndescription: d\n---\nbody")
    handler, ctx, sandbox = await started(skills=[str(skill)])
    assert sandbox.files[f"{PRIVATE}/skills/greeter/SKILL.md"].startswith(b"---")
    sandbox.outputs.append(fixture_proc("turn2_resume.jsonl"))
    await collect(handler, ctx, "x")
    assert flag(sandbox.execs[0]["cmd"], "--skill") == f"{PRIVATE}/skills"


async def test_missing_binary_names_install_command():
    with pytest.raises(HarnessInstallFailed, match="@earendil-works/pi-coding-agent"):
        await started(FakeSandbox(has_binary=False))
