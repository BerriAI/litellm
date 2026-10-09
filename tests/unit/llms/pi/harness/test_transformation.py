"""Unit tests for the pi harness config. No network, no real CLI.

Fixtures under fixtures/ are sanitized `pi --mode json` output recorded from
@earendil-works/pi-coding-agent 1.1.0 against a scripted OpenAI-compatible endpoint.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
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
from litellm.harness.types import Event, Harness, Reasoning, Text, ToolCall, ToolResult
from litellm.llms.base_llm.harness.transformation import (
    HarnessSessionSetup,
    HarnessTurnError,
)
from litellm.llms.pi.harness.transformation import (
    INSTRUCTIONS_FILENAME,
    MANAGED_CONFIG_KEYS,
    MANAGED_ENV_KEYS,
    MCP_FILENAME,
    MODELS_FILENAME,
    PI_CONFIG_DIR_ENV,
    PI_ISOLATION_ENV,
    PI_TOKEN_ENV,
    SETTINGS_FILENAME,
    PiHarnessConfig,
    PiStreamState,
    build_models_json,
    config_files,
    mcp_tool_entries,
    tool_args,
    validate_user_config,
)
from litellm.utils import ProviderConfigManager

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "tok-secret-123"
SESSION = "01a11cb0-87cf-752e-a3e3-365c7a169871"
PRIVATE = "/tmp/pi-1"
MODEL = "claude-haiku-4-5-20251001"
FULL_TOOLS = "read,bash,edit,write,grep,find,ls"
MOYAI = {"command": "/path/to/mcp-server", "args": [], "env": {}, "exposure": "direct"}
CONFIG = PiHarnessConfig()


def load_fixture(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line]


def parse(obj: Mapping[str, object], state: PiStreamState) -> Sequence[Event]:
    return CONFIG.transform_stream_line(obj, state)


def parse_all(name: str, state: PiStreamState | None = None) -> tuple[list[Event], PiStreamState]:
    state = state or CONFIG.create_stream_state()
    events = []
    for obj in load_fixture(name):
        events.extend(parse(obj, state))
    return events, state


def assistant_end(text: str, stop_reason: str, error: str | None = None) -> dict[str, object]:
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stopReason": stop_reason,
    }
    if error:
        message["errorMessage"] = error
    return {"type": "message_end", "message": message}


class FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", exit_code: int = 0) -> None:
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
    persist_ok: bool = True
    outputs: list[FakeProcess] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)
    execs: list[dict[str, object]] = field(default_factory=list)
    runs: list[Sequence[str]] = field(default_factory=list)
    tempdirs: int = 0

    async def exec(
        self, cmd: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None
    ) -> FakeProcess:
        self.execs.append({"cmd": cmd, "env": dict(env or {}), "cwd": cwd})
        return self.outputs.pop(0)

    async def run(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CompletedRun:
        self.runs.append(cmd)
        if self.persist_ok:
            return CompletedRun("", "", 0)
        return CompletedRun("", "read-only fs", 1)

    async def read(self, path: str) -> bytes:
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def host_url(self, port: int) -> str:
        return f"http://host.docker.internal:{port}"

    async def which(self, binary: str) -> str | None:
        return f"/usr/bin/{binary}" if self.has_binary else None

    async def tempdir(self) -> str:
        self.tempdirs += 1
        return f"/tmp/pi-{self.tempdirs}"

    async def snapshot(self) -> dict[str, str]:
        return {}

    async def close(self) -> None:
        return None


@dataclass
class FakeEndpoint:
    port: int = 4555
    token: str = TOKEN
    model: str | None = None


class Answer(BaseModel):
    city: str
    country: str


def make_ctx(sandbox: FakeSandbox | None = None, **kwargs: object) -> SessionContext:
    return SessionContext(
        harness=Harness.PI,
        sandbox=sandbox or FakeSandbox(),
        session_id="s1",
        model=kwargs.pop("model", MODEL),
        endpoint=kwargs.pop("endpoint", FakeEndpoint()),
        **kwargs,
    )


def setup_for(ctx: SessionContext) -> HarnessSessionSetup:
    return CONFIG.transform_session_setup(ctx, PRIVATE)


def setup_models(setup: HarnessSessionSetup) -> dict[str, object]:
    return json.loads(setup.files[MODELS_FILENAME])


def fixture_proc(name: str, exit_code: int = 0) -> FakeProcess:
    return FakeProcess((FIXTURES / name).read_bytes(), exit_code=exit_code)


async def collect(handler: CLIHarnessHandler, ctx: SessionContext, prompt: str) -> list[Event]:
    return [e async for e in handler.turn(ctx, prompt)]


async def started(
    sandbox: FakeSandbox | None = None, **kwargs: object
) -> tuple[CLIHarnessHandler, SessionContext, FakeSandbox]:
    sandbox = sandbox or FakeSandbox()
    handler = CLIHarnessHandler(PiHarnessConfig())
    ctx = make_ctx(sandbox, **kwargs)
    await handler.start(ctx)
    return handler, ctx, sandbox


def flag(argv: Sequence[str], name: str) -> str:
    args = list(argv)
    return args[args.index(name) + 1]


def written_models(sandbox: FakeSandbox, private: str = PRIVATE) -> dict[str, object]:
    return json.loads(sandbox.files[f"{private}/{MODELS_FILENAME}"])


def test_parse_write_read_turn():
    events, state = parse_all("turn1_write_read.jsonl")
    assert CONFIG.get_native_session_id(state) == SESSION
    assert [type(e) for e in events] == [
        Reasoning,
        ToolCall,
        ToolResult,
        ToolCall,
        ToolResult,
        Text,
        Text,
        Text,
    ]
    assert events[0] == Reasoning(delta="I should write the file first.")
    write, write_result = events[1], events[2]
    assert write.name == "write" and write.native_name == "write"
    assert write.builtin is True
    assert write.input == {"path": "hello.txt", "content": "hi"}
    assert write_result.id == write.id == "call_w1"
    assert write_result.output == "Successfully wrote to hello.txt"
    assert write_result.is_error is False
    read, read_result = events[3], events[4]
    assert read.name == "read" and read_result.output == "hi"
    assert "".join(e.delta for e in events[5:]) == state.final_text
    assert state.final_text == "Done! hello.txt contains: hi"
    assert state.stop_reason == "stop" and state.error is None


def test_parse_continued_session():
    events, state = parse_all("turn2_resume.jsonl")
    assert state.session_id == SESSION
    assert all(isinstance(e, Text) for e in events)
    assert state.final_text == "You asked me to create hello.txt."


def test_parse_denied_tool_is_error_result():
    events, state = parse_all("readonly_denied_bash.jsonl")
    call = next(e for e in events if isinstance(e, ToolCall))
    result = next(e for e in events if isinstance(e, ToolResult))
    assert call.native_name == "bash" and call.input == {"command": "echo x > blocked.txt"}
    assert result == ToolResult(id="call_b1", output="Tool bash not found", is_error=True)
    assert state.final_text.startswith("FAILED")


def test_parse_mcp_tool_turn():
    events, state = parse_all("mcp_tool.jsonl")
    call = next(e for e in events if isinstance(e, ToolCall))
    result = next(e for e in events if isinstance(e, ToolResult))
    assert call == ToolCall(
        id="call_m1",
        name="mcp__moyai__echo",
        native_name="mcp__moyai__echo",
        input={"text": "hello"},
        builtin=False,
    )
    assert result == ToolResult(id="call_m1", output="moyai says: hello", is_error=False)
    assert state.final_text == "The MCP tool replied."


def test_parse_api_error_records_error():
    events, state = parse_all("api_error.jsonl")
    assert events == []
    assert state.stop_reason == "error"
    assert "no healthy deployments" in state.error


def test_parse_reasoning_and_tool_error_and_name_mapping():
    state = PiStreamState()
    reasoning = parse(
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "thinking hard"},
        },
        state,
    )
    assert reasoning == [Reasoning(delta="thinking hard")]
    failed = parse(
        {
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "boom"}]},
            "isError": True,
        },
        state,
    )
    assert failed == [ToolResult(id="c1", output="boom", is_error=True)]
    for native, normalized in [
        ("find", "glob"),
        ("powershell", "bash"),
        ("grep", "grep"),
        ("ls", "ls"),
        ("edit", "edit"),
    ]:
        call = parse(
            {"type": "tool_execution_start", "toolCallId": "x", "toolName": native, "args": {}},
            state,
        )[0]
        assert call.name == normalized and call.builtin is True
    mcp = parse(
        {"type": "tool_execution_start", "toolCallId": "m", "toolName": "mcp__gh__search", "args": {"q": 1}},
        state,
    )
    assert mcp[0].builtin is False and mcp[0].input == {"q": 1}
    plain = parse(
        {"type": "tool_execution_end", "toolCallId": "m", "toolName": "mcp__gh__search", "result": "raw"},
        state,
    )
    assert plain == [ToolResult(id="m", output="raw", is_error=False)]
    no_result = parse({"type": "tool_execution_end", "toolCallId": "e", "toolName": "bash"}, state)
    assert no_result == [ToolResult(id="e", output="", is_error=False)]


def test_final_text_is_last_assistant_message():
    state = PiStreamState()
    parse(assistant_end("working", "toolUse"), state)
    parse(assistant_end("answer", "stop"), state)
    parse(
        {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "q"}]}},
        state,
    )
    parse({"type": "message_end", "message": {"role": "system", "content": ""}}, state)
    assert state.final_text == "answer"
    parse(
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
        },
        state,
    )
    assert state.final_text == "ab"


def test_message_update_ignores_non_delta_events():
    state = PiStreamState()
    for update in [
        {"type": "text_start", "contentIndex": 0},
        {"type": "text_end", "content": "full"},
        {"type": "toolcall_delta", "delta": "{"},
        {"type": "text_delta", "delta": ""},
    ]:
        assert parse({"type": "message_update", "assistantMessageEvent": update}, state) == []
    assert parse({"type": "message_update"}, state) == []


def test_retried_error_then_success_is_not_an_error():
    state = PiStreamState()
    parse(assistant_end("", "error", "529 overloaded"), state)
    parse(assistant_end("recovered", "stop"), state)
    assert state.stop_reason == "stop" and state.error is None
    assert CONFIG.transform_turn_response(make_ctx(), state, 0, []).final_text == "recovered"


def test_aborted_run_is_an_error():
    state = PiStreamState()
    parse(assistant_end("partial", "stop"), state)
    parse({"type": "agent_settled", "aborted": False}, state)
    assert state.stop_reason == "stop"
    parse({"type": "agent_settled", "aborted": True}, state)
    assert state.stop_reason == "aborted"


def test_permission_mapping():
    assert tool_args("full", ()) == ("--tools", FULL_TOOLS)
    assert tool_args("read-only", ()) == ("--tools", "read,grep,find,ls")
    assert tool_args("edit", ()) == ("--tools", "read,edit,write,grep,find,ls")
    with pytest.raises(CapabilityUnsupported):
        tool_args("ask", ())


def test_disable_tools_map_to_native_excludes():
    args = tool_args("full", ["bash", "glob", "write", "mcp__x"])
    assert flag(args, "--tools") == FULL_TOOLS
    assert flag(args, "--exclude-tools") == "bash,powershell,find,write,mcp__x"


def test_mcp_servers_keep_tools_reachable():
    assert mcp_tool_entries({}) == ()
    assert mcp_tool_entries({"mcpServers": {}}) == ()
    assert mcp_tool_entries({"mcpServers": {"moyai": MOYAI}}) == ("mcp__*",)
    assert mcp_tool_entries({"mcpServers": {"s": {"command": "x"}}}) == ("mcp__*", "codemode")
    assert mcp_tool_entries({"mcpServers": {"s": {"url": "u", "exposure": "deferred"}}}) == (
        "mcp__*",
        "tool_search",
    )
    assert mcp_tool_entries(
        {"mcpServers": {"s": {"command": "x", "exposure": "hidden", "toolExposure": {"a": "direct", "b": "deferred"}}}}
    ) == ("mcp__*", "tool_search")
    args = tool_args("read-only", ["mcp__moyai__drop"], mcp_tool_entries({"mcpServers": {"moyai": MOYAI}}))
    assert flag(args, "--tools") == "read,grep,find,ls,mcp__*"
    assert flag(args, "--exclude-tools") == "mcp__moyai__drop"


def test_config_files_split_settings_and_mcp():
    files = dict(config_files({"mcpServers": {"moyai": MOYAI}, "compaction": {"enabled": False}}))
    assert json.loads(files[MCP_FILENAME]) == {"mcpServers": {"moyai": MOYAI}}
    assert json.loads(files[SETTINGS_FILENAME]) == {"compaction": {"enabled": False}}
    assert dict(config_files({"mcpServers": {"moyai": MOYAI}})).keys() == {MCP_FILENAME}
    assert dict(config_files({"theme": "dark"})).keys() == {SETTINGS_FILENAME}
    assert config_files({}) == ()


def test_config_files_reject_values_that_are_not_json():
    with pytest.raises(TypeError, match=r"set in PiOptions\.config is not JSON serializable"):
        config_files({"theme": {"dark", "light"}})


@pytest.mark.parametrize(
    "config",
    [
        *({key: "x"} for key in sorted(MANAGED_CONFIG_KEYS)),
        {"mcpServers": "not-a-mapping"},
        {"mcpServers": {"moyai": "not-a-mapping"}},
    ],
)
def test_managed_keys_rejected(config):
    with pytest.raises(OptionsMismatch):
        validate_user_config(config)


def test_options_config_cannot_load_code_or_reroute_model():
    for key in ("extensions", "packages", "defaultProjectTrust", "defaultProvider", "defaultModel", "httpProxy"):
        assert key in MANAGED_CONFIG_KEYS
    with pytest.raises(OptionsMismatch, match="extensions"):
        validate_user_config({"extensions": ["./evil.ts"]})


def test_build_models_json():
    models = json.loads(build_models_json("m1", "http://h:1/v1"))
    assert models == {
        "providers": {
            "litellm": {
                "baseUrl": "http://h:1/v1",
                "api": "openai-completions",
                "apiKey": f"${PI_TOKEN_ENV}",
                "models": [{"id": "m1"}],
            }
        }
    }


def test_config_metadata():
    assert CONFIG.get_binary() == "pi"
    assert "@earendil-works/pi-coding-agent" in CONFIG.get_install_hint()
    assert CONFIG.uses_model_endpoint is True
    assert CONFIG.capabilities.permission_modes == {"read-only", "edit", "full"}
    assert CONFIG.capabilities.resume is True
    assert CONFIG.capabilities.tool_filtering is True
    assert isinstance(ProviderConfigManager.get_provider_harness_config(Harness.PI), PiHarnessConfig)


def test_validate_environment_rejects_wrong_options_managed_config_and_ask_mode():
    CONFIG.validate_environment(make_ctx())
    CONFIG.validate_environment(make_ctx(options=PiOptions(config={"mcpServers": {"moyai": MOYAI}})))
    with pytest.raises(OptionsMismatch):
        CONFIG.validate_environment(make_ctx(options=OpenCodeOptions()))
    with pytest.raises(OptionsMismatch, match="defaultModel"):
        CONFIG.validate_environment(make_ctx(options=PiOptions(config={"defaultModel": "openai/x"})))
    with pytest.raises(CapabilityUnsupported):
        CONFIG.validate_environment(make_ctx(permissions="ask"))


@pytest.mark.parametrize("key", sorted(MANAGED_ENV_KEYS))
def test_managed_env_keys_rejected_instead_of_silently_ignored(key):
    assert key in {PI_CONFIG_DIR_ENV, PI_TOKEN_ENV}
    with pytest.raises(OptionsMismatch, match=key):
        CONFIG.validate_environment(make_ctx(options=PiOptions(env={key: "/tmp/evil"})))
    CONFIG.validate_environment(make_ctx(options=PiOptions(env={f"{key}_SUFFIX": "ok"})))


def test_session_setup_token_only_in_env():
    setup = setup_for(make_ctx())
    assert set(setup.files) == {MODELS_FILENAME}
    assert all(TOKEN.encode() not in data for data in setup.files.values())
    assert setup.env[PI_TOKEN_ENV] == TOKEN
    provider = setup_models(setup)["providers"]["litellm"]
    assert provider["baseUrl"] == "http://host.docker.internal:4555/v1"
    assert provider["apiKey"] == f"${PI_TOKEN_ENV}"


def test_session_setup_env_and_persisted_sessions():
    """Managed keys win over PiOptions.env even here, behind validate_environment's rejection."""
    options = PiOptions(env={"FOO": "1", PI_TOKEN_ENV: "evil", PI_CONFIG_DIR_ENV: "/home/u/.pi/agent"})
    setup = setup_for(make_ctx(options=options))
    assert list(setup.persisted_dirs) == [("sessions", "pi/sessions")]
    assert setup.skills_dir == "skills"
    env = setup.env
    assert env[PI_CONFIG_DIR_ENV] == f"{PRIVATE}/agent"
    assert env[PI_TOKEN_ENV] == TOKEN
    for key, value in PI_ISOLATION_ENV.items():
        assert env[key] == value
    assert env["FOO"] == "1"


def test_session_setup_writes_user_config_next_to_managed_models():
    options = PiOptions(config={"mcpServers": {"moyai": MOYAI}, "compaction": {"enabled": False}})
    setup = setup_for(make_ctx(options=options))
    assert set(setup.files) == {MODELS_FILENAME, MCP_FILENAME, SETTINGS_FILENAME}
    assert json.loads(setup.files[MCP_FILENAME]) == {"mcpServers": {"moyai": MOYAI}}
    assert setup_models(setup)["providers"]["litellm"]["apiKey"] == f"${PI_TOKEN_ENV}"


def test_session_setup_errors():
    with pytest.raises(HarnessError):
        setup_for(make_ctx(endpoint=None))
    with pytest.raises(ValueError, match="needs model="):
        setup_for(make_ctx(model=None, endpoint=FakeEndpoint(model=None)))


def test_turn_request_without_a_model_raises_instead_of_passing_none():
    ctx = make_ctx(model=None, endpoint=FakeEndpoint(model=None))
    with pytest.raises(ValueError, match="needs model="):
        CONFIG.transform_turn_request(ctx, HarnessSessionSetup(), PRIVATE, "hi", None)


def test_session_setup_instructions_and_skills():
    ctx = make_ctx(instructions="Be terse.", output=Answer, skills=["/s/greeter"])
    setup = setup_for(ctx)
    written = setup.files[INSTRUCTIONS_FILENAME].decode()
    assert written.startswith("Be terse.")
    assert '"city"' in written and "single JSON object" in written
    argv = CONFIG.transform_turn_request(ctx, setup, PRIVATE, "hi", None).argv
    assert flag(argv, "--append-system-prompt") == f"{PRIVATE}/instructions.md"
    assert flag(argv, "--skill") == f"{PRIVATE}/skills"
    plain = make_ctx()
    plain_setup = setup_for(plain)
    plain_argv = CONFIG.transform_turn_request(plain, plain_setup, PRIVATE, "hi", None).argv
    assert INSTRUCTIONS_FILENAME not in plain_setup.files
    assert "--append-system-prompt" not in plain_argv and "--skill" not in plain_argv


def test_turn_request_argv_and_session_continuation():
    ctx = make_ctx(options=PiOptions(thinking="high"))
    setup = setup_for(ctx)
    first = CONFIG.transform_turn_request(ctx, setup, PRIVATE, "hello", None)
    assert list(first.argv) == [
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
        FULL_TOOLS,
        "--thinking",
        "high",
    ]
    assert first.cwd == "/work"
    assert first.stdin == "hello"
    assert first.env == setup.env
    second = CONFIG.transform_turn_request(ctx, setup, PRIVATE, "again", SESSION)
    assert flag(second.argv, "--session") == SESSION
    assert "again" not in " ".join(second.argv)


def test_turn_prompt_repeats_schema_when_output_set():
    ctx = make_ctx(output=Answer)
    request = CONFIG.transform_turn_request(ctx, setup_for(ctx), PRIVATE, "hi", None)
    assert request.stdin.startswith("hi\n\n") and '"city"' in request.stdin


def test_turn_response_paths():
    _, state = parse_all("structured_output.jsonl")
    ok = CONFIG.transform_turn_response(make_ctx(output=Answer), state, 0, [])
    assert json.loads(ok.output_json) == {"city": "Paris", "country": "France"}
    plain = CONFIG.transform_turn_response(make_ctx(), state, 0, [])
    assert plain.output_json is None and plain.final_text == state.final_text
    with pytest.raises(HarnessTurnError, match="boom"):
        CONFIG.transform_turn_response(make_ctx(), PiStreamState(stop_reason="error", error="boom"), 0, [])
    with pytest.raises(HarnessTurnError, match="aborted"):
        CONFIG.transform_turn_response(make_ctx(), PiStreamState(stop_reason="aborted"), 0, [])
    with pytest.raises(HarnessTurnError, match="code 3: no output"):
        CONFIG.transform_turn_response(make_ctx(), PiStreamState(), 3, [])


async def test_start_writes_token_only_in_env():
    handler, ctx, sandbox = await started()
    assert written_models(sandbox)["providers"]["litellm"]["apiKey"] == f"${PI_TOKEN_ENV}"
    assert all(TOKEN.encode() not in data for data in sandbox.files.values())
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "create hello.txt containing hi then read it")
    call = sandbox.execs[0]
    assert TOKEN not in json.dumps(call["cmd"])
    assert call["env"][PI_TOKEN_ENV] == TOKEN


async def test_start_persists_sessions_dir():
    _, _, sandbox = await started()
    assert sandbox.runs == [["sh", "-c", PERSIST_DIR_SCRIPT, "sh", "/tmp/pi-1/sessions", "pi/sessions"]]


async def test_persist_failure_still_uses_private_sessions():
    handler, ctx, sandbox = await started(FakeSandbox(persist_ok=False))
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "x")
    assert flag(sandbox.execs[0]["cmd"], "--session-dir") == "/tmp/pi-1/sessions"


async def test_turn_argv_env_and_session_continuation():
    handler, ctx, sandbox = await started(options=PiOptions(env={"FOO": "1"}))
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    events = await collect(handler, ctx, "create hello.txt containing hi then read it")
    first = sandbox.execs[0]
    assert first["cmd"] == [
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
        "/tmp/pi-1/sessions",
        "--tools",
        FULL_TOOLS,
    ]
    assert first["cwd"] == "/work"
    env = first["env"]
    assert env["PI_CODING_AGENT_DIR"] == "/tmp/pi-1/agent"
    assert env["PI_OFFLINE"] == "1" and env["PI_TELEMETRY"] == "0"
    assert env["FOO"] == "1"
    assert any(isinstance(e, ToolCall) for e in events)
    assert ctx.final_text == "Done! hello.txt contains: hi"
    assert handler.native_session_id() == SESSION

    sandbox.outputs.append(fixture_proc("turn2_resume.jsonl"))
    await collect(handler, ctx, "what file did you create?")
    second = sandbox.execs[1]["cmd"]
    assert flag(second, "--session") == SESSION
    assert "what file" not in " ".join(second)
    assert ctx.final_text == "You asked me to create hello.txt."


async def test_prompt_is_sent_on_stdin_not_argv():
    handler, ctx, sandbox = await started()
    proc = fixture_proc("turn1_write_read.jsonl")
    sandbox.outputs.append(proc)
    await collect(handler, ctx, "secret prompt text")
    assert proc.stdin.data == b"secret prompt text" and proc.stdin.closed
    assert "secret prompt text" not in sandbox.execs[0]["cmd"]


async def test_resume_sets_session():
    handler, ctx, sandbox = await started()
    await handler.resume(ctx, "prev-session")
    sandbox.outputs.append(fixture_proc("turn2_resume.jsonl"))
    await collect(handler, ctx, "hi")
    assert flag(sandbox.execs[0]["cmd"], "--session") == "prev-session"


async def test_read_only_and_disable_tools_argv():
    handler, ctx, sandbox = await started(permissions="read-only", disable_tools=["grep"])
    sandbox.outputs.append(fixture_proc("readonly_denied_bash.jsonl"))
    events = await collect(handler, ctx, "x")
    cmd = sandbox.execs[0]["cmd"]
    assert flag(cmd, "--tools") == "read,grep,find,ls"
    assert flag(cmd, "--exclude-tools") == "grep"
    assert ToolResult(id="call_b1", output="Tool bash not found", is_error=True) in events


async def test_instructions_and_structured_output():
    handler, ctx, sandbox = await started(instructions="Be terse.", output=Answer)
    written = sandbox.files["/tmp/pi-1/instructions.md"].decode()
    assert written.startswith("Be terse.")
    assert '"city"' in written and "single JSON object" in written
    proc = fixture_proc("structured_output.jsonl")
    sandbox.outputs.append(proc)
    await collect(handler, ctx, "x")
    assert flag(sandbox.execs[0]["cmd"], "--append-system-prompt") == "/tmp/pi-1/instructions.md"
    assert '"city"' in proc.stdin.data.decode()
    assert json.loads(ctx.output_json) == {"city": "Paris", "country": "France"}


async def test_skills_copied_to_private_skills_path(tmp_path):
    skill = tmp_path / "greeter"
    (skill / "ref").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: greeter\ndescription: d\n---\nbody")
    (skill / "ref" / "notes.txt").write_text("n")
    handler, ctx, sandbox = await started(skills=[str(skill)])
    assert sandbox.files["/tmp/pi-1/skills/greeter/SKILL.md"].startswith(b"---")
    assert sandbox.files["/tmp/pi-1/skills/greeter/ref/notes.txt"] == b"n"
    sandbox.outputs.append(fixture_proc("turn2_resume.jsonl"))
    await collect(handler, ctx, "x")
    cmd = sandbox.execs[0]["cmd"]
    assert flag(cmd, "--skill") == "/tmp/pi-1/skills"
    assert "--no-skills" in cmd


async def test_skill_without_manifest_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"SKILL\.md"):
        await started(skills=[str(tmp_path)])


async def test_missing_binary():
    with pytest.raises(HarnessInstallFailed, match="@earendil-works/pi-coding-agent"):
        await started(FakeSandbox(has_binary=False))


async def test_start_missing_endpoint_raises():
    with pytest.raises(HarnessError):
        await started(endpoint=None)


async def test_mcp_servers_written_and_reachable():
    handler, ctx, sandbox = await started(options=PiOptions(config={"mcpServers": {"moyai": MOYAI}}))
    assert json.loads(sandbox.files["/tmp/pi-1/agent/mcp.json"]) == {"mcpServers": {"moyai": MOYAI}}
    sandbox.outputs.append(fixture_proc("mcp_tool.jsonl"))
    events = await collect(handler, ctx, "call the moyai echo tool")
    assert flag(sandbox.execs[0]["cmd"], "--tools") == f"{FULL_TOOLS},mcp__*"
    assert ToolResult(id="call_m1", output="moyai says: hello", is_error=False) in events


async def test_wrong_options_and_ask_mode_rejected():
    with pytest.raises(OptionsMismatch):
        await started(options=OpenCodeOptions())
    with pytest.raises(OptionsMismatch):
        await started(options=PiOptions(config={"sessionDir": "/elsewhere"}))
    with pytest.raises(CapabilityUnsupported):
        await started(permissions="ask")


async def test_turn_before_start_raises():
    handler = CLIHarnessHandler(PiHarnessConfig())
    with pytest.raises(RuntimeError, match="before start"):
        await collect(handler, make_ctx(), "x")


async def test_api_error_raises_even_on_exit_zero():
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(fixture_proc("api_error.jsonl", exit_code=0))
    with pytest.raises(HarnessTurnError, match="no healthy deployments"):
        await collect(handler, ctx, "x")


async def test_nonzero_exit_raises_with_stderr_tail():
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(FakeProcess(b"", stderr=b"line1\nfatal: bad flag\n", exit_code=2))
    with pytest.raises(HarnessTurnError, match="code 2: line1\nfatal: bad flag"):
        await collect(handler, ctx, "x")


async def test_early_close_kills_process_and_stop_is_idempotent():
    handler, ctx, sandbox = await started()
    proc = fixture_proc("turn1_write_read.jsonl")
    sandbox.outputs.append(proc)
    gen = handler.turn(ctx, "x")
    await gen.__anext__()
    await gen.aclose()
    assert proc.killed
    await handler.stop(ctx)
    await handler.stop(ctx)


async def test_long_jsonl_line_is_parsed():
    handler, ctx, sandbox = await started()
    text = "x" * 200_000
    lines = [
        {"type": "session", "version": 3, "id": "s"},
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": text}},
        assistant_end(text, "stop"),
    ]
    sandbox.outputs.append(FakeProcess("\n".join(json.dumps(line) for line in lines).encode()))
    events = await collect(handler, ctx, "x")
    assert events == [Text(delta=text)]
    assert ctx.final_text == text


async def test_model_falls_back_to_endpoint_model():
    handler, ctx, sandbox = await started(model=None, endpoint=FakeEndpoint(model="gw-model"))
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "x")
    assert flag(sandbox.execs[0]["cmd"], "--model") == "gw-model"
    assert written_models(sandbox)["providers"]["litellm"]["models"] == [{"id": "gw-model"}]


def test_turn_request_never_trusts_project_files():
    """A repo's .pi/extensions would run as the host user at startup; --no-approve blocks it."""
    ctx = make_ctx()
    argv = list(CONFIG.transform_turn_request(ctx, setup_for(ctx), PRIVATE, "hi", None).argv)
    assert argv[:4] == ["pi", "--mode", "json", "--no-approve"]
