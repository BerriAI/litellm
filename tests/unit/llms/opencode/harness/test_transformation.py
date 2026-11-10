import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel

from litellm.harness.context import SessionContext
from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessError,
    HarnessInstallFailed,
    OptionsMismatch,
    SandboxError,
)
from litellm.harness.handlers.cli_handler import PERSIST_DIR_SCRIPT, CLIHarnessHandler
from litellm.harness.options import CodexOptions, OpenCodeOptions
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.types import Harness, Reasoning, Text, ToolCall, ToolResult
from litellm.llms.base_llm.harness.transformation import (
    HarnessSessionSetup,
    HarnessTurnError,
)
from litellm.llms.opencode.harness.transformation import (
    INSTRUCTIONS_FILENAME,
    OPENCODE_ISOLATION_ENV,
    OPENCODE_SESSION_TITLE,
    TOKEN_FILENAME,
    XDG_DIRNAME,
    OpenCodeHarnessConfig,
    OpenCodeStreamState,
    build_instructions,
    build_opencode_config,
    permission_rules,
    turn_prompt,
    validate_user_config,
)

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "tok-secret-123"
SESSION = "ses_f0cb977cdffeoCMeplOiw1KY25"
PRIVATE = "/tmp/oc-1"
CONFIG = OpenCodeHarnessConfig()


def load_fixture(name: str) -> list[dict]:
    return [
        json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line
    ]


def parse(obj: dict, state: OpenCodeStreamState) -> list:
    return CONFIG.transform_stream_line(obj, state)


def parse_all(name: str, state: OpenCodeStreamState | None = None):
    state = state or CONFIG.create_stream_state()
    events = []
    for obj in load_fixture(name):
        events.extend(parse(obj, state))
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


async def stall_until_timeout(cmd: Sequence[str], timeout: float | None) -> CompletedRun:
    if timeout is None:
        await asyncio.Event().wait()
    raise SandboxError(f"command timed out after {timeout}s: {cmd[0]}")


@dataclass
class FakeSandbox:
    workdir: str = "/work"
    has_binary: bool = True
    persist_ok: bool = True
    outputs: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    execs: list = field(default_factory=list)
    stall_runs: bool = False
    failing_subcommand: str | None = None
    run_stdout: str = ""
    runs: list[Sequence[str]] = field(default_factory=list)
    run_envs: list[dict[str, str]] = field(default_factory=list)
    tempdirs: int = 0

    async def exec(self, cmd, *, env=None, cwd=None):
        self.execs.append({"cmd": cmd, "env": dict(env or {}), "cwd": cwd})
        return self.outputs.pop(0)

    async def run(self, cmd, *, env=None, cwd=None, timeout=None):
        self.runs.append(cmd)
        self.run_envs.append(dict(env or {}))
        if self.stall_runs:
            return await stall_until_timeout(cmd, timeout)
        if not self.persist_ok or self.failing_subcommand in cmd:
            return CompletedRun("", "read-only fs", 1)
        return CompletedRun(self.run_stdout, "", 0)

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


def make_ctx(sandbox=None, **kwargs) -> SessionContext:
    return SessionContext(
        harness=Harness.OPENCODE,
        sandbox=sandbox or FakeSandbox(),
        session_id="s1",
        model=kwargs.pop("model", "claude-haiku-4-5-20251001"),
        endpoint=kwargs.pop("endpoint", FakeEndpoint()),
        **kwargs,
    )


def setup_for(ctx: SessionContext) -> HarnessSessionSetup:
    return CONFIG.transform_session_setup(ctx, PRIVATE)


def setup_config(setup: HarnessSessionSetup) -> dict:
    return json.loads(setup.env["OPENCODE_CONFIG_CONTENT"])


def fixture_proc(name: str, **kwargs) -> FakeProcess:
    return FakeProcess((FIXTURES / name).read_bytes(), **kwargs)


async def collect(handler, ctx, prompt):
    return [e async for e in handler.turn(ctx, prompt)]


async def started(sandbox=None, **kwargs):
    sandbox = sandbox or FakeSandbox()
    handler = CLIHarnessHandler(OpenCodeHarnessConfig())
    ctx = make_ctx(sandbox, **kwargs)
    await handler.start(ctx)
    return handler, ctx, sandbox


def exec_config(sandbox, index=0) -> dict:
    return json.loads(sandbox.execs[index]["env"]["OPENCODE_CONFIG_CONTENT"])


# --------------------------------------------------------------------------- parsing


def test_parse_write_read_turn():
    events, state = parse_all("turn1_write_read.jsonl")
    assert CONFIG.get_native_session_id(state) == SESSION
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
    state = OpenCodeStreamState()
    reasoning = parse(
        {"type": "reasoning", "sessionID": "s", "part": {"text": "thinking hard"}},
        state,
    )
    assert reasoning == [Reasoning(delta="thinking hard")]
    failed = parse(
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
        call = parse(
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
    mcp = parse(
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
    state = OpenCodeStreamState()
    parse({"type": "step_start"}, state)
    parse({"type": "text", "part": {"text": "working"}}, state)
    parse({"type": "step_start"}, state)
    parse({"type": "text", "part": {"text": "a"}}, state)
    parse({"type": "text", "part": {"text": "b"}}, state)
    assert state.final_text == "a\n\nb"


def test_error_event_message_shapes():
    state = OpenCodeStreamState()
    parse({"type": "error", "error": {"data": {"message": "m1"}}}, state)
    parse({"type": "error", "error": {"name": "APIError"}}, state)
    parse({"type": "error", "error": "raw"}, state)
    assert state.error == "m1\nAPIError\nraw"


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
        {"agent": "not-a-mapping"},
    ],
)
def test_managed_keys_rejected(config):
    with pytest.raises(OptionsMismatch):
        validate_user_config(config)


def test_config_metadata():
    assert CONFIG.get_binary() == "opencode"
    assert "opencode" in CONFIG.get_install_hint()
    assert CONFIG.uses_model_endpoint is True
    assert CONFIG.capabilities.permission_modes == {"read-only", "edit", "full"}


def test_validate_environment_rejects_wrong_options_and_managed_config():
    CONFIG.validate_environment(make_ctx())
    with pytest.raises(OptionsMismatch):
        CONFIG.validate_environment(make_ctx(options=CodexOptions()))
    with pytest.raises(OptionsMismatch):
        CONFIG.validate_environment(
            make_ctx(options=OpenCodeOptions(config={"model": "openai/x"}))
        )


def test_session_setup_token_only_in_private_file():
    setup = setup_for(make_ctx())
    assert setup.files == {TOKEN_FILENAME: TOKEN.encode()}
    assert TOKEN not in json.dumps(dict(setup.env))
    config = setup_config(setup)
    assert config["provider"]["litellm"]["options"] == {
        "baseURL": "http://host.docker.internal:4555/v1",
        "apiKey": "{file:/tmp/oc-1/token}",
    }
    assert config["permission"] == {"*": "allow"}


def test_session_setup_env_and_persisted_xdg():
    setup = setup_for(make_ctx(options=OpenCodeOptions(env={"FOO": "1"})))
    assert list(setup.persisted_dirs) == [(XDG_DIRNAME, "opencode")]
    assert setup.skills_dir == "skills"
    env = setup.env
    for sub in ("config", "data", "state", "cache"):
        assert env[f"XDG_{sub.upper()}_HOME"] == f"{PRIVATE}/xdg/{sub}"
    for key, value in OPENCODE_ISOLATION_ENV.items():
        assert env[key] == value
    assert env["OPENCODE_CONFIG"] == "" and env["OPENCODE_PERMISSION"] == ""
    assert env["FOO"] == "1"


def test_session_setup_errors():
    with pytest.raises(HarnessError):
        setup_for(make_ctx(endpoint=None))
    with pytest.raises(ValueError, match="needs model="):
        setup_for(make_ctx(model=None, endpoint=FakeEndpoint(model=None)))


def test_session_setup_read_only_and_disable_tools():
    setup = setup_for(make_ctx(permissions="read-only", disable_tools=["grep"]))
    assert setup_config(setup)["permission"] == {
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
        "grep": "deny",
    }


def test_session_setup_instructions_and_skills():
    ctx = make_ctx(instructions="Be terse.", output=Answer, skills=["/s/greeter"])
    setup = setup_for(ctx)
    written = setup.files[INSTRUCTIONS_FILENAME].decode()
    assert written == build_instructions(ctx)
    assert written.startswith("Be terse.")
    assert '"file"' in written and "single JSON object" in written
    config = setup_config(setup)
    assert config["instructions"] == ["/tmp/oc-1/instructions.md"]
    assert config["skills"] == {"paths": ["/tmp/oc-1/skills"]}
    assert build_instructions(make_ctx()) is None
    assert "skills" not in setup_config(setup_for(make_ctx()))


def test_turn_request_argv_and_session_continuation():
    ctx = make_ctx(options=OpenCodeOptions(agent="build"))
    setup = setup_for(ctx)
    first = CONFIG.transform_turn_request(ctx, setup, PRIVATE, "hello", None)
    assert list(first.argv) == [
        "opencode",
        "run",
        "--pure",
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
    assert first.cwd == "/work"
    assert first.stdin == "hello"
    assert first.env == setup.env
    second = CONFIG.transform_turn_request(ctx, setup, PRIVATE, "again", SESSION)
    argv = list(second.argv)
    assert argv[argv.index("--session") + 1] == SESSION
    assert "--title" not in argv
    assert "again" not in " ".join(argv)


def test_turn_prompt_repeats_schema_when_output_set():
    assert turn_prompt(make_ctx(), "hi") == "hi"
    prompt = turn_prompt(make_ctx(output=Answer), "hi")
    assert prompt.startswith("hi\n\n") and '"file"' in prompt
    ctx = make_ctx(output=Answer)
    request = CONFIG.transform_turn_request(ctx, setup_for(ctx), PRIVATE, "hi", None)
    assert request.stdin == prompt


def test_turn_response_paths():
    state = OpenCodeStreamState(final_text='Here: {"file": "a", "content": "hi"}')
    ok = CONFIG.transform_turn_response(make_ctx(output=Answer), state, 0, [])
    assert json.loads(ok.output_json) == {"file": "a", "content": "hi"}
    plain = CONFIG.transform_turn_response(make_ctx(), state, 0, [])
    assert plain.output_json is None and plain.final_text == state.final_text
    with pytest.raises(HarnessTurnError, match="boom"):
        CONFIG.transform_turn_response(
            make_ctx(), OpenCodeStreamState(error="boom"), 0, []
        )
    with pytest.raises(HarnessTurnError, match="code 3: no output"):
        CONFIG.transform_turn_response(make_ctx(), OpenCodeStreamState(), 3, [])


# --------------------------------------------------------------------------- handler


async def test_start_writes_token_only_in_private_file():
    handler, ctx, sandbox = await started()
    assert sandbox.files["/tmp/oc-1/token"] == TOKEN.encode()
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "create hello.txt containing hi then read it")
    call = sandbox.execs[0]
    assert TOKEN not in json.dumps(call["cmd"])
    assert TOKEN not in json.dumps(call["env"])
    config = exec_config(sandbox)
    assert config["provider"]["litellm"]["options"] == {
        "baseURL": "http://host.docker.internal:4555/v1",
        "apiKey": "{file:/tmp/oc-1/token}",
    }
    assert config["permission"] == {"*": "allow"}


async def test_start_persists_xdg_dir():
    _, _, sandbox = await started()
    assert sandbox.runs == [
        ["sh", "-c", PERSIST_DIR_SCRIPT, "sh", "/tmp/oc-1/xdg", "opencode"]
    ]


async def test_persist_failure_still_uses_private_xdg():
    handler, ctx, sandbox = await started(FakeSandbox(persist_ok=False))
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "x")
    assert sandbox.execs[0]["env"]["XDG_DATA_HOME"] == "/tmp/oc-1/xdg/data"


async def test_turn_argv_env_and_session_continuation():
    handler, ctx, sandbox = await started(
        options=OpenCodeOptions(agent="build", env={"FOO": "1"})
    )
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    events = await collect(handler, ctx, "create hello.txt containing hi then read it")
    first = sandbox.execs[0]
    assert first["cmd"] == [
        "opencode",
        "run",
        "--pure",
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
    assert env["XDG_CONFIG_HOME"] == "/tmp/oc-1/xdg/config"
    assert env["XDG_DATA_HOME"] == "/tmp/oc-1/xdg/data"
    assert env["XDG_STATE_HOME"] == "/tmp/oc-1/xdg/state"
    assert env["XDG_CACHE_HOME"] == "/tmp/oc-1/xdg/cache"
    assert env["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    assert env["OPENCODE_CONFIG"] == "" and env["OPENCODE_PERMISSION"] == ""
    assert env["FOO"] == "1"
    assert any(isinstance(e, ToolCall) for e in events)
    assert ctx.final_text.startswith("Done!")
    assert handler.native_session_id() == SESSION

    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(handler, ctx, "what file did you create?")
    second = sandbox.execs[1]["cmd"]
    assert second[second.index("--session") + 1] == SESSION
    assert "--title" not in second
    assert "what file" not in " ".join(second)


async def test_prompt_is_sent_on_stdin_not_argv():
    handler, ctx, sandbox = await started()
    proc = fixture_proc("turn1_write_read.jsonl")
    sandbox.outputs.append(proc)
    await collect(handler, ctx, "secret prompt text")
    assert proc.stdin.data == b"secret prompt text" and proc.stdin.closed
    assert "secret prompt text" not in sandbox.execs[0]["cmd"]


SAVED: Final = "ses_ea6d348fdffe9pu36Y2YN6YobB"
SAVED_USER_MSG: Final = "msg_1592cb71a001307YHyO7ReafRb"
SAVED_REPLY_MSG: Final = "msg_1592cb7f4001KgUDK3Sj79XDT4"
SAVED_PARTS: Final = ("prt_1592cb71c0013NJaqSsv4dNiSR", "prt_1592cbe0f001oKdE9pRy38AduM")
COPY_PATH: Final = "/tmp/oc-1/resume-session.json"


def exported_session(directory: str) -> str:
    user_part: Final = {"id": SAVED_PARTS[0], "sessionID": SAVED, "messageID": SAVED_USER_MSG, "type": "text"}
    reply_part: Final = {"id": SAVED_PARTS[1], "sessionID": SAVED, "messageID": SAVED_REPLY_MSG, "type": "text"}
    user: Final = {"id": SAVED_USER_MSG, "sessionID": SAVED, "role": "user"}
    reply: Final = {"id": SAVED_REPLY_MSG, "sessionID": SAVED, "parentID": SAVED_USER_MSG, "role": "assistant"}
    return json.dumps(
        {
            "info": {"id": SAVED, "directory": directory, "title": "litellm-harness"},
            "messages": [
                {"info": user, "parts": [{**user_part, "text": "Remember 4817"}]},
                {"info": reply, "parts": [{**reply_part, "text": "OK"}]},
            ],
        }
    )


async def resumed(saved_in: str) -> tuple[CLIHarnessHandler, SessionContext, FakeSandbox]:
    handler, ctx, sandbox = await started()
    sandbox.run_stdout = exported_session(saved_in)
    await handler.resume(ctx, SAVED)
    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(handler, ctx, "hi")
    return handler, ctx, sandbox


def resumed_session_arg(sandbox: FakeSandbox) -> str:
    cmd: Final = sandbox.execs[0]["cmd"]
    return cmd[cmd.index("--session") + 1]


async def test_resume_elsewhere_continues_a_copy_filed_under_the_sandbox() -> None:
    _, _, sandbox = await resumed("/other")

    copy: Final = json.loads(sandbox.files[COPY_PATH])
    copy_id: Final = resumed_session_arg(sandbox)
    assert copy["info"]["id"] == copy_id != SAVED
    assert copy["info"]["directory"] == sandbox.workdir
    assert tuple(sandbox.runs[-1]) == ("opencode", "--pure", "import", COPY_PATH)
    assert sandbox.run_envs[-1]["XDG_DATA_HOME"] == "/tmp/oc-1/xdg/data"
    assert sandbox.run_envs[-2]["XDG_DATA_HOME"] == "/tmp/oc-1/xdg/data"


async def test_resume_copy_gets_fresh_ids_that_keep_their_links_and_order() -> None:
    _, _, sandbox = await resumed("/other")

    raw_copy: Final = sandbox.files[COPY_PATH].decode()
    user, reply = json.loads(raw_copy)["messages"]
    copy_id: Final = resumed_session_arg(sandbox)
    assert all(old not in raw_copy for old in (SAVED, SAVED_USER_MSG, SAVED_REPLY_MSG, *SAVED_PARTS))
    assert {user["info"]["sessionID"], reply["info"]["sessionID"]} == {copy_id}
    assert reply["info"]["parentID"] == user["info"]["id"]
    assert user["parts"][0]["messageID"] == user["info"]["id"]
    assert reply["parts"][0]["messageID"] == reply["info"]["id"]
    assert {user["parts"][0]["sessionID"], reply["parts"][0]["sessionID"]} == {copy_id}
    assert (user["parts"][0]["text"], reply["parts"][0]["text"]) == ("Remember 4817", "OK")


def test_resume_copy_keeps_the_message_order() -> None:
    message_ids: Final = tuple(f"msg_1592cb71a{n:03x}307YHyO7ReafRb" for n in range(10))
    messages: Final = [{"info": {"id": message_id}, "parts": []} for message_id in message_ids]
    exported: Final = json.dumps({"info": {"id": SAVED, "directory": "/other"}, "messages": messages})

    copy: Final = CONFIG.transform_resume_copy(exported, PRIVATE, "/work")

    assert copy is not None
    copied_ids: Final = [message["info"]["id"] for message in json.loads(copy.data)["messages"]]
    assert sorted(copied_ids) == copied_ids


def test_resume_copy_keeps_saved_text_that_reads_like_an_id() -> None:
    tool_part: Final = {
        "id": SAVED_PARTS[1],
        "sessionID": SAVED,
        "messageID": SAVED_USER_MSG,
        "type": "tool",
        "state": {"input": {"command": SAVED_REPLY_MSG}, "output": SAVED_USER_MSG},
    }
    text_part: Final = {"id": SAVED_PARTS[0], "sessionID": SAVED, "messageID": SAVED_USER_MSG, "type": "text", "text": SAVED}
    message: Final = {"info": {"id": SAVED_USER_MSG, "sessionID": SAVED}, "parts": [text_part, tool_part]}
    exported: Final = json.dumps({"info": {"id": SAVED, "directory": "/other"}, "messages": [message]})

    copy: Final = CONFIG.transform_resume_copy(exported, PRIVATE, "/work")

    assert copy is not None
    text, tool = json.loads(copy.data)["messages"][0]["parts"]
    assert text["text"] == SAVED != text["sessionID"] == copy.session_id
    assert (tool["state"]["input"]["command"], tool["state"]["output"]) == (SAVED_REPLY_MSG, SAVED_USER_MSG)
    assert tool["messageID"] != SAVED_USER_MSG


async def test_later_turns_after_a_resume_do_not_export_again() -> None:
    handler, ctx, sandbox = await resumed("/other")

    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(handler, ctx, "again")

    assert [run[2] for run in sandbox.runs if run[0] == "opencode"] == ["export", "import"]


async def test_each_resume_elsewhere_makes_its_own_copy() -> None:
    _, _, first = await resumed("/other")
    _, _, second = await resumed("/other")

    assert resumed_session_arg(first) != resumed_session_arg(second)


async def test_resume_in_the_folder_it_was_saved_in_continues_the_original() -> None:
    _, _, sandbox = await resumed("/work")

    assert resumed_session_arg(sandbox) == SAVED
    assert COPY_PATH not in sandbox.files
    assert all("import" not in run for run in sandbox.runs)


@pytest.mark.parametrize("failure", ["export_exits_1", "import_exits_1", "export_stalls", "export_unreadable"])
async def test_resume_that_cannot_be_prepared_fails_the_turn_without_running_opencode(
    failure: str,
) -> None:
    handler, ctx, sandbox = await started()
    sandbox.run_stdout = "Error: not a session" if failure == "export_unreadable" else exported_session("/other")
    sandbox.failing_subcommand = failure.split("_")[0] if failure.endswith("exits_1") else None
    sandbox.stall_runs = failure == "export_stalls"

    await asyncio.wait_for(handler.resume(ctx, SAVED), 5)
    with pytest.raises(HarnessTurnError, match=SAVED):
        await collect(handler, ctx, "hi")
    assert sandbox.execs == []


async def test_resume_runs_nothing_until_the_first_turn() -> None:
    handler, ctx, sandbox = await started()
    runs_before_resume: Final = len(sandbox.runs)

    await handler.resume(ctx, SAVED)

    assert len(sandbox.runs) == runs_before_resume
    assert COPY_PATH not in sandbox.files


async def test_restart_after_a_failed_resume_tries_the_resume_again() -> None:
    handler, ctx, sandbox = await started()
    sandbox.run_stdout = exported_session("/other")
    sandbox.failing_subcommand = "export"
    await handler.resume(ctx, SAVED)
    with pytest.raises(HarnessTurnError, match=SAVED):
        await collect(handler, ctx, "hi")

    sandbox.failing_subcommand = None
    await handler.start(ctx)
    await handler.resume(ctx, SAVED)
    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(handler, ctx, "again")

    assert resumed_session_arg(sandbox) == json.loads(sandbox.files["/tmp/oc-2/resume-session.json"])["info"]["id"]


async def test_restart_resumes_its_own_session_without_exporting_it() -> None:
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "first")
    native_id: Final = handler.native_session_id()
    assert native_id is not None
    runs_before_restart: Final = len(sandbox.runs)

    await handler.start(ctx)
    await handler.resume(ctx, native_id)
    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(handler, ctx, "again")

    assert all("export" not in run for run in sandbox.runs[runs_before_restart:])
    cmd: Final = sandbox.execs[1]["cmd"]
    assert cmd[cmd.index("--session") + 1] == native_id


async def test_read_only_and_disable_tools_config():
    handler, ctx, sandbox = await started(
        permissions="read-only", disable_tools=["grep"]
    )
    sandbox.outputs.append(fixture_proc("readonly_denied_bash.jsonl"))
    await collect(handler, ctx, "x")
    assert exec_config(sandbox)["permission"] == {
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
        "grep": "deny",
    }


async def test_instructions_and_structured_output():
    handler, ctx, sandbox = await started(instructions="Be terse.", output=Answer)
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
    proc = FakeProcess("\n".join(json.dumps(line) for line in lines).encode())
    sandbox.outputs.append(proc)
    await collect(handler, ctx, "x")
    assert exec_config(sandbox)["instructions"] == ["/tmp/oc-1/instructions.md"]
    assert '"file"' in proc.stdin.data.decode()
    assert json.loads(ctx.output_json) == {"file": "a", "content": "hi"}


async def test_skills_copied_to_private_skills_path(tmp_path):
    skill = tmp_path / "greeter"
    (skill / "ref").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: greeter\ndescription: d\n---\nbody")
    (skill / "ref" / "notes.txt").write_text("n")
    handler, ctx, sandbox = await started(skills=[str(skill)])
    assert sandbox.files["/tmp/oc-1/skills/greeter/SKILL.md"].startswith(b"---")
    assert sandbox.files["/tmp/oc-1/skills/greeter/ref/notes.txt"] == b"n"
    sandbox.outputs.append(fixture_proc("turn2_session_skill.jsonl"))
    await collect(handler, ctx, "x")
    assert exec_config(sandbox)["skills"] == {"paths": ["/tmp/oc-1/skills"]}


async def test_missing_binary():
    with pytest.raises(HarnessInstallFailed, match="opencode"):
        await started(FakeSandbox(has_binary=False))


async def test_wrong_options_and_managed_config_rejected():
    with pytest.raises(OptionsMismatch):
        await started(options=CodexOptions())
    with pytest.raises(OptionsMismatch):
        await started(options=OpenCodeOptions(config={"model": "openai/x"}))


async def test_turn_before_start_raises():
    handler = CLIHarnessHandler(OpenCodeHarnessConfig())
    with pytest.raises(RuntimeError, match="before start"):
        await collect(handler, make_ctx(), "x")


async def test_api_error_event_raises_even_on_exit_zero():
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(fixture_proc("api_error.jsonl"))
    with pytest.raises(HarnessTurnError, match="no healthy deployments"):
        await collect(handler, ctx, "x")


async def test_nonzero_exit_raises_with_stderr_tail():
    handler, ctx, sandbox = await started()
    sandbox.outputs.append(
        FakeProcess(b"", stderr=b"line1\nfatal: bad config\n", exit_code=2)
    )
    with pytest.raises(HarnessTurnError, match="code 2: line1\nfatal: bad config"):
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


async def test_model_falls_back_to_endpoint_model():
    handler, ctx, sandbox = await started(
        model=None, endpoint=FakeEndpoint(model="gw-model")
    )
    sandbox.outputs.append(fixture_proc("turn1_write_read.jsonl"))
    await collect(handler, ctx, "x")
    assert "litellm/gw-model" in sandbox.execs[0]["cmd"]
    assert exec_config(sandbox)["provider"]["litellm"]["models"] == {"gw-model": {}}


def test_turn_request_never_loads_plugins():
    """A repo's .opencode/plugin/*.js would run as the host user at startup; --pure blocks it."""
    ctx = make_ctx()
    argv = list(CONFIG.transform_turn_request(ctx, setup_for(ctx), PRIVATE, "hi", None).argv)
    assert argv[:3] == ["opencode", "run", "--pure"]


def test_options_config_cannot_add_plugins():
    with pytest.raises(OptionsMismatch, match="plugin"):
        validate_user_config({"plugin": ["./evil.js"]})


def test_endpoint_request_fixture_documents_contract():
    requests = load_fixture("endpoint_requests.jsonl")
    assert {r["path"] for r in requests} == {"/v1/chat/completions"}
    assert all(r["stream"] is True for r in requests)
    assert all(r["stream_options"] == {"include_usage": True} for r in requests)
