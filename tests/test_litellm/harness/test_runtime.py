"""Tests for litellm/harness/runtime.py using a fake adapter, sandbox and endpoint."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from litellm.harness import runtime
from litellm.harness.adapters.base import SessionContext
from litellm.harness.errors import (
    CapabilityUnsupported,
    HarnessInstallFailed,
    OptionsMismatch,
    OutputInvalid,
    SessionClosed,
    StateIncompatible,
)
from litellm.harness.options import CodexOptions
from litellm.harness.types import (
    Approval,
    Done,
    Event,
    FileChange,
    Gateway,
    Harness,
    State,
    Text,
    ToolCall,
)
from tests.test_litellm.harness.core_fakes import (
    NARROW_CAPS,
    FakeAdapter,
    FakeEndpoint,
    FakeSandbox,
    install_adapter,
    script_approval,
    wait_forever,
)


class Answer(BaseModel):
    value: int


@pytest.fixture
def sandbox(tmp_path) -> FakeSandbox:
    return FakeSandbox(str(tmp_path))


async def _collect(stream) -> list[Event]:
    return [event async for event in stream]


# -- validation ---------------------------------------------------------------


async def test_string_harness_raises_type_error_with_hint(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    with pytest.raises(TypeError, match=r"Harness\.CODEX"):
        await runtime.arun("codex", "hi", sandbox=sandbox)  # type: ignore[arg-type]


async def test_options_mismatch(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch)
    with pytest.raises(OptionsMismatch, match="CodexOptions"):
        await runtime.arun(
            Harness.CLAUDE_CODE, "hi", sandbox=sandbox, options=CodexOptions()
        )
    assert adapter_cls.instances == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"permissions": "edit"},
        {"output": Answer},
        {"tools": [print]},
        {"disable_tools": ["bash"]},
        {"permissions": "ask", "on_approval": lambda a: True},
    ],
)
async def test_capability_errors_before_start(monkeypatch, sandbox, kwargs):
    adapter_cls = install_adapter(monkeypatch, caps=NARROW_CAPS)
    with pytest.raises(CapabilityUnsupported):
        await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, **kwargs)
    assert all("start" not in a.calls for a in adapter_cls.instances)
    assert FakeEndpoint.instances == []


async def test_skills_capability_error_before_start(monkeypatch, sandbox, tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# s")
    adapter_cls = install_adapter(monkeypatch, caps=NARROW_CAPS)
    with pytest.raises(CapabilityUnsupported, match="skills"):
        await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, skills=[skill])
    assert adapter_cls.instances == []


async def test_skill_folder_without_skill_md_rejected(monkeypatch, sandbox, tmp_path):
    install_adapter(monkeypatch)
    with pytest.raises(ValueError, match="SKILL.md"):
        await runtime.arun(
            Harness.CLAUDE_CODE, "hi", sandbox=sandbox, skills=[tmp_path]
        )


async def test_ask_without_handler_only_allowed_for_stream(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    with pytest.raises(ValueError, match="on_approval"):
        await runtime.arun(
            Harness.CLAUDE_CODE, "hi", sandbox=sandbox, permissions="ask"
        )
    stream = runtime.astream(
        Harness.CLAUDE_CODE, "hi", sandbox=sandbox, permissions="ask"
    )
    events = await _collect(stream)
    assert isinstance(events[-1], Done)


async def test_invalid_permissions_value(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    with pytest.raises(ValueError, match="permissions"):
        await runtime.arun(
            Harness.CLAUDE_CODE, "hi", sandbox=sandbox, permissions="yolo"  # type: ignore[arg-type]
        )


# -- gateway resolution -------------------------------------------------------


def test_gateway_from_env(monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://gw.example.com/")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-test")
    gateway = runtime.resolve_gateway(None)
    assert gateway == Gateway(api_base="https://gw.example.com", api_key="sk-test")


def test_gateway_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://env.example.com")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-env")
    explicit = Gateway(api_base="https://arg.example.com", api_key="sk-arg")
    assert runtime.resolve_gateway(explicit) is explicit


def test_gateway_env_empty_key_raises(monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://gw.example.com")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "   ")
    with pytest.raises(ValueError, match="LITELLM_PROXY_API_KEY"):
        runtime.resolve_gateway(None)


def test_gateway_absent_is_sdk_mode(monkeypatch):
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    assert runtime.resolve_gateway(None) is None


async def test_gateway_passed_to_endpoint(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://gw.example.com")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-test")
    await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, model="m")
    endpoint = FakeEndpoint.instances[0]
    assert endpoint.gateway.api_key == "sk-test"
    assert endpoint.model == "m"
    assert endpoint.entered and endpoint.exited


# -- event flow ---------------------------------------------------------------


async def test_text_and_tool_events_flow_and_done_last(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch)
    events = await _collect(runtime.astream(Harness.CLAUDE_CODE, "hi", sandbox=sandbox))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["Text", "ToolCall", "ToolResult", "Text", "Done"]
    assert sum(isinstance(e, Done) for e in events) == 1
    result = events[-1].result
    assert result.text == "hello world"
    assert result.stop_reason == "done"
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 5
    assert result.cost == pytest.approx(0.25)
    assert adapter_cls.instances[0].calls == ["start", "turn", "stop"]


async def test_arun_returns_result(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    result = await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert result.text == "hello world"
    assert len(result.events) == 4


async def test_final_text_from_ctx_preferred(monkeypatch, sandbox):
    async def script(adapter, ctx: SessionContext, prompt) -> AsyncIterator[Event]:
        yield Text("partial")
        ctx.final_text = "final answer"

    install_adapter(monkeypatch, script=script)
    result = await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert result.text == "final answer"


async def test_endpointless_adapter_usage(monkeypatch, sandbox):
    install_adapter(monkeypatch, uses_endpoint=False)
    result = await runtime.arun(Harness.DEEPAGENTS, "hi", sandbox=sandbox)
    assert FakeEndpoint.instances == []
    assert result.usage.calls == 1
    assert result.cost == pytest.approx(0.25)


async def test_stream_result_property(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    stream = runtime.astream(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert stream.result is None
    await _collect(stream)
    assert stream.result is not None and stream.result.text == "hello world"


# -- stop reasons -------------------------------------------------------------


async def _tool_loop(adapter, ctx, prompt) -> AsyncIterator[Event]:
    for i in range(10):
        yield ToolCall(id=str(i), name="bash", native_name="Bash", input={})


async def test_max_turns_stop(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch, script=_tool_loop)
    result = await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, max_turns=3)
    assert result.stop_reason == "max_turns"
    assert sum(isinstance(e, ToolCall) for e in result.events) == 3
    assert "stop" in adapter_cls.instances[0].calls


async def _slow(adapter, ctx, prompt) -> AsyncIterator[Event]:
    yield Text("thinking")
    await wait_forever()
    yield Text("never")


async def test_timeout_stop(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_slow)
    result = await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, timeout=0.2)
    assert result.stop_reason == "timeout"
    assert result.text == "thinking"


async def test_cancel_stop(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_slow)
    stream = runtime.astream(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    events = []
    async for event in stream:
        events.append(event)
        if isinstance(event, Text):
            stream.cancel()
    assert isinstance(events[-1], Done)
    assert events[-1].stop_reason == "cancelled"


async def _crash(adapter, ctx, prompt) -> AsyncIterator[Event]:
    yield Text("partial")
    raise RuntimeError("process exited with code 1")


async def test_runtime_error_stop_reason(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_crash)
    result = await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert result.stop_reason == "runtime_error"
    assert "process exited with code 1" in result.text


async def _missing_binary(adapter, ctx, prompt) -> AsyncIterator[Event]:
    raise HarnessInstallFailed("claude not found on PATH")
    yield Text("unreachable")  # pragma: no cover


async def test_install_failed_propagates(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_missing_binary)
    with pytest.raises(HarnessInstallFailed, match="claude"):
        await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert FakeEndpoint.instances[0].exited


# -- approvals ----------------------------------------------------------------


async def test_approval_on_approval_allow(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    result = await runtime.arun(
        Harness.CLAUDE_CODE,
        "hi",
        sandbox=sandbox,
        permissions="ask",
        on_approval=lambda approval: approval.tool == "bash",
    )
    assert adapter_cls.instances[0].approvals[0][0] is True
    assert result.text == "allowed"


async def test_approval_async_handler_deny(monkeypatch, sandbox):
    async def handler(approval: Approval) -> bool:
        await asyncio.sleep(0)
        return False

    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    result = await runtime.arun(
        Harness.CLAUDE_CODE,
        "hi",
        sandbox=sandbox,
        permissions="ask",
        on_approval=handler,
    )
    assert adapter_cls.instances[0].approvals[0][0] is False
    assert result.text == "denied"


async def test_approval_handler_raises_denies(monkeypatch, sandbox):
    def handler(approval: Approval) -> bool:
        raise RuntimeError("boom")

    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    result = await runtime.arun(
        Harness.CLAUDE_CODE,
        "hi",
        sandbox=sandbox,
        permissions="ask",
        on_approval=handler,
    )
    allowed, reason = adapter_cls.instances[0].approvals[0]
    assert allowed is False and "boom" in reason
    assert result.stop_reason == "done"


async def test_stream_consumer_answers_approval(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    stream = runtime.astream(
        Harness.CLAUDE_CODE, "hi", sandbox=sandbox, permissions="ask"
    )
    async for event in stream:
        if isinstance(event, Approval):
            event.allow()
    assert adapter_cls.instances[0].approvals[0][0] is True


async def test_unanswered_approval_denied(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    events = await _collect(
        runtime.astream(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, permissions="ask")
    )
    allowed, reason = adapter_cls.instances[0].approvals[0]
    assert allowed is False and "not answered" in reason
    assert isinstance(events[-1], Done)


async def test_approval_without_ask_denied_in_run(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert adapter_cls.instances[0].approvals[0][0] is False


# -- structured output --------------------------------------------------------


def _answer_script(text: str, output_json: str | None = None):
    async def script(adapter, ctx, prompt) -> AsyncIterator[Event]:
        yield Text(text)
        ctx.output_json = output_json

    return script


async def test_structured_output_from_text(monkeypatch, sandbox):
    install_adapter(
        monkeypatch, script=_answer_script('Sure. {"x": 1} then {"value": 42} done')
    )
    result = await runtime.arun(
        Harness.CLAUDE_CODE, "hi", sandbox=sandbox, output=Answer
    )
    assert result.output == Answer(value=42)


async def test_structured_output_from_ctx_output_json(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_answer_script("ok", '{"value": 7}'))
    result = await runtime.arun(
        Harness.CLAUDE_CODE, "hi", sandbox=sandbox, output=Answer
    )
    assert result.output == Answer(value=7)


async def test_structured_output_invalid_carries_result(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_answer_script('{"value": "nope"}'))
    with pytest.raises(OutputInvalid) as info:
        await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, output=Answer)
    assert info.value.raw == '{"value": "nope"}'
    assert info.value.result is not None
    assert info.value.result.text == '{"value": "nope"}'


async def test_structured_output_missing_json(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_answer_script("no json here"))
    with pytest.raises(OutputInvalid, match="no JSON"):
        await runtime.arun(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, output=Answer)


async def test_stream_yields_done_before_output_invalid(monkeypatch, sandbox):
    install_adapter(monkeypatch, script=_answer_script("nothing"))
    stream = runtime.astream(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, output=Answer)
    seen = []
    with pytest.raises(OutputInvalid):
        async for event in stream:
            seen.append(event)
    assert isinstance(seen[-1], Done)


def test_last_json_object():
    assert runtime.last_json_object('a {"a": {"b": 1}} b {"c": 2}') == '{"c": 2}'
    assert runtime.last_json_object("{broken") is None


# -- files --------------------------------------------------------------------


async def _edit_files(adapter, ctx, prompt) -> AsyncIterator[Event]:
    root = ctx.sandbox.workdir
    with open(os.path.join(root, "new.txt"), "w") as fh:
        fh.write("new\n")
    with open(os.path.join(root, "keep.txt"), "w") as fh:
        fh.write("changed\n")
    os.remove(os.path.join(root, "gone.txt"))
    yield FileChange(path="new.txt", kind="created", diff=None)
    yield Text("edited")


async def test_file_changes_emitted_once(monkeypatch, sandbox, tmp_path):
    (tmp_path / "keep.txt").write_text("original\n")
    (tmp_path / "gone.txt").write_text("bye\n")
    install_adapter(monkeypatch, script=_edit_files)
    events = await _collect(runtime.astream(Harness.CLAUDE_CODE, "hi", sandbox=sandbox))
    file_events = [e for e in events if isinstance(e, FileChange)]
    assert sorted((e.path, e.kind) for e in file_events) == [
        ("gone.txt", "deleted"),
        ("keep.txt", "modified"),
        ("new.txt", "created"),
    ]
    result = events[-1].result
    by_path = {f.path: f for f in result.files}
    assert set(by_path) == {"gone.txt", "keep.txt", "new.txt"}
    assert "+changed" in by_path["keep.txt"].diff
    assert "-bye" in by_path["gone.txt"].diff
    assert "+new" in by_path["new.txt"].diff
    assert isinstance(events[-1], Done)


# -- sessions -----------------------------------------------------------------


async def test_session_multi_turn_cost(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch)
    async with runtime.asession(Harness.CLAUDE_CODE, sandbox=sandbox) as session:
        first = await session.arun("one")
        second = await session.arun("two")
        assert first.cost == pytest.approx(0.25)
        assert second.cost == pytest.approx(0.25)
        assert second.usage.input_tokens == 10
        assert session.cost == pytest.approx(0.5)
        assert session.usage.calls == 2
        assert await session.history() == [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
    adapter = adapter_cls.instances[0]
    assert adapter.calls == ["start", "turn", "turn", "stop"]
    assert len(FakeEndpoint.instances) == 1
    with pytest.raises(SessionClosed):
        await session.arun("three")


async def test_await_asession(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    session = await runtime.asession(Harness.CLAUDE_CODE, sandbox=sandbox)
    result = await session.arun("hi")
    await session.aclose()
    assert result.text == "hello world"


async def test_session_restarts_after_timeout(monkeypatch, sandbox):
    calls = {"n": 0}

    async def script(adapter, ctx, prompt) -> AsyncIterator[Event]:
        calls["n"] += 1
        if calls["n"] == 1:
            await wait_forever()
        yield Text("ok")

    adapter_cls = install_adapter(monkeypatch, script=script)
    async with runtime.asession(
        Harness.CLAUDE_CODE, sandbox=sandbox, timeout=0.2
    ) as session:
        assert (await session.arun("one")).stop_reason == "timeout"
        assert (await session.arun("two")).text == "ok"
    adapter = adapter_cls.instances[0]
    assert adapter.calls[:4] == ["start", "turn", "stop", "start"]
    assert adapter.resumed_with == "native-123"


async def test_detach_state_round_trip_resume(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch)
    async with runtime.asession(
        Harness.CLAUDE_CODE, sandbox=sandbox, model="m1"
    ) as session:
        await session.arun("one")
        state = await session.adetach()
    data = state.dumps()
    assert b"sk-" not in data
    restored = State.loads(data)
    assert restored == state and restored.native_session_id == "native-123"

    async with runtime.aresume(data, sandbox=sandbox) as resumed:
        await resumed.arun("two")
    new_adapter = adapter_cls.instances[-1]
    assert new_adapter.calls[:2] == ["start", "resume"]
    assert new_adapter.resumed_with == "native-123"
    assert resumed.config.model == "m1"


async def test_resume_requires_capability(monkeypatch, sandbox):
    install_adapter(
        monkeypatch,
        caps=NARROW_CAPS.__class__(**{**NARROW_CAPS.__dict__, "resume": False}),
    )
    state = State(harness=Harness.CODEX, native_session_id="x", workdir="/tmp")
    with pytest.raises(CapabilityUnsupported, match="resume"):
        runtime.aresume(state, sandbox=sandbox)


async def test_resume_state_without_native_id(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    state = State(harness=Harness.CODEX, native_session_id=None, workdir="/tmp")
    with pytest.raises(StateIncompatible):
        runtime.aresume(state, sandbox=sandbox)


async def test_history_requires_capability(monkeypatch, sandbox):
    install_adapter(monkeypatch, caps=NARROW_CAPS)
    async with runtime.asession(Harness.CLAUDE_CODE, sandbox=sandbox) as session:
        with pytest.raises(CapabilityUnsupported):
            await session.history()


def test_capabilities_uses_registry(monkeypatch):
    install_adapter(monkeypatch, caps=NARROW_CAPS)
    assert runtime.capabilities(Harness.CODEX) is NARROW_CAPS
    with pytest.raises(TypeError):
        runtime.capabilities("codex")  # type: ignore[arg-type]


def test_fake_adapter_is_a_harness_adapter():
    assert issubclass(FakeAdapter, runtime.HarnessAdapter)
