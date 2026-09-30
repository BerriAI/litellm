"""Tests for litellm/harness/sync.py: the sync bridge over the async runtime."""

from __future__ import annotations

import asyncio
import threading

import pytest

from litellm.harness import sync
from litellm.harness.types import Approval, Done, Harness, State, Text
from tests.unit.harness.core_fakes import (
    FakeSandbox,
    install_adapter,
    script_approval,
)


@pytest.fixture
def sandbox(tmp_path) -> FakeSandbox:
    return FakeSandbox(str(tmp_path))


async def _call_run_in_loop(sandbox: FakeSandbox) -> None:
    sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)


def test_sync_run_from_plain_code(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    result = sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert result.text == "hello world"
    assert result.stop_reason == "done"


def test_sync_stream_from_plain_code(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    stream = sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, stream=True)
    events = list(stream)
    assert isinstance(events[-1], Done)
    assert [e.delta for e in events if isinstance(e, Text)] == ["hello ", "world"]
    assert stream.result is not None and stream.result.text == "hello world"
    assert list(stream) == []


def test_sync_stream_answers_approval(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch, script=script_approval)
    for event in sync.agent(
        Harness.CLAUDE_CODE, "hi", sandbox=sandbox, permissions="ask", stream=True
    ):
        if isinstance(event, Approval):
            event.allow()
    assert adapter_cls.instances[0].approvals[0][0] is True


def test_sync_stream_close_early(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch)
    with sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, stream=True) as stream:
        next(stream)
    assert adapter_cls.instances[0].calls[-1] == "stop"


def test_sync_validation_errors_raise_eagerly(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    with pytest.raises(TypeError, match=r"Harness\.OPENCODE"):
        sync.agent("opencode", "hi", sandbox=sandbox, stream=True)  # type: ignore[arg-type]


def test_sync_session_multi_turn_and_detach(monkeypatch, sandbox):
    adapter_cls = install_adapter(monkeypatch)
    with sync.agent_session(Harness.CLAUDE_CODE, sandbox=sandbox) as session:
        session.run("one")
        events = list(session.stream("two"))
        assert isinstance(events[-1], Done)
        assert session.cost == pytest.approx(0.5)
        assert len(session.history()) == 2
        state = session.detach()
    assert isinstance(state, State)
    with sync.agent_resume(state.dumps(), sandbox=sandbox) as resumed:
        assert resumed.run("three").text == "hello world"
    assert adapter_cls.instances[-1].resumed_with == "native-123"


def test_sync_session_stop_returns_state(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    session = sync.agent_session(Harness.CLAUDE_CODE, sandbox=sandbox).start()
    session.run("one")
    state = session.stop()
    assert state.native_session_id == "native-123"


def test_single_background_loop_thread(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    first = sync._LOOP.loop()
    sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    assert sync._LOOP.loop() is first
    names = [t.name for t in threading.enumerate()]
    assert names.count("litellm-harness-loop") == 1


async def test_run_inside_event_loop_raises(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    with pytest.raises(RuntimeError, match="aagent"):
        sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox)
    with pytest.raises(RuntimeError, match="aagent"):
        sync.agent(Harness.CLAUDE_CODE, "hi", sandbox=sandbox, stream=True)
    with pytest.raises(RuntimeError, match="aagent_session"):
        sync.agent_session(Harness.CLAUDE_CODE, sandbox=sandbox)


def test_run_inside_asyncio_run_raises(monkeypatch, sandbox):
    install_adapter(monkeypatch)
    with pytest.raises(RuntimeError, match="await litellm.aagent"):
        asyncio.run(_call_run_in_loop(sandbox))
