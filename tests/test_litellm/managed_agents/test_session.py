"""
Unit tests for ``litellm.managed_agents.Session``.

Covers the ``send()`` happy path: events are persisted to
``LiteLLM_AgentRunEvent`` in seq order, and session/run statuses flip
in lock-step. Also covers the error path where the runtime raises and
we still restore the session to ``ready``.
"""

from typing import AsyncIterator

import pytest

from litellm.managed_agents.agent import Agent
from litellm.managed_agents.agent_runtime.base import (
    AgentConfig,
    AgentRuntime,
    SessionState,
)
from litellm.managed_agents.events import (
    EVENT_TYPE_ASSISTANT_MESSAGE,
    EVENT_TYPE_RUN_FINISHED,
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_TOOL_USE,
    Event,
)
from litellm.managed_agents.sandbox.local import LocalSandbox


class _ScriptedRuntime(AgentRuntime):
    """Yield a fixed list of events (or raise) so tests can assert wire shape."""

    def __init__(self, events=None, raises=None):
        self.events = events or []
        self.raises = raises
        self.runs_called = 0

    async def run(
        self, prompt, sandbox, session_state: SessionState, agent_config: AgentConfig
    ) -> AsyncIterator[Event]:
        self.runs_called += 1
        for ev in self.events:
            yield ev
        if self.raises is not None:
            raise self.raises


async def _make_session(fake_db, runtime):
    row = await fake_db.db.litellm_agent.create(
        data={
            "id": "agent_1",
            "name": "x",
            "model": "m",
            "user_api_key_hash": "h",
            "system_prompt": "sp",
        }
    )
    agent = await Agent.from_db_row(
        row, db=fake_db.db, runtime=runtime, sandbox=LocalSandbox()
    )
    return await agent.create_session()


@pytest.mark.asyncio
async def test_send_persists_events_and_yields_them(fake_db):
    events = [
        Event(type=EVENT_TYPE_ASSISTANT_MESSAGE, data={"content": "hello"}),
        Event(
            type=EVENT_TYPE_TOOL_USE,
            data={"tool_use_id": "t1", "tool": "Bash", "input": {"command": "ls"}},
        ),
        Event(
            type=EVENT_TYPE_TOOL_RESULT,
            data={"tool_use_id": "t1", "output": "ok", "is_error": False},
        ),
        Event(
            type=EVENT_TYPE_RUN_FINISHED,
            data={"result": "done", "is_error": False, "stop_reason": "stop"},
        ),
    ]
    session = await _make_session(fake_db, _ScriptedRuntime(events=events))

    yielded = []
    async for event in session.send("do the thing"):
        yielded.append(event)
    assert [e.type for e in yielded] == [e.type for e in events]

    # Persisted row count == number of events; seq is monotonic.
    persisted = await fake_db.db.litellm_agentrunevent.find_many(where={})
    persisted.sort(key=lambda r: r.seq)
    assert [r.event_type for r in persisted] == [e.type for e in events]
    assert [r.seq for r in persisted] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_send_flips_session_and_run_status(fake_db):
    session = await _make_session(
        fake_db,
        _ScriptedRuntime(
            events=[
                Event(
                    type=EVENT_TYPE_RUN_FINISHED,
                    data={"result": "ok"},
                )
            ]
        ),
    )

    # ready before
    s_pre = await fake_db.db.litellm_agentsession.find_unique(where={"id": session.id})
    assert s_pre.status == "ready"

    # drain
    async for _ in session.send("hi"):
        pass

    # ready after (busy was visible mid-flight; we settle it back).
    s_post = await fake_db.db.litellm_agentsession.find_unique(where={"id": session.id})
    assert s_post.status == "ready"
    assert session.status == "ready"

    # Run row terminal.
    runs = await fake_db.db.litellm_agentrun.find_many(where={"session_id": session.id})
    assert len(runs) == 1
    assert runs[0].status == "finished"
    assert runs[0].result == "ok"
    assert runs[0].terminated_at is not None
    assert runs[0].started_at is not None


@pytest.mark.asyncio
async def test_send_restores_session_on_runtime_error(fake_db):
    session = await _make_session(
        fake_db,
        _ScriptedRuntime(events=[], raises=RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in session.send("hi"):
            pass

    # session bounced back to ready even though we raised
    s_post = await fake_db.db.litellm_agentsession.find_unique(where={"id": session.id})
    assert s_post.status == "ready"
    runs = await fake_db.db.litellm_agentrun.find_many(where={"session_id": session.id})
    assert len(runs) == 1
    assert runs[0].status == "error"


@pytest.mark.asyncio
async def test_get_run_returns_run(fake_db):
    session = await _make_session(
        fake_db,
        _ScriptedRuntime(
            events=[Event(type=EVENT_TYPE_RUN_FINISHED, data={"result": "ok"})]
        ),
    )
    async for _ in session.send("hi"):
        pass
    runs = await session.list_runs()
    assert len(runs) == 1
    fetched = await session.get_run(runs[0].id)
    assert fetched.id == runs[0].id
    assert fetched.status == "finished"


@pytest.mark.asyncio
async def test_conversation_includes_persisted_events(fake_db):
    session = await _make_session(
        fake_db,
        _ScriptedRuntime(
            events=[
                Event(type=EVENT_TYPE_ASSISTANT_MESSAGE, data={"content": "hi"}),
                Event(type=EVENT_TYPE_RUN_FINISHED, data={"result": "ok"}),
            ]
        ),
    )
    async for _ in session.send("hello"):
        pass
    convo = await session.conversation()
    assert [m["event_type"] for m in convo] == [
        EVENT_TYPE_ASSISTANT_MESSAGE,
        EVENT_TYPE_RUN_FINISHED,
    ]
    assert convo[0]["payload"] == {"content": "hi"}
