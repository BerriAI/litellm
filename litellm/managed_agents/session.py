"""
Session — Python-side handle for one managed-agent session.

A ``Session`` is the unit of work in the managed-agents API. It holds the
sandbox + runtime that any prompts sent through it will use, and it owns
the persistence of runs and events to the DB.

Lifecycle:
  * Construction is via ``Session.from_db_row(row, db, runtime, sandbox,
    agent_config)``. The owning ``Agent`` calls this internally so SDK
    users normally don't.
  * ``await session.send(prompt)`` is the primary entry point: it
    INSERTs a ``LiteLLM_AgentRun`` row, drives the runtime, persists
    every event to ``LiteLLM_AgentRunEvent``, and yields each event back
    to the caller as the runtime emits it. The session and run statuses
    flip in lock-step (``ready`` -> ``busy`` -> ``ready`` for the
    session; ``queued`` -> ``running`` -> ``finished``/``error`` for the
    run) so polling clients see consistent state.
  * Other helpers (``get_run``, ``list_runs``, ``conversation``) wrap
    read-only queries — same shape the HTTP endpoints serve, no
    surprises.

Concurrency contract: one in-flight ``send()`` per session. The session
status flag enforces this in the DB (busy -> reject), but Python callers
should not call ``send()`` twice concurrently on the same instance — the
runtime and sandbox are not assumed to be concurrent-safe.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import prisma

from litellm.managed_agents.agent_runtime.base import (
    AgentConfig,
    AgentRuntime,
    SessionState,
)
from litellm.managed_agents.events import Event
from litellm.managed_agents.run import Run
from litellm.managed_agents.sandbox.base import Sandbox
from litellm.proxy.agent_session_endpoints.constants import (
    RUN_STATUS_ERROR,
    RUN_STATUS_FINISHED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    SESSION_STATUS_BUSY,
    SESSION_STATUS_READY,
)
from litellm.proxy.agent_session_endpoints.ids import new_run_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    """A live, in-process handle on one ``LiteLLM_AgentSession`` row.

    Holds runtime + sandbox + agent_config (a snapshot of the parent
    agent's runtime-relevant fields). The ``daemon_token`` field is set
    only at create-time and is the same JWT the proxy stores; it's
    surfaced here so callers that still want to talk to the proxy's
    HTTP endpoints (e.g. SSE event tail) have what they need.
    """

    id: str
    agent_id: str
    status: str
    runtime: AgentRuntime
    sandbox: Sandbox
    agent_config: AgentConfig
    db: Any
    repos: List[Dict[str, Any]] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    daemon_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def from_db_row(
        cls,
        row: Any,
        db: Any,
        runtime: Optional[AgentRuntime],
        sandbox: Optional[Sandbox],
        agent_config: AgentConfig,
        daemon_token: Optional[str] = None,
    ) -> "Session":
        if runtime is None or sandbox is None:
            raise RuntimeError(
                "Session requires both runtime and sandbox; the parent Agent "
                "must be constructed with both before spawning sessions."
            )
        return cls(
            id=getattr(row, "id"),
            agent_id=getattr(row, "agent_id"),
            status=getattr(row, "status", "unknown"),
            runtime=runtime,
            sandbox=sandbox,
            agent_config=agent_config,
            db=db,
            repos=_coerce_list_of_dict(getattr(row, "repos", None)),
            env_vars=_coerce_dict(getattr(row, "env_vars", None)),
            daemon_token=daemon_token,
        )

    # ------------------------------------------------------------------
    # The interesting one — send a prompt and yield events as they happen
    # ------------------------------------------------------------------

    async def send(self, prompt: str) -> AsyncIterator[Event]:
        """Drive one run end-to-end and yield events as the runtime emits them.

        Steps (mirrors what the HTTP create_run + event_stream pair does
        on the proxy):
          1. INSERT ``LiteLLM_AgentRun`` (status=``queued``, prompt={text}).
          2. UPDATE ``self.status = busy`` and the run row to ``running``.
          3. ``async for event in self.runtime.run(...)``:
             - INSERT ``LiteLLM_AgentRunEvent(run_id, seq, event_type, payload)``.
             - yield the event.
          4. On clean exit: UPDATE run -> ``finished``, session -> ``ready``.
          5. On exception: UPDATE run -> ``error``, session -> ``ready``,
             then re-raise. We always restore the session to ``ready``
             so a single failed send doesn't lock the session forever.

        The ``finally`` block restores session status even on early
        cancellation by the consumer (their generator close raises
        ``GeneratorExit``). That's important because async generators
        can be closed without ever reaching their tail.
        """
        run_id = await self._insert_run(prompt)

        await self._set_session_status(SESSION_STATUS_BUSY)
        await self._set_run_status(run_id, RUN_STATUS_RUNNING, started_at=_now())

        seq = 0
        terminal_status = RUN_STATUS_FINISHED
        terminal_result: Optional[str] = None

        session_state = SessionState(
            session_id=self.id,
            cwd=self.sandbox.cwd,
            env_vars=dict(self.env_vars),
            repos=list(self.repos),
        )

        try:
            async for event in self.runtime.run(
                prompt=prompt,
                sandbox=self.sandbox,
                session_state=session_state,
                agent_config=self.agent_config,
            ):
                seq += 1
                await self._insert_event(run_id, seq, event)

                # Track the LLM's last "result" so we can store it on the
                # run row — matches what the HTTP endpoint does when the
                # daemon reports a run_finished event.
                if event.type == "run_finished":
                    result = event.data.get("result")
                    if isinstance(result, str):
                        terminal_result = result

                yield event
        except Exception:
            terminal_status = RUN_STATUS_ERROR
            raise
        finally:
            # Always close out the run + restore the session, even if the
            # consumer cancelled early. The runtime is responsible for
            # any sandbox-side cleanup it owns.
            await self._set_run_status(
                run_id,
                terminal_status,
                terminated_at=_now(),
                result=terminal_result,
            )
            await self._set_session_status(SESSION_STATUS_READY)

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    async def get_run(self, run_id: str) -> Run:
        """Fetch a single run by id; raises ``LookupError`` if not found / not ours."""
        row = await self.db.litellm_agentrun.find_unique(where={"id": run_id})
        if row is None or getattr(row, "session_id", None) != self.id:
            raise LookupError(f"Run {run_id!r} not found for session {self.id!r}")
        return await Run.from_db_row(row, db=self.db)

    async def list_runs(self) -> List[Run]:
        """Return all runs in this session, newest first."""
        rows = await self.db.litellm_agentrun.find_many(
            where={"session_id": self.id},
            order={"created_at": "desc"},
        )
        return [await Run.from_db_row(row, db=self.db) for row in rows]

    async def conversation(self) -> List[Dict[str, Any]]:
        """Return every event across every run in this session, in order.

        Wire shape matches ``GET /v2/sessions/{sid}/conversation``: each
        item is ``{run_id, seq, event_type, payload, created_at}``. The
        order key is ``(created_at ASC, seq ASC)`` so events from
        concurrent runs interleave by wall-clock — the HTTP serializer
        sorts the same way.
        """
        runs = await self.db.litellm_agentrun.find_many(
            where={"session_id": self.id},
        )
        run_ids = [r.id for r in runs]
        if not run_ids:
            return []

        events = await self.db.litellm_agentrunevent.find_many(
            where={"run_id": {"in": run_ids}},
            order={"created_at": "asc"},
        )

        out: List[Dict[str, Any]] = []
        for ev in events:
            payload = getattr(ev, "payload", None)
            out.append(
                {
                    "run_id": getattr(ev, "run_id"),
                    "seq": getattr(ev, "seq"),
                    "event_type": getattr(ev, "event_type"),
                    "payload": payload if isinstance(payload, dict) else {},
                    "created_at": _iso(getattr(ev, "created_at", None)),
                }
            )
        return out

    # ------------------------------------------------------------------
    # DB write helpers — kept small + named so the send() flow reads
    # like a state machine instead of a wall of prisma calls.
    # ------------------------------------------------------------------

    async def _insert_run(self, prompt: str) -> str:
        """Create the LiteLLM_AgentRun row in queued state. Returns the run id.

        We wrap the prompt in ``{text: prompt}`` so the JSON column has
        a stable shape — the HTTP RunCreate schema does the same and
        downstream serializers expect a dict.
        """
        run_id = new_run_id()
        payload: Dict[str, Any] = {
            "id": run_id,
            "session": {"connect": {"id": self.id}},
            "status": RUN_STATUS_QUEUED,
            "prompt": prisma.Json({"text": prompt}),
            "updated_at": _now(),
        }
        await self.db.litellm_agentrun.create(data=payload)
        return run_id

    async def _set_run_status(
        self,
        run_id: str,
        status: str,
        started_at: Optional[datetime] = None,
        terminated_at: Optional[datetime] = None,
        result: Optional[str] = None,
    ) -> None:
        data: Dict[str, Any] = {"status": status, "updated_at": _now()}
        if started_at is not None:
            data["started_at"] = started_at
        if terminated_at is not None:
            data["terminated_at"] = terminated_at
        if result is not None:
            data["result"] = result
        await self.db.litellm_agentrun.update(where={"id": run_id}, data=data)

    async def _set_session_status(self, status: str) -> None:
        """Flip the session row's status. Also keeps a local mirror in sync."""
        await self.db.litellm_agentsession.update(
            where={"id": self.id},
            data={"status": status, "updated_at": _now()},
        )
        self.status = status

    async def _insert_event(self, run_id: str, seq: int, event: Event) -> None:
        await self.db.litellm_agentrunevent.create(
            data={
                "run_id": run_id,
                "seq": seq,
                "event_type": event.type,
                "payload": prisma.Json(event.to_payload()),
            }
        )


# ---------------------------------------------------------------------------
# Coercion helpers — same minimal shape as in ``agent.py``. Inlined here
# instead of pulling from a shared module to keep ``Session`` and ``Agent``
# independently importable.
# ---------------------------------------------------------------------------


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_list_of_dict(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
