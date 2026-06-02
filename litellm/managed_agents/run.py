"""
Run — read-only Python accessor for one ``LiteLLM_AgentRun`` row.

This is the SDK-side counterpart to the ``GET /v2/sessions/{sid}/runs/{rid}``
HTTP endpoint. The proxy persists every run to ``LiteLLM_AgentRun`` and every
event to ``LiteLLM_AgentRunEvent``; ``Run`` is the typed accessor in-process
code uses to read those rows back.

Lifecycle notes:
  * Construction does not touch the DB. Use ``Run.from_db_row(...)`` to
    build an instance from a Prisma row.
  * ``stream(starting_seq=N)`` is an async iterator over events ordered by
    ``seq``; pass ``N>0`` to resume from where you stopped. Iteration is
    snapshot-based — it yields events that are persisted at the time of
    the call and stops. Live tailing (waiting for new events as the run
    progresses) is the SSE endpoint's job, not this class.
  * Read-only by design. ``Run`` never writes to the DB. The owning
    ``Session`` writes; ``Run`` only reads.
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from litellm.managed_agents.events import Event


# Default page size when pulling events from the DB. Picked to keep memory
# footprint modest on chatty runs (a busy run can have thousands of events).
_DEFAULT_EVENT_PAGE_SIZE = 200


@dataclass
class Run:
    """Read-only view of one ``LiteLLM_AgentRun`` row.

    The ``db`` reference is the Prisma client — kept here so ``stream()``
    can fetch events without callers having to wire the client through.
    Tests and offline tools that don't need ``stream()`` can pass
    ``db=None`` and just consume the static fields.
    """

    id: str
    session_id: str
    status: str
    prompt: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    parent_run_id: Optional[str] = None
    db: Any = None

    @classmethod
    async def from_db_row(cls, row: Any, db: Any) -> "Run":
        """Build a ``Run`` from a Prisma ``LiteLLM_AgentRun`` row.

        Defensive on field access so partial selects (e.g. via Prisma's
        ``select=...`` parameter) don't blow up — missing fields just
        default. The caller is responsible for fetching all the columns
        it needs for whatever it plans to do with the ``Run``.
        """
        prompt = _coerce_dict(getattr(row, "prompt", None))
        return cls(
            id=getattr(row, "id"),
            session_id=getattr(row, "session_id"),
            status=getattr(row, "status", "unknown"),
            prompt=prompt,
            result=getattr(row, "result", None),
            parent_run_id=getattr(row, "parent_run_id", None),
            db=db,
        )

    async def stream(
        self,
        starting_seq: int = 0,
    ) -> AsyncIterator[Event]:
        """Yield persisted events in ``seq`` order, starting from ``starting_seq``.

        ``starting_seq`` is exclusive of itself when passed (matches the
        SSE endpoint contract: client passes the last seq it saw, server
        replays everything strictly greater). Pass ``0`` for "from the
        beginning".

        Iterates in batches to keep peak memory bounded — ``LiteLLM_AgentRunEvent``
        is wide (JSON payload) and a busy run can have thousands of rows.
        Stops as soon as the DB returns fewer rows than the page size,
        which is the canonical "no more pages" signal in cursor-style
        pagination.
        """
        if self.db is None:
            raise RuntimeError(
                "Run.stream() requires a Prisma client; construct with db=<client> "
                "or via Run.from_db_row(row, db=<client>)."
            )

        cursor_seq = int(starting_seq)
        while True:
            rows = await self.db.litellm_agentrunevent.find_many(
                where={
                    "run_id": self.id,
                    "seq": {"gt": cursor_seq},
                },
                order={"seq": "asc"},
                take=_DEFAULT_EVENT_PAGE_SIZE,
            )
            if not rows:
                return

            for row in rows:
                yield _event_from_row(row)
                cursor_seq = max(cursor_seq, int(getattr(row, "seq", cursor_seq)))

            # Short-circuit when the page wasn't full — no more rows exist.
            if len(rows) < _DEFAULT_EVENT_PAGE_SIZE:
                return


def _coerce_dict(value: Any) -> Dict[str, Any]:
    """Prisma JSON columns come back as dicts; pass strings/None through."""
    if isinstance(value, dict):
        return value
    return {}


def _event_from_row(row: Any) -> Event:
    """Translate a ``LiteLLM_AgentRunEvent`` row into our ``Event`` dataclass."""
    payload = getattr(row, "payload", None)
    if not isinstance(payload, dict):
        payload = {}
    return Event(
        type=getattr(row, "event_type", "unknown"),
        data=dict(payload),
    )
