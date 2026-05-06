"""
Session status driver — flips ``ready`` <-> ``busy`` based on the runs table.

Pure-function logic lives in ``state_machine.derive_session_status_from_runs``.
This module is the I/O wrapper that reads the session row, counts active
runs, and persists the new status only when the helper says it should change.

Called from every code path that flips a run's status:

  * ``POST /v2/sessions/{sid}/runs``      (create — queued -> session busy)
  * ``POST /v2/sessions/{sid}/followup``  (create new run — same)
  * ``GET  /v2/sessions/{sid}/runs/next/internal/poll`` (claim — queued -> running)
  * ``POST /v2/sessions/{sid}/runs/{rid}/events:append`` (terminal -> ready)

Without these hooks, sessions that started ``ready`` would stay ``busy``
forever after the first run started — see Greptile P1 review on
PR #27330.
"""

from datetime import datetime, timezone

from litellm.proxy.agent_session_endpoints.constants import RUN_ACTIVE_STATUSES
from litellm.proxy.agent_session_endpoints.state_machine import (
    derive_session_status_from_runs,
)


async def refresh_session_status_from_runs(prisma_client, session_id: str) -> None:
    """Look up the session, decide if it should flip, persist if yes.

    Idempotent: if no transition is needed, no DB write happens.
    Best-effort: callers should treat failures as non-fatal — the
    cleanup sweeper is the safety net that catches sessions stuck in
    ``busy`` after the run rows have all moved to terminal.
    """
    session = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    if session is None:
        return
    active = await prisma_client.db.litellm_agentrun.find_first(
        where={
            "session_id": session_id,
            "status": {"in": list(RUN_ACTIVE_STATUSES)},
        }
    )
    new_status = derive_session_status_from_runs(
        current_session_status=session.status,
        has_active_run=active is not None,
    )
    if new_status is None:
        return
    await prisma_client.db.litellm_agentsession.update(
        where={"id": session_id},
        data={
            "status": new_status,
            "updated_at": datetime.now(timezone.utc),
        },
    )
