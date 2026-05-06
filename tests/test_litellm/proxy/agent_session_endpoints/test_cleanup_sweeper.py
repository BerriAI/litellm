"""
Validation #13 — cleanup sweeper.

Drives:
  * Force ``expires_at`` past + status=ready → sweeper marks terminated, calls provider.terminate
  * Force ``last_heartbeat_at`` > 90s ago → sweeper marks error
  * Force ``status=running, updated_at`` > idle timeout → sweeper marks run error
"""

from datetime import datetime, timedelta, timezone

import pytest

from litellm.proxy.agent_session_endpoints.cleanup import run_cleanup_pass
from litellm.proxy.agent_session_endpoints.constants import (
    DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS,
    RUN_IDLE_TIMEOUT_SECONDS,
    RUN_STATUS_ERROR,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_READY,
    SESSION_STATUS_TERMINATED,
)


@pytest.mark.asyncio
async def test_sweeper_terminates_expired_sessions(
    client, noop_provider, fake_prisma_client
):
    a = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{a["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sid = sess["id"]

    # Force expiry into the past.
    row = fake_prisma_client.db.litellm_agentsession.rows[0]
    row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    row.status = SESSION_STATUS_READY

    summary = await run_cleanup_pass(fake_prisma_client)
    assert summary["expired_sessions"] == 1
    assert row.status == SESSION_STATUS_TERMINATED

    # provider.terminate was called.
    assert any(c["session_id"] == sid for c in noop_provider.terminate_calls)


@pytest.mark.asyncio
async def test_sweeper_marks_dead_daemon_sessions_error(
    client, noop_provider, fake_prisma_client
):
    a = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{a["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sid = sess["id"]

    row = fake_prisma_client.db.litellm_agentsession.rows[0]
    row.status = SESSION_STATUS_READY
    row.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(
        seconds=DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS + 30
    )

    summary = await run_cleanup_pass(fake_prisma_client)
    assert summary["dead_daemon_sessions"] == 1
    assert row.status == SESSION_STATUS_ERROR
    # Greptile P1: dead-daemon sweep MUST go through
    # ``_terminate_session_internal`` so the VM provider is notified.
    # Otherwise EC2 instances orphan once a real provider replaces
    # ``NoopVMProvider``.
    assert any(
        c["session_id"] == sid for c in noop_provider.terminate_calls
    ), "provider.terminate was never called for the dead-daemon session"


@pytest.mark.asyncio
async def test_sweeper_marks_stuck_runs_error(
    client, noop_provider, fake_prisma_client
):
    a = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{a["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sid = sess["id"]
    client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "x"}},
    ).json()

    # Force the run into stuck state, and put the session into ``busy``
    # (which is what the run-create hook would have done, modulo
    # provisioning).
    run_row = fake_prisma_client.db.litellm_agentrun.rows[0]
    run_row.status = "running"
    run_row.updated_at = datetime.now(timezone.utc) - timedelta(
        seconds=RUN_IDLE_TIMEOUT_SECONDS + 60
    )
    session_row = next(
        r for r in fake_prisma_client.db.litellm_agentsession.rows if r.id == sid
    )
    session_row.status = "busy"

    summary = await run_cleanup_pass(fake_prisma_client)
    assert summary["stuck_runs"] == 1
    assert run_row.status == RUN_STATUS_ERROR
    # Greptile P1: after the sweeper reaps a stuck run, the session must
    # transition ``busy`` -> ``ready`` so SDK consumers don't see it
    # stuck on ``busy`` forever.
    assert session_row.status == SESSION_STATUS_READY


@pytest.mark.asyncio
async def test_sweeper_pass_with_nothing_to_do(
    client, noop_provider, fake_prisma_client
):
    """Empty-DB pass returns all-zero summary, no exception."""
    summary = await run_cleanup_pass(fake_prisma_client)
    assert summary == {
        "expired_sessions": 0,
        "dead_daemon_sessions": 0,
        "stuck_runs": 0,
    }
