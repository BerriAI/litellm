"""
Validation #17 — session toggles ``ready`` <-> ``busy`` based on runs.

A prior version of the endpoint layer defined
``derive_session_status_from_runs`` in ``state_machine.py`` and
unit-tested it, but never called it from any endpoint. After the first
run completed, every session stayed permanently ``busy`` regardless of
whether any active run existed — clients polling
``GET /v2/sessions/{id}`` would always see ``busy``.

The fix wires ``refresh_session_status_from_runs`` (in
``session_status.py``) into every endpoint that flips a run's status:
  * ``POST /v2/sessions/{sid}/runs``                 — queued -> session busy
  * ``POST /v2/sessions/{sid}/followup`` (new run)   — same
  * ``POST /v2/sessions/{sid}/runs/{rid}/cancel``    — terminal -> ready
  * ``GET  /v2/sessions/{sid}/runs/next/internal/poll`` — claim
  * ``POST /v2/sessions/{sid}/runs/{rid}/events:append`` — terminal -> ready

This file walks the SDK-visible state across the lifecycle and asserts
the session status flips correctly at each step.
"""


def _bootstrap(client):
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "osc", "model": "gpt-4"},
    ).json()
    sess = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": agent["id"], "repos": []},
    ).json()
    return sess["id"]


def _force_session_ready(fake_prisma_client, session_id):
    """Skip the daemon-register step. ``derive_session_status_from_runs``
    intentionally returns None for ``provisioning`` (see state_machine
    docstring), so we manually flip to ``ready`` to test the
    ``ready`` <-> ``busy`` oscillation directly."""
    for row in fake_prisma_client.db.litellm_agentsession.rows:
        if row.id == session_id:
            row.status = "ready"
            return
    raise AssertionError(f"session {session_id} not found in fake prisma")


def _get_status(client, session_id):
    return client.get(
        f"/v2/sessions/{session_id}",
        headers={"Authorization": "Bearer k"},
    ).json()["status"]


def test_session_flips_busy_when_run_created(client, fake_prisma_client, noop_provider):
    sid = _bootstrap(client)
    _force_session_ready(fake_prisma_client, sid)

    assert _get_status(client, sid) == "ready"

    client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hello"}},
    )
    assert _get_status(client, sid) == "busy"


def test_session_flips_ready_after_run_cancelled(
    client, fake_prisma_client, noop_provider
):
    sid = _bootstrap(client)
    _force_session_ready(fake_prisma_client, sid)

    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    assert _get_status(client, sid) == "busy"

    client.post(
        f"/v2/sessions/{sid}/runs/{run['id']}/cancel",
        headers={"Authorization": "Bearer k"},
    )
    assert _get_status(client, sid) == "ready"


def test_session_flips_ready_after_followup_run_cancelled(
    client, fake_prisma_client, noop_provider
):
    """Followup creating a new run should also drive the session to busy,
    and cancelling that run should drive it back to ready."""
    sid = _bootstrap(client)
    _force_session_ready(fake_prisma_client, sid)

    fr = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "from followup"}},
    ).json()
    assert _get_status(client, sid) == "busy"

    client.post(
        f"/v2/sessions/{sid}/runs/{fr['run_id']}/cancel",
        headers={"Authorization": "Bearer k"},
    )
    assert _get_status(client, sid) == "ready"


def test_refresh_helper_is_idempotent(fake_prisma_client):
    """Calling ``refresh_session_status_from_runs`` twice in a row with
    the same DB state must not produce conflicting writes — the helper
    is a no-op when no transition is needed."""
    import asyncio
    from datetime import datetime, timezone

    from litellm.proxy.agent_session_endpoints.session_status import (
        refresh_session_status_from_runs,
    )
    from tests.test_litellm.proxy.agent_session_endpoints.conftest import _Row

    fake_prisma_client.db.litellm_agentsession.rows.append(
        _Row(
            id="sess_idem",
            status="ready",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    async def _go():
        # No active runs -> derived status = "ready" -> no transition.
        await refresh_session_status_from_runs(fake_prisma_client, "sess_idem")
        await refresh_session_status_from_runs(fake_prisma_client, "sess_idem")

    asyncio.get_event_loop().run_until_complete(_go())
    row = next(
        r
        for r in fake_prisma_client.db.litellm_agentsession.rows
        if r.id == "sess_idem"
    )
    assert row.status == "ready"


def test_refresh_helper_returns_quietly_for_missing_session(fake_prisma_client):
    """If the session row was deleted between event handling and the
    refresh call, the helper must not raise."""
    import asyncio

    from litellm.proxy.agent_session_endpoints.session_status import (
        refresh_session_status_from_runs,
    )

    asyncio.get_event_loop().run_until_complete(
        refresh_session_status_from_runs(fake_prisma_client, "sess_does_not_exist")
    )
