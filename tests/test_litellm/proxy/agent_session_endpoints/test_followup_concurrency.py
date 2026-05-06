"""
Validation #16 — `/followup` does not bypass the run-busy guard.

A prior version of ``followup`` skipped the ``_has_active_run`` check
when ``latest_run`` was terminal-or-absent and went straight to
``litellm_agentrun.create``. Two concurrent ``POST /followup`` requests
on an idle session both passed the ``latest_run.status in
RUN_ACTIVE_STATUSES`` check and both inserted runs, breaking the
"one active run per session" invariant that ``POST /runs`` enforces
via 409 ``run_busy``.

This test reproduces the race deterministically by setting up a session
that already has an active run, then sending /followup with a fresh
request — followup must return 409 ``run_busy`` rather than enqueue a
duplicate run.
"""


def _bootstrap_session(client):
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "concurrent", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{agent["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    return sess["id"]


def test_followup_returns_409_when_active_run_is_not_the_latest(
    client, fake_prisma_client, noop_provider
):
    """Reproducer for the original bug.

    Scenario: a session has both
      * an OLDER run still in ``running`` status, and
      * a NEWER run already in a terminal status (``finished``).

    The buggy code path used ``latest_run`` (newest by ``created_at``) to
    decide whether to fall through to the "create new run" branch. Since
    the newest run is terminal, the buggy version skipped ``_has_active_run``
    and inserted a duplicate run — breaking the one-active-run invariant.

    Two concurrent /followup calls trigger this same race in production
    even when there's a single run: both observe the same terminal-or-absent
    ``latest_run`` and both reach the create branch. We can't easily race
    asyncio in a unit test, so we model the equivalent state directly: an
    older active run + newer terminal run. With the fix in place, the
    busy check catches the older active run and returns 409.
    """
    from datetime import datetime, timedelta, timezone

    sid = _bootstrap_session(client)

    earlier = datetime.now(timezone.utc) - timedelta(minutes=10)
    later = datetime.now(timezone.utc)

    # Inject directly into the fake prisma rows so we control created_at
    # exactly. Bypassing ``.create()`` keeps the ``_now()``-based default
    # from collapsing the timestamps.
    from tests.test_litellm.proxy.agent_session_endpoints.conftest import _Row

    fake_prisma_client.db.litellm_agentrun.rows.append(
        _Row(
            id="run_old_active",
            session_id=sid,
            status="running",
            prompt={"text": "old"},
            created_at=earlier,
            updated_at=earlier,
        )
    )
    fake_prisma_client.db.litellm_agentrun.rows.append(
        _Row(
            id="run_newer_terminal",
            session_id=sid,
            status="finished",
            prompt={"text": "newer"},
            created_at=later,
            updated_at=later,
        )
    )

    res = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "should be blocked"}},
    )
    assert res.status_code == 409
    # Error envelope shape: {"error": {"code", "message", ...}}
    assert "run_busy" in res.json()["error"]["message"]


def test_followup_normal_path_creates_run_when_no_active_runs(client, noop_provider):
    """The new busy check must not break the happy path: when there is
    no active run, /followup still creates a fresh queued run."""
    sid = _bootstrap_session(client)
    res = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "first"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["action"] == "new_run"
    assert body["run_id"]


def test_followup_appends_event_when_run_is_active(client, noop_provider):
    """Other happy path: latest run IS active — /followup injects a
    user_message event rather than a new run. Unchanged by the new
    busy guard (we only fall through to the create branch when the
    latest run is terminal/absent)."""
    sid = _bootstrap_session(client)
    # First followup creates a queued run.
    first = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "first"}},
    ).json()
    # Second followup should append an event onto the now-active run.
    second = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "appended"}},
    ).json()
    assert second["action"] == "queued"
    assert second["run_id"] == first["run_id"]
