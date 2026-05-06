"""
Validation #11 — cascade delete + VM termination.

DELETE /v2/agents/{id} cancels active runs, terminates sessions, calls
provider.terminate.
"""

from litellm.proxy.agent_session_endpoints.constants import (
    SESSION_STATUS_ERROR,
    SESSION_STATUS_TERMINATED,
)


def test_delete_agent_terminates_sessions_and_calls_provider(
    client, noop_provider, fake_prisma_client
):
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    aid = agent["id"]

    sess_a = client.post(
        f"/v2/agents/{aid}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sess_b = client.post(
        f"/v2/agents/{aid}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()

    # Create a run on session A so cascade exercises run-cancel path too.
    run = client.post(
        f"/v2/sessions/{sess_a['id']}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()

    # DELETE the agent. Cascade: terminate both sessions, cancel the run.
    res = client.delete(f"/v2/agents/{aid}", headers={"Authorization": "Bearer k"})
    assert res.status_code == 200
    assert res.json()["deleted"] is True

    # Both sessions should be marked terminated in DB.
    sessions_in_db = fake_prisma_client.db.litellm_agentsession.rows
    for s in sessions_in_db:
        assert s.status == SESSION_STATUS_TERMINATED

    # provider.terminate was called once per session.
    terminate_session_ids = {c["session_id"] for c in noop_provider.terminate_calls}
    assert sess_a["id"] in terminate_session_ids
    assert sess_b["id"] in terminate_session_ids


def test_delete_agent_skips_session_already_in_terminal_status(
    client, noop_provider, fake_prisma_client
):
    """Greptile P3 (regression): the cascade filter must use
    ``status not in SESSION_TERMINAL_STATUSES`` rather than
    ``terminated_at is None``. A session whose status was flipped to
    ``error`` (terminal) but whose ``terminated_at`` was never written
    must NOT be re-terminated by the cascade — that would double-fire
    ``provider.terminate`` and confuse the audit trail.
    """
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    aid = agent["id"]

    # An "active" session that should still get terminated by the cascade.
    sess_active = client.post(
        f"/v2/agents/{aid}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    # A session whose status was already flipped to ``error`` (terminal)
    # but with ``terminated_at`` deliberately left None — the legacy
    # filter would have re-terminated this row, but the status-based
    # filter should skip it.
    sess_already_terminal = client.post(
        f"/v2/agents/{aid}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    terminal_row = next(
        r
        for r in fake_prisma_client.db.litellm_agentsession.rows
        if r.id == sess_already_terminal["id"]
    )
    terminal_row.status = SESSION_STATUS_ERROR
    terminal_row.terminated_at = None  # explicit — this is the bug surface

    # Reset terminate_calls so we can isolate what cascade fires.
    noop_provider.terminate_calls.clear()

    res = client.delete(f"/v2/agents/{aid}", headers={"Authorization": "Bearer k"})
    assert res.status_code == 200

    # Only the active session should have been routed through
    # provider.terminate; the already-terminal one must be skipped.
    terminate_session_ids = {c["session_id"] for c in noop_provider.terminate_calls}
    assert sess_active["id"] in terminate_session_ids
    assert (
        sess_already_terminal["id"] not in terminate_session_ids
    ), "cascade re-terminated a session that was already in a terminal status"

    # The already-terminal session's status must be unchanged.
    assert terminal_row.status == SESSION_STATUS_ERROR


def test_delete_session_terminates_active_runs(
    client, noop_provider, fake_prisma_client
):
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{agent["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sid = sess["id"]
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "x"}},
    ).json()
    rid = run["id"]

    res = client.delete(f"/v2/sessions/{sid}", headers={"Authorization": "Bearer k"})
    assert res.status_code == 200

    final_run = client.get(
        f"/v2/sessions/{sid}/runs/{rid}", headers={"Authorization": "Bearer k"}
    ).json()
    assert final_run["status"] == "cancelled"

    # provider.terminate hit.
    assert any(c["session_id"] == sid for c in noop_provider.terminate_calls)
