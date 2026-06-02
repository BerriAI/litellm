"""
Validation #11 — cascade delete + VM termination.

DELETE /v2/agents/{id} cancels active runs, terminates sessions, calls
provider.terminate.
"""

from litellm.proxy.agent_session_endpoints.constants import (
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
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": aid, "repos": []},
    ).json()
    sess_b = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": aid, "repos": []},
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


def test_delete_session_terminates_active_runs(
    client, noop_provider, fake_prisma_client
):
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": agent["id"], "repos": []},
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
