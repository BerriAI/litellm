"""
Validation #9 — idempotency.

POST /sessions twice with same Idempotency-Key  -> same session_id.
POST /runs twice with same Idempotency-Key      -> same run_id, count == 1.
"""


def _create_agent(client) -> str:
    return client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()["id"]


def test_session_idempotency(client, noop_provider, fake_prisma_client):
    agent_id = _create_agent(client)

    headers = {"Authorization": "Bearer k", "Idempotency-Key": "uuid-A"}
    a = client.post(
        f"/v2/agents/{agent_id}/sessions",
        headers=headers,
        json={"repos": []},
    )
    b = client.post(
        f"/v2/agents/{agent_id}/sessions",
        headers=headers,
        json={"repos": []},
    )
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]
    # Only one row in the table.
    assert len(fake_prisma_client.db.litellm_agentsession.rows) == 1


def test_run_idempotency(client, noop_provider, fake_prisma_client):
    agent_id = _create_agent(client)
    sess = client.post(
        f"/v2/agents/{agent_id}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sid = sess["id"]
    daemon_token = sess["daemon_token"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )

    headers = {"Authorization": "Bearer k", "Idempotency-Key": "run-key-1"}
    a = client.post(
        f"/v2/sessions/{sid}/runs",
        headers=headers,
        json={"prompt": {"text": "x"}},
    )
    b = client.post(
        f"/v2/sessions/{sid}/runs",
        headers=headers,
        json={"prompt": {"text": "x"}},
    )
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]
    assert len(fake_prisma_client.db.litellm_agentrun.rows) == 1
