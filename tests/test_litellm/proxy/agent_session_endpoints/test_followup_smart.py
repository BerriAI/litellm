"""
Validation #6 — followup smart behavior.

Case 1: latest run is active -> followup adds user_message event to it.
Case 2: latest run is terminal (or no runs exist) -> followup creates NEW run.
"""

from litellm.proxy.agent_session_endpoints.constants import (
    EVENT_TYPE_RUN_FINISHED,
    EVENT_TYPE_USER_MESSAGE,
)


def _bootstrap_ready(client, noop_provider):
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
    daemon_token = sess["daemon_token"]
    sid = sess["id"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )
    return sid, daemon_token


def test_followup_no_runs_creates_new_run(client, noop_provider):
    sid, _ = _bootstrap_ready(client, noop_provider)

    res = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "first message"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["action"] == "new_run"
    assert body["run_id"]


def test_followup_with_active_run_appends_user_message(
    client, noop_provider, fake_prisma_client
):
    sid, daemon_token = _bootstrap_ready(client, noop_provider)
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    rid = run["id"]

    res = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "by the way..."}},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["action"] == "queued"
    assert body["run_id"] == rid

    # The user_message event should be appended to the active run.
    events = fake_prisma_client.db.litellm_agentrunevent.rows
    user_messages = [
        e for e in events if e.event_type == EVENT_TYPE_USER_MESSAGE and e.run_id == rid
    ]
    assert len(user_messages) == 1
    assert user_messages[0].payload == {"text": "by the way..."}


def test_followup_after_run_finishes_creates_new_run(
    client, noop_provider, fake_prisma_client
):
    sid, daemon_token = _bootstrap_ready(client, noop_provider)
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "first"}},
    ).json()
    rid = run["id"]

    # Finish the run so it goes terminal.
    client.get(
        f"/v2/sessions/{sid}/runs/next/internal/poll",
        headers={"Authorization": f"Bearer {daemon_token}"},
    )
    client.post(
        f"/v2/sessions/{sid}/runs/{rid}/events:append",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"event_type": EVENT_TYPE_RUN_FINISHED, "payload": {"result": "ok"}},
    )

    res = client.post(
        f"/v2/sessions/{sid}/followup",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "second turn"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["action"] == "new_run"
    assert body["run_id"] != rid
