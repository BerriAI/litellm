"""
Validation #3 — one agent, multiple sessions; agent_id is stable across
sessions.
"""


def _create_agent(client) -> str:
    res = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "test", "model": "gpt-4"},
    )
    assert res.status_code == 200, res.text
    return res.json()["id"]


def _create_session(client, agent_id: str) -> str:
    res = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": agent_id, "repos": []},
    )
    assert res.status_code == 200, res.text
    return res.json()["id"]


def test_agent_reused_across_three_sessions(client, noop_provider):
    agent_id = _create_agent(client)

    session_ids = [_create_session(client, agent_id) for _ in range(3)]

    # Sessions are distinct IDs but all reference the same agent_id.
    assert len(set(session_ids)) == 3
    for sid in session_ids:
        get_res = client.get(
            f"/v2/sessions/{sid}", headers={"Authorization": "Bearer k"}
        )
        assert get_res.status_code == 200
        assert get_res.json()["agent_id"] == agent_id


def test_agent_id_format(client, noop_provider):
    agent_id = _create_agent(client)
    assert agent_id.startswith("agent_")


def test_sessions_under_agent_listed_by_filter(client, noop_provider):
    agent_id = _create_agent(client)
    sids = [_create_session(client, agent_id) for _ in range(2)]

    res = client.get(
        f"/v2/sessions?agent_id={agent_id}",
        headers={"Authorization": "Bearer k"},
    )
    assert res.status_code == 200
    listed_ids = {s["id"] for s in res.json()["data"]}
    assert listed_ids == set(sids)
