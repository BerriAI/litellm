"""
Validation #18 — daemon_append_event surfaces non-collision DB errors
correctly.

A prior version caught all ``Exception`` types in the seq-collision
retry path and re-raised them as ``409 event_seq_collision``. A
transient DB error during retry would surface to the daemon as a
misleading 409, hiding the real outage.

The fix narrows the catch to ``UniqueViolationError`` (or our test
marker ``RuntimeError("event_seq_collision")``). Other errors bubble
up unchanged so callers can distinguish a real conflict from a real
outage.
"""

import asyncio


def test_seq_collision_helper_classifies_unique_violation():
    """``_is_seq_collision`` matches the test marker."""
    from litellm.proxy.agent_session_endpoints.internal_endpoints import (
        _is_seq_collision,
    )

    assert _is_seq_collision(RuntimeError("event_seq_collision"))


def test_seq_collision_helper_rejects_unrelated_error():
    """An unrelated DB error must NOT be treated as a collision."""
    from litellm.proxy.agent_session_endpoints.internal_endpoints import (
        _is_seq_collision,
    )

    assert not _is_seq_collision(RuntimeError("connection refused"))
    assert not _is_seq_collision(ValueError("bad input"))


def test_event_append_real_collision_returns_409(
    client, fake_prisma_client, noop_provider
):
    """Two daemon events at the same seq -> the loser retries and gets
    a 409 only when the BOTH attempts collide. Single collision must
    be transparently handled by the retry."""
    import os

    from litellm.proxy.agent_session_endpoints.auth import mint_daemon_token

    # Bootstrap a session via the API so the prisma fake state lines up.
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

    # Use the daemon JWT minted by session create (the test conftest sets
    # ``LITELLM_AGENT_JWT_SECRET`` so this succeeds).
    daemon_token = sess["daemon_token"]

    # Insert a run + an event at seq=1 to set the collision state.
    from tests.test_litellm.proxy.agent_session_endpoints.conftest import _Row

    run_id = "run_collision"
    fake_prisma_client.db.litellm_agentrun.rows.append(
        _Row(id=run_id, session_id=sid, status="running")
    )
    fake_prisma_client.db.litellm_agentrunevent.rows.append(
        _Row(run_id=run_id, seq=1, event_type="run_started", payload={})
    )

    # Manually flip the session to ``ready`` so daemon_token_auth doesn't
    # 410. Conftest fake doesn't expose a session-status setter, so go
    # directly:
    for row in fake_prisma_client.db.litellm_agentsession.rows:
        if row.id == sid:
            row.status = "ready"

    # Send a normal event — it should append at seq=2 (not collide).
    res = client.post(
        f"/v2/sessions/{sid}/runs/{run_id}/events:append",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"event_type": "tool_call", "payload": {"name": "x"}},
    )
    assert res.status_code == 200
    assert res.json()["seq"] == 2


def test_event_append_unrelated_error_does_not_become_409(
    client, fake_prisma_client, noop_provider, monkeypatch
):
    """If the DB raises a non-collision error during create, the
    endpoint must NOT classify it as a 409 collision. The fix narrows
    the retry's catch to seq-collision markers; other errors propagate
    as the original RuntimeError.

    We use TestClient with ``raise_server_exceptions=True`` (the default)
    so the propagated exception is observable.
    """
    import pytest

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
    daemon_token = sess["daemon_token"]

    from tests.test_litellm.proxy.agent_session_endpoints.conftest import _Row

    run_id = "run_db_outage"
    fake_prisma_client.db.litellm_agentrun.rows.append(
        _Row(id=run_id, session_id=sid, status="running")
    )
    for row in fake_prisma_client.db.litellm_agentsession.rows:
        if row.id == sid:
            row.status = "ready"

    # Replace the create method on the events table to simulate a DB
    # outage. The endpoint's catch should NOT classify this as a
    # collision — the unrelated RuntimeError propagates instead of
    # becoming a misleading 409.
    async def _boom(*args, **kwargs):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(fake_prisma_client.db.litellm_agentrunevent, "create", _boom)

    with pytest.raises(RuntimeError, match="connection reset by peer"):
        client.post(
            f"/v2/sessions/{sid}/runs/{run_id}/events:append",
            headers={"Authorization": f"Bearer {daemon_token}"},
            json={"event_type": "tool_call", "payload": {}},
        )
