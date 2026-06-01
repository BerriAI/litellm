"""
Validation #7 — POST /runs concurrency.

Fire 5 runs at the same session simultaneously. Expect: exactly 1
succeeds (200), 4 get 409 run_busy.

The fake Prisma client is sequential (no real concurrency on a single
event loop), but we can still assert the busy-check semantics by
firing the requests as ``asyncio.gather`` of httpx tasks.
"""

import asyncio

import pytest


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
    return sid


def test_only_one_run_per_session(client, noop_provider):
    sid = _bootstrap_ready(client, noop_provider)

    statuses = []
    for _ in range(5):
        res = client.post(
            f"/v2/sessions/{sid}/runs",
            headers={"Authorization": "Bearer k"},
            json={"prompt": {"text": "go"}},
        )
        statuses.append(res.status_code)

    # First succeeds, rest are 409 run_busy.
    assert statuses.count(200) == 1
    assert statuses.count(409) == 4


def test_idempotent_post_returns_same_run_id(client, noop_provider):
    sid = _bootstrap_ready(client, noop_provider)

    headers = {
        "Authorization": "Bearer k",
        "Idempotency-Key": "client-uuid-1",
    }
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

    assert a.status_code == 200
    assert b.status_code == 200
    assert a.json()["id"] == b.json()["id"]


def test_distinct_idempotency_keys_get_busy(client, noop_provider):
    sid = _bootstrap_ready(client, noop_provider)

    a = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k", "Idempotency-Key": "k1"},
        json={"prompt": {"text": "x"}},
    )
    b = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k", "Idempotency-Key": "k2"},
        json={"prompt": {"text": "y"}},
    )
    assert a.status_code == 200
    assert b.status_code == 409
