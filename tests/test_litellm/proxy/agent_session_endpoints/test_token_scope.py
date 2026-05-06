"""
Validation #12 — JWT scoping.

* Daemon JWT for session A cannot append events to session B (sub mismatch)
* Expired JWT rejected with 401
* Terminated session rejects its still-valid JWT (status check)
* Daemon JWT cannot call /v1/chat/completions (wrong scope) — covered by
  the dedicated daemon_token_auth dependency, since it's only mounted on
  internal endpoints.
"""

import time

import jwt as pyjwt
import pytest

from litellm.proxy.agent_session_endpoints.auth import (
    AGENT_RUNTIME_SCOPE,
    decode_daemon_token,
    hash_daemon_token,
    mint_daemon_token,
)
from litellm.proxy.agent_session_endpoints.constants import (
    AGENT_JWT_ALGORITHM,
)


def test_mint_and_decode_roundtrip():
    token = mint_daemon_token(
        session_id="sess_a",
        agent_id="agent_a",
        expires_at_epoch=int(time.time()) + 3600,
    )
    payload = decode_daemon_token(token)
    assert payload["sub"] == "sess_a"
    assert payload["agent_id"] == "agent_a"
    assert payload["scope"] == AGENT_RUNTIME_SCOPE


def test_token_for_session_a_rejected_by_session_b(client, noop_provider):
    # Create two sessions, get two daemon tokens.
    a = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "x", "model": "gpt-4"},
    ).json()
    sess_a = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": a["id"], "repos": []},
    ).json()
    sess_b = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": a["id"], "repos": []},
    ).json()

    # Use sess_a's daemon token to register sess_b.
    res = client.post(
        f"/v2/sessions/{sess_b['id']}/internal/register",
        headers={"Authorization": f"Bearer {sess_a['daemon_token']}"},
        json={"vm_id": "i-x"},
    )
    assert res.status_code == 403, res.text


def test_expired_jwt_rejected(client, noop_provider):
    a = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": a["id"], "repos": []},
    ).json()

    # Mint a token that expired 1 minute ago.
    expired = mint_daemon_token(
        session_id=sess["id"],
        agent_id=a["id"],
        expires_at_epoch=int(time.time()) - 60,
    )
    res = client.post(
        f"/v2/sessions/{sess['id']}/internal/register",
        headers={"Authorization": f"Bearer {expired}"},
        json={"vm_id": "i-x"},
    )
    assert res.status_code == 401
    assert "expired" in res.text.lower()


def test_terminated_session_rejects_its_token(
    client, noop_provider, fake_prisma_client
):
    a = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": a["id"], "repos": []},
    ).json()
    sid = sess["id"]
    daemon_token = sess["daemon_token"]

    # Delete the session.
    client.delete(f"/v2/sessions/{sid}", headers={"Authorization": "Bearer k"})

    # Token should now be rejected with 410 Gone.
    res = client.post(
        f"/v2/sessions/{sid}/internal/heartbeat",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-x"},
    )
    assert res.status_code == 410


def test_wrong_scope_rejected(client, noop_provider):
    """A token signed with the right secret but wrong scope is rejected."""
    import os

    secret = os.environ["LITELLM_AGENT_JWT_SECRET"]
    bad = pyjwt.encode(
        {
            "sub": "sess_x",
            "agent_id": "agent_x",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "scope": "user_api_key",  # NOT agent_runtime_internal
        },
        secret,
        algorithm=AGENT_JWT_ALGORITHM,
    )
    res = client.post(
        "/v2/sessions/sess_x/internal/register",
        headers={"Authorization": f"Bearer {bad}"},
        json={"vm_id": "i-x"},
    )
    assert res.status_code == 401


def test_token_hash_helper():
    h1 = hash_daemon_token("abc")
    h2 = hash_daemon_token("abc")
    h3 = hash_daemon_token("abd")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex
