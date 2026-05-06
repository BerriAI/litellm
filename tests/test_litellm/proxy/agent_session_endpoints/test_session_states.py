"""
Validation #4 — session state machine.

Drives:
  * provisioning -> ready (via daemon register)
  * ready -> busy (via run create) -> ready (via run finish event)
  * provisioning -> error (provisioning failure)
  * ready -> error (daemon goes silent for 90s — covered by sweeper test)
"""

import asyncio

import pytest

from litellm.proxy.agent_session_endpoints.auth import mint_daemon_token
from litellm.proxy.agent_session_endpoints.constants import (
    SESSION_STATUS_ERROR,
    SESSION_STATUS_PROVISIONING,
    SESSION_STATUS_READY,
)
from litellm.proxy.agent_session_endpoints.state_machine import (
    derive_session_status_from_runs,
    is_valid_session_transition,
    session_can_accept_runs,
)


def test_state_machine_transitions_pure():
    # provisioning -> ready, error, terminated
    assert is_valid_session_transition("provisioning", "ready")
    assert is_valid_session_transition("provisioning", "error")
    assert is_valid_session_transition("provisioning", "terminated")
    # ready -> busy
    assert is_valid_session_transition("ready", "busy")
    # busy -> ready
    assert is_valid_session_transition("busy", "ready")
    # terminated is a sink
    assert not is_valid_session_transition("terminated", "ready")
    assert not is_valid_session_transition("terminated", "busy")


def test_session_can_accept_runs():
    assert session_can_accept_runs("ready")
    assert session_can_accept_runs("busy")
    assert not session_can_accept_runs("provisioning")
    assert not session_can_accept_runs("error")
    assert not session_can_accept_runs("terminated")


def test_derive_session_status_from_runs():
    # provisioning is gated until daemon registers
    assert derive_session_status_from_runs("provisioning", True) is None
    # ready -> busy when there's an active run
    assert derive_session_status_from_runs("ready", True) == "busy"
    # busy -> ready when no active runs
    assert derive_session_status_from_runs("busy", False) == "ready"
    # terminal sessions never transition
    assert derive_session_status_from_runs("terminated", False) is None
    assert derive_session_status_from_runs("error", True) is None
    # No-op when target == current
    assert derive_session_status_from_runs("ready", False) is None
    assert derive_session_status_from_runs("busy", True) is None


def test_session_starts_provisioning(client, noop_provider):
    res = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    )
    agent_id = res.json()["id"]

    res = client.post(
        f"/v2/agents/{agent_id}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == SESSION_STATUS_PROVISIONING
    assert body["daemon_token"]


def test_daemon_register_flips_to_ready(client, noop_provider, fake_prisma_client):
    res = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    )
    agent_id = res.json()["id"]

    res = client.post(
        f"/v2/agents/{agent_id}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    )
    body = res.json()
    sid = body["id"]
    daemon_token = body["daemon_token"]

    # Daemon registers — endpoint requires its own JWT.
    reg = client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )
    assert reg.status_code == 200, reg.text
    assert reg.json()["status"] == SESSION_STATUS_READY

    after = client.get(f"/v2/sessions/{sid}", headers={"Authorization": "Bearer k"})
    assert after.json()["status"] == SESSION_STATUS_READY
