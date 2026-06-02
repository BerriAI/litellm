"""
Validation #5 — run state machine.

queued -> running -> finished/cancelled/error.
Each transition emits a status event.
"""

from litellm.proxy.agent_session_endpoints.constants import (
    EVENT_TYPE_RUN_CANCELLED,
    EVENT_TYPE_RUN_ERROR,
    EVENT_TYPE_RUN_FINISHED,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_ERROR,
    RUN_STATUS_FINISHED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
)
from litellm.proxy.agent_session_endpoints.state_machine import (
    is_valid_run_transition,
    run_is_active,
    run_is_terminal,
)


def test_run_state_machine_pure():
    assert is_valid_run_transition("queued", "running")
    assert is_valid_run_transition("queued", "cancelled")
    assert is_valid_run_transition("queued", "error")
    assert is_valid_run_transition("running", "finished")
    assert is_valid_run_transition("running", "cancelled")
    assert is_valid_run_transition("running", "error")
    # No transitions out of terminal
    assert not is_valid_run_transition("finished", "running")
    assert not is_valid_run_transition("cancelled", "running")
    assert not is_valid_run_transition("error", "finished")


def test_run_helpers():
    assert run_is_active("queued")
    assert run_is_active("running")
    assert not run_is_active("finished")
    assert run_is_terminal("finished")
    assert run_is_terminal("cancelled")
    assert run_is_terminal("error")
    assert not run_is_terminal("queued")


def _bootstrap(client, noop_provider):
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
    return agent, sess


def test_run_starts_queued(client, noop_provider):
    _, sess = _bootstrap(client, noop_provider)
    daemon_token = sess["daemon_token"]
    sid = sess["id"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )

    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    assert run["status"] == RUN_STATUS_QUEUED


def test_run_finishes_via_events_append(client, noop_provider):
    _, sess = _bootstrap(client, noop_provider)
    daemon_token = sess["daemon_token"]
    sid = sess["id"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    rid = run["id"]

    # Daemon claims the run via long-poll (turns it running).
    poll = client.get(
        f"/v2/sessions/{sid}/runs/next/internal/poll",
        headers={"Authorization": f"Bearer {daemon_token}"},
    )
    assert poll.status_code == 200
    assert poll.json()["run_id"] == rid

    after_poll = client.get(
        f"/v2/sessions/{sid}/runs/{rid}", headers={"Authorization": "Bearer k"}
    ).json()
    assert after_poll["status"] == RUN_STATUS_RUNNING

    # Daemon emits run_finished — flips to finished.
    append = client.post(
        f"/v2/sessions/{sid}/runs/{rid}/events:append",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"event_type": EVENT_TYPE_RUN_FINISHED, "payload": {"result": "done"}},
    )
    assert append.status_code == 200

    final = client.get(
        f"/v2/sessions/{sid}/runs/{rid}", headers={"Authorization": "Bearer k"}
    ).json()
    assert final["status"] == RUN_STATUS_FINISHED
    assert final["result"] == "done"


def test_run_cancel_endpoint(client, noop_provider):
    _, sess = _bootstrap(client, noop_provider)
    sid = sess["id"]
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    rid = run["id"]

    cancel = client.post(
        f"/v2/sessions/{sid}/runs/{rid}/cancel",
        headers={"Authorization": "Bearer k"},
    )
    assert cancel.status_code == 200
    assert cancel.json()["status"] == RUN_STATUS_CANCELLED

    # Idempotent — cancelling again returns same status.
    cancel2 = client.post(
        f"/v2/sessions/{sid}/runs/{rid}/cancel",
        headers={"Authorization": "Bearer k"},
    )
    assert cancel2.status_code == 200
    assert cancel2.json()["status"] == RUN_STATUS_CANCELLED


def test_terminal_event_via_append_emits_status_change(client, noop_provider):
    """run_error event flips run status."""
    _, sess = _bootstrap(client, noop_provider)
    daemon_token = sess["daemon_token"]
    sid = sess["id"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "boom"}},
    ).json()
    rid = run["id"]
    client.get(
        f"/v2/sessions/{sid}/runs/next/internal/poll",
        headers={"Authorization": f"Bearer {daemon_token}"},
    )

    client.post(
        f"/v2/sessions/{sid}/runs/{rid}/events:append",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"event_type": EVENT_TYPE_RUN_ERROR, "payload": {"reason": "boom"}},
    )

    final = client.get(
        f"/v2/sessions/{sid}/runs/{rid}", headers={"Authorization": "Bearer k"}
    ).json()
    assert final["status"] == RUN_STATUS_ERROR
