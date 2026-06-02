"""
Validation #14 — view-only admin cannot mutate any /v2/agents or
/v2/sessions resource.

The ``PROXY_ADMIN_VIEW_ONLY`` role is intended to grant cross-tenant
READ access (e.g. for a support UI) without granting write access. A
prior version of ``ownership.is_proxy_admin`` returned True for both
``PROXY_ADMIN`` and ``PROXY_ADMIN_VIEW_ONLY``, which let view-only
admins bypass the per-tenant ownership assertion on every write
endpoint and create / update / delete other tenants' rows.

Every state-mutating endpoint on the four routers (POST/PUT/PATCH/DELETE)
must call :func:`assert_caller_can_mutate` and return 403 for the
view-only admin role. Read endpoints (GET) must continue to work — that's
the whole point of the role.
"""


def _create_tenant_a_resources(client):
    """Bootstrap an agent + session + run owned by tenant A so we have
    something for the view-only admin to try (and fail) to mutate."""
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "tenant-a-agent", "model": "gpt-4"},
    ).json()
    session = client.post(
        f"/v2/agents/{agent["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    run = client.post(
        f"/v2/sessions/{session['id']}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    return agent["id"], session["id"], run["id"]


def test_view_only_admin_cannot_create_agent(view_only_admin_client, noop_provider):
    res = view_only_admin_client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer view-only"},
        json={"name": "evil", "model": "gpt-4"},
    )
    assert res.status_code == 403
    # Error envelope: {"error": {"code", "message", "status"}}
    assert "view-only" in res.json()["error"]["message"].lower()


def test_view_only_admin_cannot_update_other_tenant_agent(
    client, view_only_admin_client, noop_provider
):
    agent_id, _, _ = _create_tenant_a_resources(client)
    res = view_only_admin_client.patch(
        f"/v2/agents/{agent_id}",
        headers={"Authorization": "Bearer view-only"},
        json={"name": "hacked"},
    )
    assert res.status_code == 403


def test_view_only_admin_cannot_delete_other_tenant_agent(
    client, view_only_admin_client, noop_provider
):
    agent_id, _, _ = _create_tenant_a_resources(client)
    res = view_only_admin_client.delete(
        f"/v2/agents/{agent_id}",
        headers={"Authorization": "Bearer view-only"},
    )
    assert res.status_code == 403


def test_view_only_admin_cannot_create_session(view_only_admin_client, noop_provider):
    # Even minting a session with a non-existent agent must short-circuit
    # to 403 BEFORE any DB activity — the role check is the first guard.
    res = view_only_admin_client.post(
        f"/v2/agents/{"agt_doesnotexist"}/sessions",
        headers={"Authorization": "Bearer view-only"},
        json={"repos": []},
    )
    assert res.status_code == 403


def test_view_only_admin_cannot_delete_other_tenant_session(
    client, view_only_admin_client, noop_provider
):
    _, session_id, _ = _create_tenant_a_resources(client)
    res = view_only_admin_client.delete(
        f"/v2/sessions/{session_id}",
        headers={"Authorization": "Bearer view-only"},
    )
    assert res.status_code == 403


def test_view_only_admin_cannot_create_run(
    client, view_only_admin_client, noop_provider
):
    _, session_id, _ = _create_tenant_a_resources(client)
    res = view_only_admin_client.post(
        f"/v2/sessions/{session_id}/runs",
        headers={"Authorization": "Bearer view-only"},
        json={"prompt": {"text": "evil"}},
    )
    assert res.status_code == 403


def test_view_only_admin_cannot_cancel_run(
    client, view_only_admin_client, noop_provider
):
    _, session_id, run_id = _create_tenant_a_resources(client)
    res = view_only_admin_client.post(
        f"/v2/sessions/{session_id}/runs/{run_id}/cancel",
        headers={"Authorization": "Bearer view-only"},
    )
    assert res.status_code == 403


def test_view_only_admin_cannot_followup(client, view_only_admin_client, noop_provider):
    _, session_id, _ = _create_tenant_a_resources(client)
    res = view_only_admin_client.post(
        f"/v2/sessions/{session_id}/followup",
        headers={"Authorization": "Bearer view-only"},
        json={"prompt": {"text": "evil followup"}},
    )
    assert res.status_code == 403


def test_view_only_admin_can_still_read_other_tenant_resources(
    client, view_only_admin_client, noop_provider
):
    """The whole point of view-only is cross-tenant READ access — make
    sure we didn't accidentally lock that down too."""
    agent_id, session_id, run_id = _create_tenant_a_resources(client)

    # Reads must succeed.
    assert (
        view_only_admin_client.get(
            f"/v2/agents/{agent_id}",
            headers={"Authorization": "Bearer view-only"},
        ).status_code
        == 200
    )
    assert (
        view_only_admin_client.get(
            f"/v2/sessions/{session_id}",
            headers={"Authorization": "Bearer view-only"},
        ).status_code
        == 200
    )
    assert (
        view_only_admin_client.get(
            f"/v2/sessions/{session_id}/runs/{run_id}",
            headers={"Authorization": "Bearer view-only"},
        ).status_code
        == 200
    )
    # And list endpoints return tenant A's data (not filtered out).
    res = view_only_admin_client.get(
        "/v2/agents", headers={"Authorization": "Bearer view-only"}
    )
    assert res.status_code == 200
    assert any(a["id"] == agent_id for a in res.json()["items"])
