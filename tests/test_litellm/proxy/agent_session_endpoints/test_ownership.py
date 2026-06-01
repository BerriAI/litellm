"""
Validation #10 — cross-tenant isolation at all 3 levels.

Tenant A creates an agent + session + run. Tenant B (different api_key)
must get 404 on every read/write of those resources.
"""


def test_cross_tenant_isolation(client, other_tenant_client, noop_provider):
    # Tenant A creates an agent + session + run.
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "secret", "model": "gpt-4"},
    ).json()
    aid = agent["id"]
    sess = client.post(
        "/v2/sessions",
        headers={"Authorization": "Bearer k"},
        json={"agent_id": aid, "repos": []},
    ).json()
    sid = sess["id"]
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "x"}},
    ).json()
    rid = run["id"]

    # Tenant B sees nothing.
    assert (
        other_tenant_client.get(
            f"/v2/agents/{aid}", headers={"Authorization": "Bearer other"}
        ).status_code
        == 404
    )
    assert (
        other_tenant_client.get(
            f"/v2/sessions/{sid}", headers={"Authorization": "Bearer other"}
        ).status_code
        == 404
    )
    assert (
        other_tenant_client.get(
            f"/v2/sessions/{sid}/runs/{rid}",
            headers={"Authorization": "Bearer other"},
        ).status_code
        == 404
    )

    # Tenant B's list endpoints return only their own (none).
    assert (
        other_tenant_client.get(
            "/v2/agents", headers={"Authorization": "Bearer other"}
        ).json()["data"]
        == []
    )
    assert (
        other_tenant_client.get(
            "/v2/sessions", headers={"Authorization": "Bearer other"}
        ).json()["data"]
        == []
    )

    # Tenant B can't delete tenant A's agent/session.
    assert (
        other_tenant_client.delete(
            f"/v2/agents/{aid}", headers={"Authorization": "Bearer other"}
        ).status_code
        == 404
    )
    assert (
        other_tenant_client.delete(
            f"/v2/sessions/{sid}", headers={"Authorization": "Bearer other"}
        ).status_code
        == 404
    )


def test_admin_sees_all_tenants(client, admin_client, noop_provider):
    """Proxy admin can read across tenants — admins exist for support
    and ops and bypassing the owner filter is intentional."""
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "tenant-A", "model": "gpt-4"},
    ).json()

    # Admin can fetch tenant A's agent.
    res = admin_client.get(
        f"/v2/agents/{agent['id']}", headers={"Authorization": "Bearer admin"}
    )
    assert res.status_code == 200
