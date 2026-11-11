"""Behavior scenarios for the credential re-encryption migration endpoints.

These run against the live ASGI app + DB. The migration POST is *not* exercised
end-to-end here because it mutates shared at-rest data (it delegates to the
master-key rotation path); that full flow is covered by the unit suite and a
live proxy run. Here we pin the HTTP-boundary contract: the read-only check is
admin-reachable, and both routes are admin-gated.

Both routes are admin-only management routes, so a non-admin key is rejected by
the ``user_api_key_auth`` layer (401) before the endpoint's own admin guard runs
-- the negative scenarios assert that framework-level rejection.
"""

import pytest


pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_migrate_encryption_check_as_admin_is_read_only(gateway_client):
    """GET /credentials/migrate-encryption/check returns a residual report (no writes)."""
    resp = await gateway_client.get(
        "/credentials/migrate-encryption/check",
        headers={"Authorization": f"Bearer {gateway_client.gateway.key}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert "residual_legacy" in body["report"]


async def test_migrate_encryption_check_requires_admin(gateway_client, resources):
    """A non-admin key cannot reach the residual scan (auth layer rejects, 401)."""
    gen = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {gateway_client.gateway.key}"},
        json={"key_alias": resources.tag("check"), "user_id": resources.tag("check-user")},
    )
    assert gen.status_code == 200, gen.text
    nonadmin_key = gen.json()["key"]

    resp = await gateway_client.get(
        "/credentials/migrate-encryption/check",
        headers={"Authorization": f"Bearer {nonadmin_key}"},
    )
    assert resp.status_code == 401, resp.text


async def test_migrate_encryption_requires_admin(gateway_client, resources):
    """A non-admin key cannot trigger the migration (auth layer rejects, 401).

    Rejection happens before any write: the admin-only route check fires in
    ``user_api_key_auth``, ahead of the endpoint body.
    """
    gen = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {gateway_client.gateway.key}"},
        json={
            "key_alias": resources.tag("migrate"),
            "user_id": resources.tag("migrate-user"),
        },
    )
    assert gen.status_code == 200, gen.text
    nonadmin_key = gen.json()["key"]

    resp = await gateway_client.post(
        "/credentials/migrate-encryption",
        headers={"Authorization": f"Bearer {nonadmin_key}"},
    )
    assert resp.status_code == 401, resp.text
