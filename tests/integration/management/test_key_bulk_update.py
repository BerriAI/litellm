import uuid

import pytest

from litellm.proxy.utils import hash_token

from tests.integration._support.actors import Actor
from tests.integration._support.actors import create_scratch_key, create_scratch_team, owned_settings_client

pytestmark = pytest.mark.asyncio(loop_scope="session")

_MARKER_BUDGET = 42.0


# POST /key/bulk_update is PROXY_ADMIN-only. The handler's own gate is
# user_role != PROXY_ADMIN -> 403, but ORG_ADMIN-role callers never reach it:
# the management-route gate 401s them first (the body carries no org context,
# and /key/bulk_update is an internal_user route, not an org-admin one).
# INTERNAL_USER-role callers clear the route gate and hit the handler's 403.
_MATRIX = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_admin", Actor.ORG_ADMIN, 401),
    ("team_admin", Actor.TEAM_ADMIN, 403),
    ("internal_user", Actor.INTERNAL_USER, 403),
    ("owner", Actor.OWNER, 403),
    ("unrelated_same_org", Actor.UNRELATED_SAME_ORG, 403),
    ("cross_org_user", Actor.CROSS_ORG_USER, 403),
    ("service_account", Actor.SERVICE_ACCOUNT, 403),
]


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_key_bulk_update_authz_matrix(
    actor: Actor,
    expected_status: int,
    gateway_client,
    db,
    resources,
    actors,
):
    caller = actors.keys[actor]
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    target = await create_scratch_key(
        gateway_client, seeder, resources.prefix, user_id=caller.user_id
    )
    hashed = hash_token(target)

    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"keys": [{"key": target, "max_budget": _MARKER_BUDGET}]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    row = await db.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None
    if expected_status == 200:
        body = resp.json()
        assert len(body["successful_updates"]) == 1
        assert body["failed_updates"] == []
        assert row.max_budget == _MARKER_BUDGET
    else:
        assert row.max_budget != _MARKER_BUDGET, "denied but key mutated"


async def test_key_bulk_update_empty_keys_is_400(gateway_client, actors):
    """An empty batch is rejected 400 before any per-key processing."""
    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"keys": []},
    )
    assert resp.status_code == 400, resp.text


async def test_key_bulk_update_over_max_batch_is_400(gateway_client, actors):
    """A batch larger than the 500-key cap is rejected 400."""
    items = [{"key": "sk-" + uuid.uuid4().hex} for _ in range(501)]
    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"keys": items},
    )
    assert resp.status_code == 400, resp.text


async def test_key_bulk_update_per_key_failure_is_isolated(
    gateway_client, db, resources, actors
):
    """One bad key in the batch does not abort the others — it lands in
    failed_updates while the valid key is still updated."""
    admin = actors.keys[Actor.PROXY_ADMIN]
    valid = await create_scratch_key(
        gateway_client, admin.cleartext, resources.prefix, user_id=admin.user_id
    )
    missing = "sk-" + uuid.uuid4().hex

    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {admin.cleartext}"},
        json={
            "keys": [
                {"key": valid, "max_budget": _MARKER_BUDGET},
                {"key": missing, "max_budget": _MARKER_BUDGET},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    assert len(body["successful_updates"]) == 1
    assert len(body["failed_updates"]) == 1

    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(valid)}
    )
    assert row is not None and row.max_budget == _MARKER_BUDGET


async def test_key_bulk_updates_explicit_null_cannot_remove_capped_limit(
    gateway_client, db, resources, actors, tmp_path
):
    admin = actors.keys[Actor.PROXY_ADMIN]
    headers = {"Authorization": f"Bearer {admin.cleartext}"}
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    limited, unlimited, team_limited = [
        await create_scratch_key(
            gateway_client,
            admin.cleartext,
            resources.prefix,
            user_id=admin.user_id,
            team_id=resources.prefix if name == "team-limited" else None,
            key_alias=f"{resources.prefix}-{name}",
        )
        for name in ("limited", "unlimited", "team-limited")
    ]
    for key in (limited, team_limited):
        seeded = await gateway_client.post("/key/update", headers=headers, json={"key": key, "max_budget": 50.0})
        assert seeded.status_code == 200, seeded.text

    with owned_settings_client(
        gateway_client, tmp_path, {"upperbound_key_generate_params": {"max_budget": 100.0}}
    ) as client:
        bulk = await client.post(
            "/key/bulk_update",
            headers=headers,
            json={"keys": [{"key": limited, "max_budget": None}, {"key": unlimited, "max_budget": None}]},
        )
        team_bulk = await client.post(
            "/team/key/bulk_update",
            headers=headers,
            json={"team_id": resources.prefix, "key_ids": [team_limited], "update_fields": {"max_budget": None}},
        )

    assert bulk.status_code == 200, bulk.text
    assert [item["key"] for item in bulk.json()["failed_updates"]] == [limited]
    assert [item["key"] for item in bulk.json()["successful_updates"]] == [unlimited]
    assert team_bulk.status_code == 200, team_bulk.text
    assert [item["key"] for item in team_bulk.json()["failed_updates"]] == [team_limited]
    for key in (limited, team_limited):
        row = await db.db.litellm_verificationtoken.find_unique(where={"token": hash_token(key)})
        assert row is not None and row.max_budget == 50.0
