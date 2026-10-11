import litellm
import pytest

from litellm.proxy._types import KeyManagementRoutes

from tests.integration._support.actors import Actor
from tests.integration._support.actors import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

_PERM = KeyManagementRoutes.KEY_INFO.value


# GET /team/permissions_list and POST /team/permissions_update are self-managed
# routes, so every actor reaches the handler. Both grant access to PROXY_ADMIN,
# the team admin, or an org admin of the team's org. The resources team is in
# ORG_A with TEAM_ADMIN as its team admin.
_MATRIX = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_admin", Actor.ORG_ADMIN, 200),
    ("team_admin", Actor.TEAM_ADMIN, 200),
    ("internal_user", Actor.INTERNAL_USER, 403),
    ("owner", Actor.OWNER, 403),
    ("unrelated_same_org", Actor.UNRELATED_SAME_ORG, 403),
    ("cross_org_user", Actor.CROSS_ORG_USER, 403),
    ("service_account", Actor.SERVICE_ACCOUNT, 403),
    ("org_b_admin", Actor.ORG_B_ADMIN, 403),
]


async def _seed_team(db, scratch_prefix, actors) -> None:
    await create_scratch_team(
        db,
        scratch_prefix,
        organization_id=actors.org_a_id,
        admin_user_ids=[actors.keys[Actor.TEAM_ADMIN].user_id],
        member_user_ids=[
            actors.keys[Actor.INTERNAL_USER].user_id,
            actors.keys[Actor.OWNER].user_id,
            actors.keys[Actor.UNRELATED_SAME_ORG].user_id,
            actors.keys[Actor.SERVICE_ACCOUNT].user_id,
        ],
    )


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_permissions_list_authz_matrix(
    actor: Actor, expected_status: int, gateway_client, db, resources, actors
):
    await _seed_team(db, resources.prefix, actors)
    resp = await gateway_client.get(
        f"/team/permissions_list?team_id={resources.prefix}",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"
    if expected_status == 200:
        assert resp.json()["team_id"] == resources.prefix


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_permissions_update_authz_matrix(
    actor: Actor, expected_status: int, gateway_client, db, resources, actors
):
    await _seed_team(db, resources.prefix, actors)
    resp = await gateway_client.post(
        "/team/permissions_update",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
        json={"team_id": resources.prefix, "team_member_permissions": [_PERM]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert _PERM in (row.team_member_permissions or [])
    else:
        assert _PERM not in (row.team_member_permissions or []), "denied but mutated"


async def test_team_permissions_available_team_self_join_divergence(
    gateway_client, db, resources, actors, monkeypatch
):
    """permissions_list honours the available-team self-join — a non-admin can
    READ an available team's permissions — but permissions_update deliberately
    does not: the same caller is 403 on update. default_internal_user_params is
    module-level litellm.* state, so monkeypatch save/restores it."""
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    monkeypatch.setattr(
        litellm, "default_internal_user_params", {"available_teams": [resources.prefix]}
    )
    caller = actors.keys[Actor.CROSS_ORG_USER]  # non-admin, unrelated to the team

    listed = await gateway_client.get(
        f"/team/permissions_list?team_id={resources.prefix}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert listed.status_code == 200, listed.text

    updated = await gateway_client.post(
        "/team/permissions_update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_id": resources.prefix, "team_member_permissions": [_PERM]},
    )
    assert updated.status_code == 403, updated.text


# POST /team/permissions_bulk_update is PROXY_ADMIN-only. ORG_ADMIN-role
# callers are stopped 401 by the management-route gate; INTERNAL_USER-role
# callers, on a route that is neither internal_user nor self-managed, are 401
# there too — only PROXY_ADMIN reaches the handler's own admin gate.
_BULK_MATRIX = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_admin", Actor.ORG_ADMIN, 401),
    ("team_admin", Actor.TEAM_ADMIN, 401),
    ("internal_user", Actor.INTERNAL_USER, 401),
    ("cross_org_user", Actor.CROSS_ORG_USER, 401),
    ("org_b_admin", Actor.ORG_B_ADMIN, 401),
]


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _BULK_MATRIX],
    ids=[s[0] for s in _BULK_MATRIX],
)
async def test_team_permissions_bulk_update_authz_matrix(
    actor: Actor, expected_status: int, gateway_client, db, resources, actors
):
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
        json={"team_ids": [resources.prefix], "permissions": [_PERM]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert _PERM in (row.team_member_permissions or [])
    else:
        assert _PERM not in (row.team_member_permissions or []), "denied but mutated"


async def test_team_permissions_bulk_update_no_selector_is_400(gateway_client, actors):
    """Neither team_ids nor apply_to_all_teams is a 400."""
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"permissions": [_PERM]},
    )
    assert resp.status_code == 400, resp.text
