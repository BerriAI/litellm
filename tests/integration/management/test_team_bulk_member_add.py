import pytest

from tests.integration._support.actors import Actor
from tests.integration._support.actors import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _member_ids(row) -> list:
    return [m["user_id"] for m in (row.members_with_roles or [])]


async def test_team_bulk_member_add_proxy_admin_adds_explicit_members(
    gateway_client, db, resources, actors
):
    """PROXY_ADMIN bulk-adds an explicit member list to a resources team."""
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    new_member = resources.tag("m1")
    resp = await gateway_client.post(
        "/team/bulk_member_add",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "team_id": resources.prefix,
            "members": [{"user_id": new_member, "role": "user"}],
        },
    )
    assert resp.status_code == 200, resp.text
    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None and new_member in _member_ids(row)


async def test_team_bulk_member_add_empty_members_is_400(
    gateway_client, db, resources, actors
):
    """An empty member list (with all_users unset) is rejected 400."""
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    resp = await gateway_client.post(
        "/team/bulk_member_add",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "members": []},
    )
    assert resp.status_code == 400, resp.text


async def test_team_bulk_member_add_over_max_batch_is_400(
    gateway_client, db, resources, actors
):
    """A member list larger than the 500-member cap is rejected 400."""
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    members = [
        {"user_id": f"{resources.prefix}-u{i}", "role": "user"} for i in range(501)
    ]
    resp = await gateway_client.post(
        "/team/bulk_member_add",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "members": members},
    )
    assert resp.status_code == 400, resp.text


@pytest.mark.parametrize(
    "actor",
    [Actor.TEAM_ADMIN, Actor.INTERNAL_USER],
    ids=["team_admin", "internal_user"],
)
async def test_team_bulk_member_add_non_admin_is_401(
    actor: Actor, gateway_client, db, resources, actors
):
    """/team/bulk_member_add is neither an internal_user nor a self-managed
    route — a non-proxy-admin with no org context is 401 at the route gate."""
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    resp = await gateway_client.post(
        "/team/bulk_member_add",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
        json={
            "team_id": resources.prefix,
            "members": [{"user_id": resources.tag("m"), "role": "user"}],
        },
    )
    assert resp.status_code == 401, f"{actor.value}: {resp.status_code} {resp.text}"


async def test_team_bulk_member_add_all_users_proxy_admin(
    gateway_client, db, resources, actors
):
    """all_users=True pulls every user in the DB into the team. The route is
    reachable only by PROXY_ADMIN (the route gate 401s every other actor — even
    an org admin with organization_id in the body), so the handler's own
    all_users PROXY_ADMIN gate is never the deciding check at the boundary."""
    await create_scratch_team(db, resources.prefix, organization_id=actors.org_a_id)
    resp = await gateway_client.post(
        "/team/bulk_member_add",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "all_users": True},
    )
    assert resp.status_code == 200, resp.text
    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None
    member_ids = _member_ids(row)
    # every actors actor is a user in the DB, so all are now team members
    assert actors.keys[Actor.INTERNAL_USER].user_id in member_ids
