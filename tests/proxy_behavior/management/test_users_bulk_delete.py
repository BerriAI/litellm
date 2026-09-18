import pytest

from .actors import Actor
from .conftest import create_scratch_team, create_scratch_user

pytestmark = pytest.mark.asyncio(loop_scope="session")

_URL = "/management/v1/users/bulk_delete"

# (id, actor, victims' org, expected status, whether the victims are gone afterwards)
_MATRIX = [
    ("org_a/proxy_admin", Actor.PROXY_ADMIN, "a", 200, True),
    ("org_a/org_admin", Actor.ORG_ADMIN, "a", 200, True),
    ("org_a/org_b_admin", Actor.ORG_B_ADMIN, "a", 200, False),
    ("org_a/team_admin", Actor.TEAM_ADMIN, "a", 403, False),
    ("org_a/internal_user", Actor.INTERNAL_USER, "a", 403, False),
    ("org_a/owner", Actor.OWNER, "a", 403, False),
    ("org_a/service_account", Actor.SERVICE_ACCOUNT, "a", 403, False),
    ("no_org/proxy_admin", Actor.PROXY_ADMIN, None, 200, True),
    ("no_org/org_admin", Actor.ORG_ADMIN, None, 200, False),
]


def _member_ids(row) -> list:
    return [m["user_id"] for m in (row.members_with_roles or [])]


async def _seed_team_members(prisma, scratch, world, member_ids: list, org_id) -> None:
    """Leave behind what /team/member_add would: roster entry, `teams` array, and org membership."""
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id, member_user_ids=member_ids)
    await prisma.db.litellm_usertable.update_many(
        where={"user_id": {"in": member_ids}}, data={"teams": {"set": [scratch.prefix]}}
    )
    if org_id is None:
        return
    for uid in member_ids:
        await prisma.db.litellm_organizationmembership.create(
            data={"user_id": uid, "organization_id": org_id, "user_role": "internal_user"}
        )


@pytest.mark.parametrize(
    "actor,org,expected_status,expect_deleted",
    [(a, o, s, d) for (_id, a, o, s, d) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_users_bulk_delete_authz_matrix(
    actor: Actor,
    org,
    expected_status: int,
    expect_deleted: bool,
    proxy_client,
    prisma,
    scratch,
    world,
):
    victims = [await create_scratch_user(prisma, scratch.prefix, suffix=s) for s in ("v1", "v2")]
    keep = await create_scratch_user(prisma, scratch.prefix, suffix="keep")
    await _seed_team_members(prisma, scratch, world, victims + [keep], world.org_a_id if org == "a" else None)

    resp = await proxy_client.post(
        _URL,
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
        json={"user_ids": victims},
    )
    assert resp.status_code == expected_status, f"{actor.value}: {resp.status_code} {resp.text}"

    team = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert team is not None and keep in _member_ids(team), "unrelated member removed"
    remaining = {u.user_id for u in await prisma.db.litellm_usertable.find_many(where={"user_id": {"in": victims}})}
    if expected_status == 403:
        assert resp.headers["content-type"] == "application/problem+json"
        assert resp.json()["type"] == "urn:litellm:error:forbidden"
        assert remaining == set(victims), "denied but users deleted"
        assert set(victims) <= set(_member_ids(team)), "denied but members removed"
        return

    body = resp.json()
    assert set(body) == {"data"}
    rows = [(r["user_id"], r["success"], r["teams_removed"]) for r in body["data"]]
    if expect_deleted:
        assert rows == [(v, True, [scratch.prefix]) for v in victims]
        assert remaining == set()
        assert not set(victims) & set(_member_ids(team))
        return
    assert rows == [(v, False, []) for v in victims]
    assert all("not within your admin scope" in r["error"] for r in body["data"])
    assert remaining == set(victims), "out-of-scope rows reported failed but users deleted"
    assert set(victims) <= set(_member_ids(team))


async def test_users_bulk_delete_reports_each_row_in_order(proxy_client, prisma, scratch, world):
    victim = await create_scratch_user(prisma, scratch.prefix, suffix="victim")
    ghost = scratch.tag("ghost")

    resp = await proxy_client.post(
        _URL,
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [ghost, victim, victim]},
    )
    assert resp.status_code == 200, resp.text
    assert [(r["user_id"], r["success"]) for r in resp.json()["data"]] == [
        (ghost, False),
        (victim, True),
        (victim, False),
    ]
    assert await prisma.db.litellm_usertable.find_unique(where={"user_id": victim}) is None


async def test_users_bulk_delete_unknown_query_param_is_400_problem(proxy_client, prisma, scratch, world):
    victim = await create_scratch_user(prisma, scratch.prefix, suffix="victim")

    resp = await proxy_client.post(
        f"{_URL}?dry_run=1",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [victim]},
    )
    assert resp.status_code == 400, resp.text
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["type"] == "urn:litellm:error:unknown-query-parameter"
    assert "dry_run" in resp.json()["detail"]
    assert await prisma.db.litellm_usertable.find_unique(where={"user_id": victim}) is not None


async def test_users_bulk_delete_unknown_body_field_is_422_problem(proxy_client, prisma, scratch, world):
    victim = await create_scratch_user(prisma, scratch.prefix, suffix="victim")

    resp = await proxy_client.post(
        _URL,
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [victim], "dry_run": True},
    )
    assert resp.status_code == 422, resp.text
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["type"] == "urn:litellm:error:invalid-request-body"
    assert "dry_run" in resp.json()["detail"]
    assert await prisma.db.litellm_usertable.find_unique(where={"user_id": victim}) is not None
