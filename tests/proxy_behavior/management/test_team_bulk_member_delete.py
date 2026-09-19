import pytest
from prisma import Json

from .actors import Actor
from .conftest import create_scratch_team, create_scratch_user

pytestmark = pytest.mark.asyncio(loop_scope="session")

_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 200),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 403),
    ("alpha/owner", Actor.OWNER, "alpha", 403),
    ("alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "alpha", 403),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 403),
    ("alpha/service_account", Actor.SERVICE_ACCOUNT, "alpha", 403),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 403),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 403),
    ("beta/team_admin", Actor.TEAM_ADMIN, "beta", 403),
    ("beta/internal_user", Actor.INTERNAL_USER, "beta", 403),
    ("beta/owner", Actor.OWNER, "beta", 403),
    ("beta/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "beta", 403),
    ("beta/cross_org_user", Actor.CROSS_ORG_USER, "beta", 403),
    ("beta/service_account", Actor.SERVICE_ACCOUNT, "beta", 403),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


async def _seed_target(prisma, world, shape: str, team_id: str, victim_ids: list) -> None:
    if shape == "alpha":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_a_id,
            admin_user_ids=[world.keys[Actor.TEAM_ADMIN].user_id],
            member_user_ids=victim_ids,
        )
    elif shape == "beta":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_b_id,
            member_user_ids=victim_ids,
        )
    else:  # pragma: no cover - guard
        pytest.fail(f"unknown shape={shape}")


def _member_ids(row) -> list:
    return [m["user_id"] for m in (row.members_with_roles or [])]


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_bulk_member_delete_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    victims = [scratch.tag("v1"), scratch.tag("v2")]
    keep = scratch.tag("keep")
    await _seed_target(prisma, world, shape, scratch.prefix, victims + [keep])
    caller = world.keys[actor]

    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.prefix}/members/bulk_delete",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"members": [{"user_id": v} for v in victims]},
    )
    assert resp.status_code == expected_status, f"{actor.value} {shape}: {resp.status_code} {resp.text}"
    if expected_status == 403:
        assert resp.headers["content-type"] == "application/problem+json"
        assert resp.json()["type"] == "urn:litellm:error:forbidden"

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert row is not None
    assert keep in _member_ids(row), "unrelated member removed"
    if expected_status == 200:
        assert [(r["user_id"], r["success"]) for r in resp.json()["data"]] == [(v, True) for v in victims]
        assert not set(victims) & set(_member_ids(row))
    else:
        assert set(victims) <= set(_member_ids(row)), "denied but members removed"


async def test_team_bulk_member_delete_reports_each_row_in_order(proxy_client, prisma, scratch, world):
    victim = scratch.tag("victim")
    keep = scratch.tag("keep")
    stranger = scratch.tag("stranger")
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id, member_user_ids=[victim, keep])

    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.prefix}/members/bulk_delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"members": [{"user_id": stranger}, {"user_id": victim}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"data"}
    assert [(r["user_id"], r["success"]) for r in body["data"]] == [
        (stranger, False),
        (victim, True),
    ]
    assert body["data"][0]["error"] == "User not found in team"

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert row is not None and _member_ids(row) == [keep]


async def test_team_bulk_member_delete_by_id_removes_a_legacy_email_only_roster_entry(
    proxy_client, prisma, scratch, world
):
    email = f"{scratch.prefix}@example.com"
    victim = await create_scratch_user(prisma, scratch.prefix, suffix="victim", user_email=email)
    keep = scratch.tag("keep")
    await prisma.db.litellm_teamtable.create(
        data={
            "team_id": scratch.prefix,
            "team_alias": scratch.prefix,
            "organization_id": world.org_a_id,
            "members_with_roles": Json([{"user_email": email, "role": "user"}, {"user_id": keep, "role": "user"}]),
        }
    )
    await prisma.db.litellm_usertable.update(where={"user_id": victim}, data={"teams": [scratch.prefix]})

    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.prefix}/members/bulk_delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"members": [{"user_id": victim}]},
    )
    assert resp.status_code == 200, resp.text
    assert [(r["user_id"], r["success"]) for r in resp.json()["data"]] == [(victim, True)]

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert row is not None and [(m["user_id"], m.get("user_email")) for m in row.members_with_roles] == [(keep, None)]
    user = await prisma.db.litellm_usertable.find_unique(where={"user_id": victim})
    assert user is not None and user.teams == []


async def test_team_bulk_member_delete_row_naming_both_identifiers_is_422(proxy_client, prisma, scratch, world):
    victim = scratch.tag("victim")
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id, member_user_ids=[victim])

    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.prefix}/members/bulk_delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"members": [{"user_id": victim, "user_email": f"{victim}@example.com"}]},
    )
    assert resp.status_code == 422, resp.text
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["type"] == "urn:litellm:error:invalid-request-body"
    assert (
        resp.json()["detail"]
        == "members.0: Value error, Each member must be identified by exactly one of user_id or user_email"
    )

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert row is not None and victim in _member_ids(row)


async def test_team_bulk_member_delete_unknown_query_param_is_400(proxy_client, prisma, scratch, world):
    victim = scratch.tag("victim")
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id, member_user_ids=[victim])

    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.prefix}/members/bulk_delete?dry_run=1",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"members": [{"user_id": victim}]},
    )
    assert resp.status_code == 400, resp.text
    assert resp.headers["content-type"] == "application/problem+json"
    assert "dry_run" in resp.json()["detail"]

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert row is not None and victim in _member_ids(row)


async def test_team_bulk_member_delete_unknown_body_field_is_422(proxy_client, prisma, scratch, world):
    victim = scratch.tag("victim")
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id, member_user_ids=[victim])

    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.prefix}/members/bulk_delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": scratch.prefix, "members": [{"user_id": victim}]},
    )
    assert resp.status_code == 422, resp.text
    assert "team_id" in resp.json()["detail"]

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": scratch.prefix})
    assert row is not None and victim in _member_ids(row)


async def test_team_bulk_member_delete_unknown_team_is_404_problem(proxy_client, scratch, world):
    resp = await proxy_client.post(
        f"/management/v1/teams/{scratch.tag('missing')}/members/bulk_delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"members": [{"user_id": scratch.tag("victim")}]},
    )
    assert resp.status_code == 404, resp.text
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["type"] == "urn:litellm:error:team-not-found"
