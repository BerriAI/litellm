import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/block + /team/unblock. The handler gate is _verify_team_access
# (proxy admin / team admin / org admin), but the management-route gate fronts
# it: the request carries the team's organization_id so an org admin of that
# org clears the gate's org-scoped branch. A team admin is an INTERNAL_USER
# and these are not internal_user routes, so a team admin can never reach the
# handler — only PROXY_ADMIN and an org admin of the team's own org pass.
_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 401),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 401),
    ("alpha/owner", Actor.OWNER, "alpha", 401),
    ("alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "alpha", 401),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 401),
    ("alpha/service_account", Actor.SERVICE_ACCOUNT, "alpha", 401),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 401),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 401),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


async def _seed_target(prisma, world, shape: str, team_id: str) -> str:
    """Raw-seed the scratch target team; returns its organization_id."""
    org_id = world.org_a_id if shape == "alpha" else world.org_b_id
    await create_scratch_team(prisma, team_id, organization_id=org_id)
    return org_id


@pytest.mark.parametrize("route", ["block", "unblock"])
@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_block_unblock_authz_matrix(
    route: str,
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = await _seed_target(prisma, world, shape, scratch.prefix)
    caller = world.keys[actor]

    # /unblock starts from a blocked row so a 200 is observable as True->False.
    if route == "unblock":
        await prisma.db.litellm_teamtable.update(
            where={"team_id": scratch.prefix}, data={"blocked": True}
        )

    resp = await proxy_client.post(
        f"/team/{route}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_id": scratch.prefix, "organization_id": org_id},
    )
    assert (
        resp.status_code == expected_status
    ), f"{route} {actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert bool(row.blocked) is (route == "block")
    else:
        assert bool(row.blocked) is (route == "unblock"), "denied but blocked mutated"


async def test_team_block_unblock_round_trip(proxy_client, prisma, scratch, world):
    """PROXY_ADMIN block then unblock flips the blocked column True then False."""
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    headers = {"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"}

    blocked = await proxy_client.post(
        "/team/block", headers=headers, json={"team_id": scratch.prefix}
    )
    assert blocked.status_code == 200, blocked.text
    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None and row.blocked is True

    unblocked = await proxy_client.post(
        "/team/unblock", headers=headers, json={"team_id": scratch.prefix}
    )
    assert unblocked.status_code == 200, unblocked.text
    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None and row.blocked is False


@pytest.mark.parametrize("route", ["block", "unblock"])
async def test_team_block_unblock_missing_team_is_404(route: str, proxy_client, world):
    """A team_id absent from the DB is 404 — the existence check precedes authz."""
    resp = await proxy_client.post(
        f"/team/{route}",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": "behavior-pin-no-such-team"},
    )
    assert resp.status_code == 404, resp.text
