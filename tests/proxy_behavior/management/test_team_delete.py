import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/delete runs per-team _verify_team_access. The request carries the
# team's organization_id so an org admin of that org clears the management-
# route gate; a team admin is an INTERNAL_USER on a non-internal_user route,
# so a team admin never reaches the handler. Only PROXY_ADMIN and an org admin
# of the team's own org can delete it.
_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 401),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 401),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 401),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 401),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 401),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_delete_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = world.org_a_id if shape == "alpha" else world.org_b_id
    await create_scratch_team(prisma, scratch.prefix, organization_id=org_id)
    caller = world.keys[actor]

    resp = await proxy_client.post(
        "/team/delete",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_ids": [scratch.prefix], "organization_id": org_id},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    if expected_status == 200:
        assert row is None, "deleted but team row survives"
    else:
        assert row is not None, "denied but team row vanished"


async def test_team_delete_batch_with_missing_id_deletes_nothing(
    proxy_client, prisma, scratch, world
):
    """A batch is validated whole before any deletion: one missing team_id
    fails the request 404 and the accessible team in the batch survives."""
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    resp = await proxy_client.post(
        "/team/delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_ids": [scratch.prefix, "behavior-pin-no-such-team"]},
    )
    assert resp.status_code == 404, resp.text
    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None, "batch aborted but the accessible team was deleted"
