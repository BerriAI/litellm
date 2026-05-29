import uuid

import pytest

from litellm.proxy.utils import hash_token

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")

_SEED_SPEND = 5.0
_RESET_TO = 2.0


# POST /key/{key}/reset_spend. The target key is pre-seeded with spend=5.0 so
# reset_to=2.0 always clears _validate_reset_spend_value (which runs before
# authz). _check_proxy_or_team_admin_for_key then allows only PROXY_ADMIN or a
# team admin of the key's team — there is no org-admin branch, and a teamless
# "self" key has no team to admin. ORG_ADMIN-role callers are stopped 401 at
# the management-route gate before the handler runs.
_SCENARIOS = [
    ("self/proxy_admin", Actor.PROXY_ADMIN, "self", 200),
    ("self/org_admin", Actor.ORG_ADMIN, "self", 401),
    ("self/team_admin", Actor.TEAM_ADMIN, "self", 403),
    ("self/internal_user", Actor.INTERNAL_USER, "self", 403),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, "self", 403),
    ("team_alpha/proxy_admin", Actor.PROXY_ADMIN, "team_alpha", 200),
    ("team_alpha/org_admin", Actor.ORG_ADMIN, "team_alpha", 401),
    ("team_alpha/team_admin", Actor.TEAM_ADMIN, "team_alpha", 200),
    ("team_alpha/internal_user", Actor.INTERNAL_USER, "team_alpha", 403),
    ("team_alpha/owner", Actor.OWNER, "team_alpha", 403),
    ("team_alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "team_alpha", 403),
    ("team_alpha/cross_org_user", Actor.CROSS_ORG_USER, "team_alpha", 403),
    ("team_alpha/service_account", Actor.SERVICE_ACCOUNT, "team_alpha", 403),
    ("team_alpha/org_b_admin", Actor.ORG_B_ADMIN, "team_alpha", 401),
    ("team_beta/proxy_admin", Actor.PROXY_ADMIN, "team_beta", 200),
    ("team_beta/org_admin", Actor.ORG_ADMIN, "team_beta", 401),
    ("team_beta/team_admin", Actor.TEAM_ADMIN, "team_beta", 403),
    ("team_beta/cross_org_user", Actor.CROSS_ORG_USER, "team_beta", 403),
    ("team_beta/org_b_admin", Actor.ORG_B_ADMIN, "team_beta", 401),
]


async def _seed_target(proxy_client, seeder, prefix, world, shape, caller) -> str:
    if shape == "self":
        return await create_scratch_key(
            proxy_client, seeder, prefix, user_id=caller.user_id
        )
    if shape == "team_alpha":
        return await create_scratch_key(
            proxy_client,
            seeder,
            prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=TEAM_ALPHA,
        )
    if shape == "team_beta":
        return await create_scratch_key(
            proxy_client,
            seeder,
            prefix,
            user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=TEAM_BETA,
        )
    pytest.fail(f"unknown shape={shape}")  # pragma: no cover


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_reset_spend_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[actor]
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    target = await _seed_target(
        proxy_client, seeder, scratch.prefix, world, shape, caller
    )
    hashed = hash_token(target)
    await prisma.db.litellm_verificationtoken.update(
        where={"token": hashed}, data={"spend": _SEED_SPEND}
    )

    resp = await proxy_client.post(
        f"/key/{target}/reset_spend",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"reset_to": _RESET_TO},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None
    if expected_status == 200:
        assert row.spend == _RESET_TO
    else:
        assert row.spend == _SEED_SPEND, "denied but spend reset"


@pytest.mark.parametrize(
    "actor", [Actor.PROXY_ADMIN, Actor.TEAM_ADMIN], ids=["proxy_admin", "team_admin"]
)
async def test_key_reset_spend_missing_key_is_404(actor: Actor, proxy_client, world):
    """A well-formed but unseeded key is 404 before any spend validation."""
    resp = await proxy_client.post(
        f"/key/sk-{uuid.uuid4().hex}/reset_spend",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
        json={"reset_to": 0.0},
    )
    assert resp.status_code == 404, resp.text


async def test_key_reset_spend_above_current_spend_is_400(
    proxy_client, prisma, scratch, world
):
    """reset_to above the key's current spend is rejected 400."""
    admin = world.keys[Actor.PROXY_ADMIN]
    target = await create_scratch_key(
        proxy_client, admin.cleartext, scratch.prefix, user_id=admin.user_id
    )
    resp = await proxy_client.post(
        f"/key/{target}/reset_spend",
        headers={"Authorization": f"Bearer {admin.cleartext}"},
        json={"reset_to": 1.0},
    )
    assert resp.status_code == 400, resp.text
