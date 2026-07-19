import uuid

import pytest

from litellm.proxy.utils import hash_token

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /key/block + /key/unblock. PROXY_ADMIN bypasses. ORG_ADMIN-role callers
# are stopped 401 by the management-route gate BEFORE the handler runs — the
# body carries no organization_id, so the gate has no org context and falls
# back to proxy-admin-only. The handler's own _check_key_admin_access org-admin
# branch is therefore unreachable via these routes. INTERNAL_USER-role callers
# do reach _check_key_admin_access: a team admin of the key's team passes (200);
# everyone else (incl. a teamless "self" key with no team to admin) is 403.
_SCENARIOS = [
    ("self/proxy_admin", Actor.PROXY_ADMIN, "self", 200),
    ("self/org_admin", Actor.ORG_ADMIN, "self", 401),
    ("self/team_admin", Actor.TEAM_ADMIN, "self", 403),
    ("self/internal_user", Actor.INTERNAL_USER, "self", 403),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, "self", 403),
    ("owner/proxy_admin", Actor.PROXY_ADMIN, "owner", 200),
    ("owner/org_admin", Actor.ORG_ADMIN, "owner", 401),
    ("owner/team_admin", Actor.TEAM_ADMIN, "owner", 200),
    ("owner/internal_user", Actor.INTERNAL_USER, "owner", 403),
    ("owner/owner", Actor.OWNER, "owner", 403),
    ("owner/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "owner", 403),
    ("owner/cross_org_user", Actor.CROSS_ORG_USER, "owner", 403),
    ("owner/service_account", Actor.SERVICE_ACCOUNT, "owner", 403),
    ("owner/org_b_admin", Actor.ORG_B_ADMIN, "owner", 401),
    ("cross_org/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org/team_admin", Actor.TEAM_ADMIN, "cross_org", 403),
    ("cross_org/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 403),
    ("cross_org/org_b_admin", Actor.ORG_B_ADMIN, "cross_org", 401),
]


async def _seed_target(proxy_client, seeder, scratch_prefix, world, shape, caller):
    if shape == "self":
        return await create_scratch_key(
            proxy_client, seeder, scratch_prefix, user_id=caller.user_id
        )
    if shape == "owner":
        return await create_scratch_key(
            proxy_client,
            seeder,
            scratch_prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=TEAM_ALPHA,
        )
    if shape == "cross_org":
        return await create_scratch_key(
            proxy_client,
            seeder,
            scratch_prefix,
            user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=TEAM_BETA,
        )
    pytest.fail(f"unknown shape={shape}")  # pragma: no cover


@pytest.mark.parametrize("route", ["block", "unblock"])
@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_block_unblock_authz_matrix(
    route: str,
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
    target_cleartext = await _seed_target(
        proxy_client, seeder, scratch.prefix, world, shape, caller
    )
    target_hashed = hash_token(target_cleartext)

    # /unblock starts from a blocked row so a 200 is observable as True->False.
    if route == "unblock":
        await prisma.db.litellm_verificationtoken.update(
            where={"token": target_hashed}, data={"blocked": True}
        )

    resp = await proxy_client.post(
        f"/key/{route}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"key": target_cleartext},
    )
    assert (
        resp.status_code == expected_status
    ), f"{route} {actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": target_hashed}
    )
    assert row is not None
    # A never-blocked key reads back blocked=None; treat that as not-blocked.
    if expected_status == 200:
        assert bool(row.blocked) is (route == "block")
    else:
        # A denial leaves the blocked column at its pre-request value.
        assert bool(row.blocked) is (route == "unblock"), "denied but blocked mutated"


async def test_key_block_unblock_round_trip(proxy_client, prisma, scratch, world):
    """PROXY_ADMIN block then unblock flips the blocked column True then False."""
    admin = world.keys[Actor.PROXY_ADMIN]
    target = await create_scratch_key(
        proxy_client, admin.cleartext, scratch.prefix, user_id=admin.user_id
    )
    hashed = hash_token(target)
    headers = {"Authorization": f"Bearer {admin.cleartext}"}

    blocked = await proxy_client.post(
        "/key/block", headers=headers, json={"key": target}
    )
    assert blocked.status_code == 200, blocked.text
    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None and row.blocked is True

    unblocked = await proxy_client.post(
        "/key/unblock", headers=headers, json={"key": target}
    )
    assert unblocked.status_code == 200, unblocked.text
    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None and row.blocked is False


@pytest.mark.parametrize("route", ["block", "unblock"])
@pytest.mark.parametrize(
    "actor", [Actor.PROXY_ADMIN, Actor.TEAM_ADMIN], ids=["proxy_admin", "team_admin"]
)
async def test_key_block_unblock_missing_key_returns_404(
    route: str, actor: Actor, proxy_client, world
):
    """A well-formed but unseeded key is 404 — not 401/403 — for both the
    PROXY_ADMIN existence check and the non-admin _check_key_admin_access path."""
    caller = world.keys[actor]
    missing = "sk-" + uuid.uuid4().hex
    resp = await proxy_client.post(
        f"/key/{route}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"key": missing},
    )
    assert (
        resp.status_code == 404
    ), f"{route} {actor.value}: {resp.status_code} {resp.text}"
