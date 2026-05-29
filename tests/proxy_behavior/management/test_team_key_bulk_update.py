import uuid

import pytest

from litellm.proxy._types import KeyManagementRoutes
from litellm.proxy.utils import hash_token

from .actors import Actor
from .conftest import create_scratch_key, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

_MARKER_BUDGET = 42.0
_KEY_UPDATE = KeyManagementRoutes.KEY_UPDATE.value


# POST /team/key/bulk_update — PROXY_ADMIN bypasses; otherwise
# can_team_member_execute_key_management_endpoint runs with route=KEY_UPDATE.
# A team admin always passes; a "user"-role member passes only when the team's
# team_member_permissions grants /key/update; a non-member is 401. ORG_ADMIN is
# stopped 401 at the management-route gate before the handler (the body has a
# team_id but no organization_id, so the org-admin route branch never matches).
_MATRIX = [
    ("admin/proxy_admin", Actor.PROXY_ADMIN, "admin", 200),
    ("admin/internal_user", Actor.INTERNAL_USER, "admin", 200),
    ("member_allowed/internal_user", Actor.INTERNAL_USER, "member_allowed", 200),
    ("member_denied/internal_user", Actor.INTERNAL_USER, "member_denied", 401),
    ("nonmember/internal_user", Actor.INTERNAL_USER, "nonmember", 401),
    ("nonmember/org_admin", Actor.ORG_ADMIN, "nonmember", 401),
    ("nonmember/proxy_admin", Actor.PROXY_ADMIN, "nonmember", 200),
]


async def _seed_team_key(prisma, proxy_client, prefix: str, world, shape: str) -> str:
    """Raw-seed the scratch team for `shape`, return a team key's cleartext."""
    internal = world.keys[Actor.INTERNAL_USER].user_id
    owner = world.keys[Actor.OWNER].user_id
    if shape == "admin":
        await create_scratch_team(
            prisma, prefix, organization_id=world.org_a_id, admin_user_ids=[internal]
        )
        key_owner = internal
    elif shape == "member_allowed":
        await create_scratch_team(
            prisma,
            prefix,
            organization_id=world.org_a_id,
            admin_user_ids=[owner],
            member_user_ids=[internal],
            team_member_permissions=[_KEY_UPDATE],
        )
        key_owner = owner
    elif shape == "member_denied":
        await create_scratch_team(
            prisma,
            prefix,
            organization_id=world.org_a_id,
            admin_user_ids=[owner],
            member_user_ids=[internal],
            team_member_permissions=[],
        )
        key_owner = owner
    elif shape == "nonmember":
        await create_scratch_team(
            prisma, prefix, organization_id=world.org_a_id, admin_user_ids=[owner]
        )
        key_owner = owner
    else:
        pytest.fail(f"unknown shape={shape}")  # pragma: no cover
    return await create_scratch_key(
        proxy_client,
        world.keys[Actor.PROXY_ADMIN].cleartext,
        prefix,
        user_id=key_owner,
        team_id=prefix,
    )


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_key_bulk_update_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    key = await _seed_team_key(prisma, proxy_client, scratch.prefix, world, shape)
    hashed = hash_token(key)
    caller = world.keys[actor]

    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={
            "team_id": scratch.prefix,
            "key_ids": [key],
            "update_fields": {"max_budget": _MARKER_BUDGET},
        },
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None
    if expected_status == 200:
        assert len(resp.json()["successful_updates"]) == 1
        assert row.max_budget == _MARKER_BUDGET
    else:
        assert row.max_budget != _MARKER_BUDGET, "denied but key mutated"


async def test_team_key_bulk_update_requires_team_id(
    proxy_client, prisma, scratch, world
):
    """An empty team_id is rejected 400."""
    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "team_id": "",
            "key_ids": ["sk-" + uuid.uuid4().hex],
            "update_fields": {"max_budget": _MARKER_BUDGET},
        },
    )
    assert resp.status_code == 400, resp.text


async def test_team_key_bulk_update_all_keys_in_team(
    proxy_client, prisma, scratch, world
):
    """all_keys_in_team=True broadcasts the update to every key in the team."""
    admin = world.keys[Actor.PROXY_ADMIN].cleartext
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    keys = [
        await create_scratch_key(
            proxy_client,
            admin,
            scratch.prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=scratch.prefix,
            key_alias=f"{scratch.prefix}-k{i}",
        )
        for i in range(2)
    ]

    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {admin}"},
        json={
            "team_id": scratch.prefix,
            "all_keys_in_team": True,
            "update_fields": {"max_budget": _MARKER_BUDGET},
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["successful_updates"]) == 2
    for key in keys:
        row = await prisma.db.litellm_verificationtoken.find_unique(
            where={"token": hash_token(key)}
        )
        assert row is not None and row.max_budget == _MARKER_BUDGET


async def test_team_key_bulk_update_no_keys_found_is_404(
    proxy_client, prisma, scratch, world
):
    """all_keys_in_team=True on a team with no keys is a top-level 404."""
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "team_id": scratch.prefix,
            "all_keys_in_team": True,
            "update_fields": {"max_budget": _MARKER_BUDGET},
        },
    )
    assert resp.status_code == 404, resp.text


async def test_team_key_bulk_update_missing_key_is_isolated(
    proxy_client, prisma, scratch, world
):
    """A key_id absent from the team lands in failed_updates; the batch still
    returns 200 and the real key is updated."""
    admin = world.keys[Actor.PROXY_ADMIN].cleartext
    real = await _seed_team_key(
        prisma, proxy_client, scratch.prefix, world, "nonmember"
    )
    missing = "sk-" + uuid.uuid4().hex

    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {admin}"},
        json={
            "team_id": scratch.prefix,
            "key_ids": [real, missing],
            "update_fields": {"max_budget": _MARKER_BUDGET},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    assert len(body["successful_updates"]) == 1
    assert len(body["failed_updates"]) == 1

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(real)}
    )
    assert row is not None and row.max_budget == _MARKER_BUDGET
