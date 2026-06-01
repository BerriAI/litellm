"""Phase 4 F4 — payload-level pins for member-info population.

Pins `_validate_and_populate_member_user_info` (team_endpoints.py:2275),
reached via /team/member_add.

PROXY_ADMIN is the caller so the upstream `_validate_team_member_add_permissions`
gate never short-circuits the payload check. Each scenario asserts BOTH the
HTTP status and the DB end-state — for accepted cases, the
LiteLLM_TeamMembership row reflects the resolved user_id (the regression
shape: a payload that silently lands the membership against the WRONG
user_id is invisible from response-body alone).
"""

from typing import Any, Dict, Optional

import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_scratch_user(
    prisma,
    scratch_prefix: str,
    *,
    suffix: str,
    user_email: Optional[str] = None,
) -> str:
    """Raw-seed a scratch-prefixed user row; returns user_id. Scratch teardown
    reclaims by user_id prefix."""
    user_id = f"{scratch_prefix}-{suffix}"
    data: Dict[str, Any] = {"user_id": user_id, "user_role": "internal_user"}
    if user_email is not None:
        data["user_email"] = user_email
    await prisma.db.litellm_usertable.create(data=data)
    return user_id


# ---------------------------------------------------------------------------
# Both None → 400 ("Either user_id or user_email must be provided")
# ---------------------------------------------------------------------------


async def test_member_add_both_none_rejected(proxy_client, prisma, scratch, world):
    team_id = await create_scratch_team(prisma, team_id=scratch.tag("team"))
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"team_id": team_id, "member": {"role": "user"}},
    )
    # The Pydantic Member model may also catch this at the validation layer
    # (422). Either shape proves the empty-Member payload is rejected before
    # any membership row is written — pin both.
    assert resp.status_code in (400, 422), resp.text
    rows = await prisma.db.litellm_teammembership.find_many(where={"team_id": team_id})
    assert rows == [], "empty-Member payload leaked a membership row"


# ---------------------------------------------------------------------------
# Email + id given, but they point at different users → 400
# ---------------------------------------------------------------------------


async def test_member_add_email_id_mismatch_rejected(
    proxy_client, prisma, scratch, world
):
    email = f"{scratch.prefix}-mismatch@example.com"
    real_user_id = await _seed_scratch_user(
        prisma, scratch.prefix, suffix="real", user_email=email
    )
    other_user_id = await _seed_scratch_user(prisma, scratch.prefix, suffix="other")
    assert real_user_id != other_user_id  # sanity
    team_id = await create_scratch_team(prisma, team_id=scratch.tag("team"))
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "member": {
                "role": "user",
                "user_email": email,
                "user_id": other_user_id,
            },
        },
    )
    assert resp.status_code == 400, resp.text
    assert "do not belong to the same user" in resp.text, resp.text
    rows = await prisma.db.litellm_teammembership.find_many(where={"team_id": team_id})
    assert rows == [], "mismatch payload leaked a membership row"


# ---------------------------------------------------------------------------
# Email-only resolves to user_id when exactly one user matches
# ---------------------------------------------------------------------------


async def test_member_add_email_only_resolves_user_id(
    proxy_client, prisma, scratch, world
):
    email = f"{scratch.prefix}-resolve@example.com"
    user_id = await _seed_scratch_user(
        prisma, scratch.prefix, suffix="lookup", user_email=email
    )
    team_id = await create_scratch_team(prisma, team_id=scratch.tag("team"))
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "member": {"role": "user", "user_email": email},
        },
    )
    assert resp.status_code == 200, resp.text
    # litellm_teammembership rows are only written when a per-member budget
    # is assigned; the default member-add path stores membership in the
    # team's members_with_roles JSON. Re-read that and assert the resolved
    # user_id landed — the regression shape is "email resolved to the WRONG
    # user_id and was silently written to members_with_roles".
    team_row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert team_row is not None
    member_user_ids = [m.get("user_id") for m in team_row.members_with_roles]
    assert (
        user_id in member_user_ids
    ), f"email did not resolve to {user_id}; members={member_user_ids}"


# ---------------------------------------------------------------------------
# id-only, user does NOT yet exist — passes through, member is upserted.
# ---------------------------------------------------------------------------


async def test_member_add_unknown_user_id_upserted(
    proxy_client, prisma, scratch, world
):
    team_id = await create_scratch_team(prisma, team_id=scratch.tag("team"))
    new_user_id = f"{scratch.prefix}-fresh"
    # Sanity — user does not exist yet.
    pre = await prisma.db.litellm_usertable.find_unique(where={"user_id": new_user_id})
    assert pre is None
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "member": {"role": "user", "user_id": new_user_id},
        },
    )
    assert resp.status_code == 200, resp.text
    post = await prisma.db.litellm_usertable.find_unique(where={"user_id": new_user_id})
    assert post is not None, "user_id was not upserted"
    # The user row was created with NULL email (the helper returned the
    # member as-is, no email lookup happened because the user didn't exist).
    assert (
        post.user_email is None
    ), f"upserted user has unexpected email: {post.user_email!r}"


# ---------------------------------------------------------------------------
# Duplicate-email rejection — two scratch users share an email; email-only
# add → 400 with "Multiple users found" detail.
# ---------------------------------------------------------------------------


async def test_member_add_duplicate_email_rejected(
    proxy_client, prisma, scratch, world
):
    email = f"{scratch.prefix}-dup@example.com"
    await _seed_scratch_user(prisma, scratch.prefix, suffix="dup1", user_email=email)
    await _seed_scratch_user(prisma, scratch.prefix, suffix="dup2", user_email=email)
    team_id = await create_scratch_team(prisma, team_id=scratch.tag("team"))
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "member": {"role": "user", "user_email": email},
        },
    )
    assert resp.status_code == 400, resp.text
    assert "Multiple users found" in resp.text, resp.text
    rows = await prisma.db.litellm_teammembership.find_many(where={"team_id": team_id})
    assert rows == [], "duplicate-email payload leaked a membership row"
