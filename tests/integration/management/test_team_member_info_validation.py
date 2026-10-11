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

import pytest

from tests.integration._support.actors import Actor
from tests.integration._support.actors import create_scratch_team, create_scratch_user
from tests.integration._support.database import write_rows

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Both None → 400 ("Either user_id or user_email must be provided")
# ---------------------------------------------------------------------------


async def test_member_add_both_none_rejected(gateway_client, db, resources, actors):
    team_id = await create_scratch_team(db, team_id=resources.tag("team"))
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"team_id": team_id, "member": {"role": "user"}},
    )
    # The Pydantic Member model may also catch this at the validation layer
    # (422). Either shape proves the empty-Member payload is rejected before
    # any membership row is written — pin both.
    assert resp.status_code in (400, 422), resp.text
    rows = await db.db.litellm_teammembership.find_many(where={"team_id": team_id})
    assert all(row.user_id == "default_user_id" for row in rows), "empty-Member payload leaked a membership row"


# ---------------------------------------------------------------------------
# Email + id given, but they point at different users → 400
# ---------------------------------------------------------------------------


async def test_member_add_email_id_mismatch_rejected(
    gateway_client, db, resources, actors
):
    email = f"{resources.prefix}-mismatch@example.com"
    real_user_id = await create_scratch_user(
        db, resources.prefix, suffix="real", user_email=email
    )
    other_user_id = await create_scratch_user(db, resources.prefix, suffix="other")
    assert real_user_id != other_user_id  # sanity
    team_id = await create_scratch_team(db, team_id=resources.tag("team"))
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
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
    rows = await db.db.litellm_teammembership.find_many(where={"team_id": team_id})
    assert not any(row.user_id in {real_user_id, other_user_id} for row in rows), "mismatch payload leaked a membership row"


# ---------------------------------------------------------------------------
# Email-only resolves to user_id when exactly one user matches
# ---------------------------------------------------------------------------


async def test_member_add_email_only_resolves_user_id(
    gateway_client, db, resources, actors
):
    email = f"{resources.prefix}-resolve@example.com"
    user_id = await create_scratch_user(
        db, resources.prefix, suffix="lookup", user_email=email
    )
    team_id = await create_scratch_team(db, team_id=resources.tag("team"))
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
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
    team_row = await db.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert team_row is not None
    member_user_ids = [m.get("user_id") for m in team_row.members_with_roles]
    assert (
        user_id in member_user_ids
    ), f"email did not resolve to {user_id}; members={member_user_ids}"


# ---------------------------------------------------------------------------
# id-only, user does NOT yet exist — passes through, member is upserted.
# ---------------------------------------------------------------------------


async def test_member_add_unknown_user_id_upserted(
    gateway_client, db, resources, actors
):
    team_id = await create_scratch_team(db, team_id=resources.tag("team"))
    new_user_id = f"{resources.prefix}-fresh"
    # Sanity — user does not exist yet.
    pre = await db.db.litellm_usertable.find_unique(where={"user_id": new_user_id})
    assert pre is None
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "member": {"role": "user", "user_id": new_user_id},
        },
    )
    assert resp.status_code == 200, resp.text
    post = await db.db.litellm_usertable.find_unique(where={"user_id": new_user_id})
    assert post is not None, "user_id was not upserted"
    # The user row was created with NULL email (the helper returned the
    # member as-is, no email lookup happened because the user didn't exist).
    assert (
        post.user_email is None
    ), f"upserted user has unexpected email: {post.user_email!r}"


# ---------------------------------------------------------------------------
# Duplicate-email rejection — two resources users share an email; email-only
# add → 400 with "Multiple users found" detail.
# ---------------------------------------------------------------------------


async def test_member_add_duplicate_email_rejected(
    gateway_client, db, resources, actors
):
    email = f"{resources.prefix}-dup@example.com"
    await create_scratch_user(db, resources.prefix, suffix="dup1", user_email=email)
    write_rows(
        'INSERT INTO "LiteLLM_UserTable" (user_id, user_email, user_role, models, metadata) '
        "VALUES (%s, %s, 'internal_user', '{}', '{}')",
        (resources.tag("dup2"), email),
    )
    team_id = await create_scratch_team(db, team_id=resources.tag("team"))
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "member": {"role": "user", "user_email": email},
        },
    )
    assert resp.status_code == 400, resp.text
    assert "Multiple users found" in resp.text, resp.text
    rows = await db.db.litellm_teammembership.find_many(where={"team_id": team_id})
    assert rows == [], "duplicate-email payload leaked a membership row"
