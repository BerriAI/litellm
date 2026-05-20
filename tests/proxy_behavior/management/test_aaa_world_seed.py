"""Slice 4 smoke: every seeded actor key can authenticate and call /key/info on itself.

This is the minimum proof that the world is reachable via the real auth stack —
the real ``user_api_key_auth`` dependency hashes the cleartext token from the
Bearer header, looks it up, resolves the user/role, and the handler returns the
key info row.

If any actor 401/403s here, the seed is wrong (user_role mismatch, scope
fields, etc.) before we scale out to the matrix in Slices 7–12.
"""

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.parametrize("actor", list(Actor), ids=[a.value for a in Actor])
async def test_each_actor_can_self_info(actor, proxy_client, world):
    seeded = world.keys[actor]
    resp = await proxy_client.get(
        "/key/info",
        headers={"Authorization": f"Bearer {seeded.cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.status_code} {resp.text}"
    body = resp.json()
    # /key/info returns {"key": <hashed token>, "info": <row dict with `token` popped>}.
    assert body.get("key") == seeded.hashed, (
        f"{actor.value}: /key/info returned the wrong key "
        f"(got {body.get('key')!r}, expected {seeded.hashed!r})"
    )
    info = body["info"]
    assert info.get("user_id") == seeded.user_id, (
        f"{actor.value}: /key/info returned the wrong user_id "
        f"(got {info.get('user_id')!r}, expected {seeded.user_id!r})"
    )


async def test_proxy_admin_actor_can_create_keys_for_others(
    proxy_client, prisma, world
):
    """Diagnostic: the seeded PROXY_ADMIN actor must be able to /key/generate
    a key for another user. If this fails, the user_role is not propagating
    through user_api_key_auth → the actor's auth context disagrees with the
    DB row, and the cause is elsewhere in the auth stack (not the seed).

    On failure we also probe the underlying state so the CI log tells us
    exactly which surface returned the wrong shape: the raw token row, the
    user row, and the combined view (what the auth resolver consumes)."""
    seeder = world.keys[Actor.PROXY_ADMIN]
    target_user_id = world.keys[Actor.OWNER].user_id

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder.cleartext}"},
        json={"key_alias": "diag-proxy-admin-seeder", "user_id": target_user_id},
    )
    if resp.status_code != 200:
        token_row = await prisma.db.litellm_verificationtoken.find_unique(
            where={"token": seeder.hashed}
        )
        user_row = await prisma.db.litellm_usertable.find_unique(
            where={"user_id": seeder.user_id}
        )
        view_rows = await prisma.db.query_raw(
            'SELECT user_id, user_role FROM "LiteLLM_VerificationTokenView" '
            'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = '
            '"LiteLLM_VerificationTokenView".user_id WHERE token = $1',
            seeder.hashed,
        )
        pytest.fail(
            f"PROXY_ADMIN-seeded actor can't create keys for others: "
            f"{resp.status_code} {resp.text}\n"
            f"  seeder user_id (expected): {seeder.user_id}\n"
            f"  token row.user_id        : {getattr(token_row, 'user_id', '<missing>')!r}\n"
            f"  user  row.user_role      : {getattr(user_row, 'user_role', '<missing>')!r}\n"
            f"  view  row                 : {view_rows!r}"
        )
