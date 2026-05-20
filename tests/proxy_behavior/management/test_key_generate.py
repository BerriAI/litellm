"""Slice 7 — actor × target authz matrix for ``POST /key/generate``.

The matrix pins the proxy's current authorization behavior for key creation.
Two boundary axes:

  * **Self-create.** Each of the 8 seeded actors creates a key with no
    ``team_id`` / no ``user_id`` override. The expected outcome is the actor's
    current right to create a virtual key for themselves.
  * **Cross-scope create.** A subset of actors create a key scoped to a team
    that may or may not match their org / team membership. This is the IDOR
    boundary — a passing test means an unauthorized actor was *correctly*
    blocked; a failing test (after a refactor) means the boundary moved.

Expected status codes were observed against the real handler and pinned here.
Future PRs that change these codes will turn this matrix red, surfacing the
behavior change for review.
"""

from typing import Any, Dict, Optional

import pytest

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import MASTER_KEY

pytestmark = pytest.mark.asyncio(loop_scope="session")


# Each row is (id, actor, body_extras, expected_status). Codes are PINNED
# against the current handler's observed behavior — the point of these tests
# is to red-flag *changes* to that behavior, not to assert what's ideal.
# A future PR that flips any code here is a behavior change that needs review.
_SCENARIOS = [
    # ─── Self-create: actor generates a key for themselves ───────────────
    ("self/proxy_admin", Actor.PROXY_ADMIN, {}, 200),
    # org_admin currently 401s on /key/generate (role-gate before scope check).
    ("self/org_admin", Actor.ORG_ADMIN, {}, 401),
    ("self/team_admin", Actor.TEAM_ADMIN, {}, 200),
    ("self/internal_user", Actor.INTERNAL_USER, {}, 200),
    ("self/owner", Actor.OWNER, {}, 200),
    ("self/unrelated_same_org", Actor.UNRELATED_SAME_ORG, {}, 200),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, {}, 200),
    ("self/service_account", Actor.SERVICE_ACCOUNT, {}, 200),
    # ─── team_id = team_alpha (org_a) ─────────────────────────────────────
    ("team_alpha/proxy_admin", Actor.PROXY_ADMIN, {"team_id": TEAM_ALPHA}, 200),
    # org_admin is blocked by the role gate before the team scope check runs.
    ("team_alpha/org_admin", Actor.ORG_ADMIN, {"team_id": TEAM_ALPHA}, 401),
    # team_admin is admin of team_alpha — allowed.
    ("team_alpha/team_admin", Actor.TEAM_ADMIN, {"team_id": TEAM_ALPHA}, 200),
    # Regular team member without key-create permissions: 401 + team_member_permission_error.
    ("team_alpha/internal_user", Actor.INTERNAL_USER, {"team_id": TEAM_ALPHA}, 401),
    # Cross-org user is "not assigned" — 400 fires before team-member-perms.
    ("team_alpha/cross_org_user", Actor.CROSS_ORG_USER, {"team_id": TEAM_ALPHA}, 400),
    # ─── team_id = team_beta (org_b) ──────────────────────────────────────
    ("team_beta/proxy_admin", Actor.PROXY_ADMIN, {"team_id": TEAM_BETA}, 200),
    ("team_beta/org_admin", Actor.ORG_ADMIN, {"team_id": TEAM_BETA}, 401),
    # team_admin is not a member of team_beta → "not assigned" 400.
    ("team_beta/team_admin", Actor.TEAM_ADMIN, {"team_id": TEAM_BETA}, 400),
    ("team_beta/internal_user", Actor.INTERNAL_USER, {"team_id": TEAM_BETA}, 400),
    # cross_org_user IS a member of team_beta (no admin) → team_member_perm 401.
    ("team_beta/cross_org_user", Actor.CROSS_ORG_USER, {"team_id": TEAM_BETA}, 401),
]


@pytest.mark.parametrize(
    "actor,body_extras,expected_status",
    [(actor, body, expected) for (_id, actor, body, expected) in _SCENARIOS],
    ids=[scenario[0] for scenario in _SCENARIOS],
)
async def test_key_generate_authz_matrix(
    actor: Actor,
    body_extras: Dict[str, Any],
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    seeded = world.keys[actor]
    body: Dict[str, Any] = {"key_alias": scratch.prefix, **body_extras}

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeded.cleartext}"},
        json=body,
    )
    assert resp.status_code == expected_status, (
        f"{actor.value} POST /key/generate {body!r} → {resp.status_code} "
        f"(expected {expected_status}). body={resp.text}"
    )

    if expected_status == 200:
        # Allowed: prove the row landed under the scratch namespace.
        body_json = resp.json()
        cleartext = body_json["key"]
        assert cleartext.startswith("sk-")
        rows = await prisma.db.litellm_verificationtoken.find_many(
            where={"key_alias": scratch.prefix}
        )
        assert (
            len(rows) == 1
        ), f"{actor.value}: expected exactly one row under scratch, got {len(rows)}"
    else:
        # Denied: prove no row was written.
        rows = await prisma.db.litellm_verificationtoken.find_many(
            where={"key_alias": scratch.prefix}
        )
        assert rows == [], (
            f"{actor.value}: handler returned {expected_status} but row leaked: "
            f"{rows[0].token_id if rows else None}"
        )
