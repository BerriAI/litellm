"""Phase 4 F2 — payload-level pins for key↔team reassignment.

Pins `validate_key_team_change` (key_management_endpoints.py:2953), reached
via /key/update when the request changes `team_id`.

The handler runs four guards in order; each scenario isolates one path and
asserts the rejected row's `team_id` is UNCHANGED on a fresh DB re-read —
the regression shape that matters for cross-team / IDOR-class bugs is
"row mutated despite the helper raising", and the only way to catch it
is to compare the persisted state, not the response body.

The accepted scenario pins the happy path: re-read confirms team_id
flipped to the new team and the row is otherwise intact.
"""

import uuid
from typing import Any, Dict, Optional

import pytest
from prisma import Json

from litellm.proxy.utils import hash_token

from .actors import TEAM_ALPHA, Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

KEY_MODEL = "phase4-f2-key-model"


async def _seed_key_with_limits(
    prisma,
    scratch_prefix: str,
    *,
    user_id: str,
    team_id: str,
    models: Optional[list] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
) -> str:
    """Raw-seed a scratch key with explicit models / tpm / rpm — /key/generate
    can't set these against a non-throughput team without firing F1's guards.
    Returns the cleartext key for /key/update calls."""
    cleartext = "sk-" + uuid.uuid4().hex
    data: Dict[str, Any] = {
        "token": hash_token(cleartext),
        "key_alias": f"{scratch_prefix}-key",
        "key_name": f"{scratch_prefix}-key",
        "user_id": user_id,
        "team_id": team_id,
        "models": models or [],
    }
    if tpm_limit is not None:
        data["tpm_limit"] = tpm_limit
    if rpm_limit is not None:
        data["rpm_limit"] = rpm_limit
    await prisma.db.litellm_verificationtoken.create(data=data)
    return cleartext


# ---------------------------------------------------------------------------
# Accepted — proxy admin moves a key into a scratch team that has the model,
# accommodates the limits, and has the key owner as a member.
# ---------------------------------------------------------------------------


async def test_key_team_change_accepted(proxy_client, prisma, scratch, world):
    owner_id = world.keys[Actor.OWNER].user_id
    target_team = await create_scratch_team(
        prisma,
        team_id=scratch.tag("target"),
        member_user_ids=[owner_id],
        models=[KEY_MODEL],
        tpm_limit=10_000,
        rpm_limit=1_000,
    )
    key_cleartext = await _seed_key_with_limits(
        prisma,
        scratch.prefix,
        user_id=owner_id,
        team_id=TEAM_ALPHA,
        models=[KEY_MODEL],
        tpm_limit=500,
        rpm_limit=50,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext

    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key": key_cleartext, "team_id": target_team},
    )
    assert resp.status_code == 200, resp.text

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(key_cleartext)}
    )
    assert row is not None
    assert row.team_id == target_team, "key did not move to target team"


# ---------------------------------------------------------------------------
# Rejected — each path verifies row.team_id is UNCHANGED on DB re-read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        "team_lacks_model",
        "key_tpm_exceeds_team_tpm",
        "key_rpm_exceeds_team_rpm",
        "key_owner_not_team_member",
    ],
)
async def test_key_team_change_rejected_guards(
    scenario: str, proxy_client, prisma, scratch, world
):
    owner_id = world.keys[Actor.OWNER].user_id

    team_kwargs: Dict[str, Any] = {
        "member_user_ids": [owner_id],
        "models": [KEY_MODEL],
        "tpm_limit": 10_000,
        "rpm_limit": 1_000,
    }
    key_kwargs: Dict[str, Any] = {
        "models": [KEY_MODEL],
        "tpm_limit": 500,
        "rpm_limit": 50,
    }

    if scenario == "team_lacks_model":
        team_kwargs["models"] = ["something-else"]
    elif scenario == "key_tpm_exceeds_team_tpm":
        team_kwargs["tpm_limit"] = 100  # < 500
    elif scenario == "key_rpm_exceeds_team_rpm":
        team_kwargs["rpm_limit"] = 5  # < 50
    elif scenario == "key_owner_not_team_member":
        team_kwargs["member_user_ids"] = []

    target_team = await create_scratch_team(
        prisma,
        team_id=scratch.tag("target"),
        **team_kwargs,
    )
    key_cleartext = await _seed_key_with_limits(
        prisma,
        scratch.prefix,
        user_id=owner_id,
        team_id=TEAM_ALPHA,
        **key_kwargs,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext

    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key": key_cleartext, "team_id": target_team},
    )
    # The model-mismatch path lands as 400 (ProxyException from
    # _can_object_call_model); the limit + membership paths land as 403
    # (validate_key_team_change's own HTTPException). Both shapes count as
    # "rejected" for the pin; what matters is the row stayed put.
    assert resp.status_code in (
        400,
        403,
    ), f"{scenario}: expected 400/403, got {resp.status_code}: {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(key_cleartext)}
    )
    assert row is not None
    assert row.team_id == TEAM_ALPHA, (
        f"{scenario}: row team_id mutated despite rejection — " f"got {row.team_id!r}"
    )


# ---------------------------------------------------------------------------
# Rejected — initiator is neither proxy admin, team admin, nor permission-
# granted. Pinned separately because the source team here must also have the
# initiator listed as a non-admin member (otherwise the earlier user_id
# membership guard fires first).
# ---------------------------------------------------------------------------


async def test_key_team_change_rejected_initiator_not_admin(
    proxy_client, prisma, scratch, world
):
    owner_id = world.keys[Actor.OWNER].user_id
    target_team = await create_scratch_team(
        prisma,
        team_id=scratch.tag("target"),
        # Owner is the key's user_id; member of the team to clear membership
        # guard. Internal-user role on world.keys[INTERNAL_USER] is also a
        # member of TEAM_ALPHA but never an admin → role-gate path fires.
        member_user_ids=[owner_id, world.keys[Actor.INTERNAL_USER].user_id],
        models=[KEY_MODEL],
        tpm_limit=10_000,
        rpm_limit=1_000,
    )
    key_cleartext = await _seed_key_with_limits(
        prisma,
        scratch.prefix,
        user_id=owner_id,
        team_id=TEAM_ALPHA,
        models=[KEY_MODEL],
        tpm_limit=500,
        rpm_limit=50,
    )
    # The internal_user actor is the initiator: a TEAM_ALPHA member, but not
    # an admin anywhere, and has no team_member_permissions for /key/update.
    initiator = world.keys[Actor.INTERNAL_USER].cleartext

    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {initiator}"},
        json={"key": key_cleartext, "team_id": target_team},
    )
    assert resp.status_code in (
        401,
        403,
    ), f"expected 401/403, got {resp.status_code}: {resp.text}"
    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(key_cleartext)}
    )
    assert row is not None
    assert row.team_id == TEAM_ALPHA, "row team_id mutated despite rejection"
