"""Phase 4 F3 — payload-level pins for team budget & rate-limit enforcement.

Pins the five helpers
  * _check_team_model_specific_limits     (team_endpoints.py:442)
  * _check_team_rpm_tpm_limits            (team_endpoints.py:527)
  * check_org_team_model_specific_limits  (team_endpoints.py:569)
  * check_org_team_rpm_tpm_limits         (team_endpoints.py:603)
  * _check_org_team_limits                (team_endpoints.py:628)
  * _check_user_team_limits               (team_endpoints.py:734)

Driven through /team/new + /team/update.

Structural finding pinned here, identical in shape to F1's org aggregate:
both call sites (lines 985 + 1751) load the org via `get_org_object`
WITHOUT `include_budget_table=True`, so `org_table.litellm_budget_table`
is `None` and the org max_budget / org tpm / org rpm guards inside
`_check_org_team_limits` (lines 641–694, 670–694) silently no-op. The
`models` subset guard (lines 654–667) IS reachable because it reads
`org_table.models` directly. The `_check_user_team_limits` guards reach
all branches through `user_api_key_dict`, no relation include needed.
"""

import uuid
from typing import Any, Dict, Optional

import pytest

from litellm.proxy.utils import hash_token

from .actors import Actor
from .conftest import MASTER_KEY, create_scratch_org, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_scratch_actor_with_caps(
    prisma,
    scratch_prefix: str,
    *,
    user_role: str = "internal_user",
    models: Optional[list] = None,
    max_budget: Optional[float] = None,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
) -> str:
    """Raw-seed a scratch actor + verification token, with the token carrying
    explicit caps that flow into `user_api_key_dict` at request time.

    Returns cleartext key. Used by F3 user-team-limit scenarios where the
    caller's caps drive the rejection. Uses 'internal_user' role plus a
    token-level `allowed_routes` whitelist of /team/new + /team/update —
    that whitelist is the only way past the admin-only role gate in
    `RouteChecks.non_proxy_admin_allowed_routes_check`; without it the
    non-admin caller would 401 before `_check_user_team_limits` ever fires.
    """
    user_id = f"{scratch_prefix}-team-creator"
    cleartext = "sk-" + uuid.uuid4().hex
    await prisma.db.litellm_usertable.create(
        data={
            "user_id": user_id,
            "user_role": user_role,
            "max_budget": max_budget,
        }
    )
    token_data: Dict[str, Any] = {
        "token": hash_token(cleartext),
        "key_name": f"{scratch_prefix}-team-creator-key",
        "key_alias": f"{scratch_prefix}-team-creator-alias",
        "user_id": user_id,
        "models": models if models is not None else [],
        "allowed_routes": ["/team/new", "/team/update"],
    }
    if tpm_limit is not None:
        token_data["tpm_limit"] = tpm_limit
    if rpm_limit is not None:
        token_data["rpm_limit"] = rpm_limit
    await prisma.db.litellm_verificationtoken.create(data=token_data)
    return cleartext


# ---------------------------------------------------------------------------
# _check_org_team_limits — models subset guard (the one path that reaches)
# ---------------------------------------------------------------------------

_ORG_MODEL_SCENARIOS = [
    (
        "org_models/team_subset_accepted",
        ["allowed-model"],
        {"models": ["allowed-model"]},
        200,
    ),
    (
        "org_models/team_extra_model_rejected",
        ["allowed-model"],
        {"models": ["forbidden-model"]},
        400,
    ),
    (
        "org_models/all_proxy_models_skips_check",
        ["all-proxy-models"],
        {"models": ["any-model-at-all"]},
        200,
    ),
]


@pytest.mark.parametrize(
    "org_models,body_extras,expected_status",
    [(b, c, d) for (_id, b, c, d) in _ORG_MODEL_SCENARIOS],
    ids=[s[0] for s in _ORG_MODEL_SCENARIOS],
)
async def test_check_org_team_limits_models_subset(
    org_models,
    body_extras: Dict[str, Any],
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = await create_scratch_org(prisma, scratch.prefix, models=org_models)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    team_id = scratch.tag("team")
    body: Dict[str, Any] = {
        "team_id": team_id,
        "team_alias": scratch.prefix,
        "organization_id": org_id,
        **body_extras,
    }
    resp = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{body!r} → {resp.status_code}: {resp.text}"

    rows = await prisma.db.litellm_teamtable.find_many(where={"team_id": team_id})
    assert len(rows) == (1 if expected_status == 200 else 0)


# ---------------------------------------------------------------------------
# _check_org_team_limits — budget / tpm / rpm structurally unreachable
# (org_table.litellm_budget_table is None at guard time). Pin the
# no-op behavior so a future change that flips include_budget_table=True
# turns these into reds.
# ---------------------------------------------------------------------------

_ORG_BUDGET_DEAD_SCENARIOS = [
    (
        "org_budget/over_max_budget_unenforced",
        {"max_budget": 100, "tpm_limit": None, "rpm_limit": None},
        {"max_budget": 999_999},
    ),
    (
        "org_tpm/over_unenforced",
        {"max_budget": None, "tpm_limit": 100, "rpm_limit": None},
        {"tpm_limit": 999_999},
    ),
    (
        "org_rpm/over_unenforced",
        {"max_budget": None, "tpm_limit": None, "rpm_limit": 100},
        {"rpm_limit": 999_999},
    ),
]


@pytest.mark.parametrize(
    "org_budget,body_extras",
    [(b, c) for (_id, b, c) in _ORG_BUDGET_DEAD_SCENARIOS],
    ids=[s[0] for s in _ORG_BUDGET_DEAD_SCENARIOS],
)
async def test_check_org_team_limits_budget_dead_code_pin(
    org_budget,
    body_extras: Dict[str, Any],
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = await create_scratch_org(prisma, scratch.prefix, **org_budget)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    team_id = scratch.tag("team")
    resp = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "team_alias": scratch.prefix,
            "organization_id": org_id,
            **body_extras,
        },
    )
    assert resp.status_code == 200, resp.text
    rows = await prisma.db.litellm_teamtable.find_many(where={"team_id": team_id})
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# _check_user_team_limits — fires for standalone (no-org) teams created by
# a non-admin caller. Each guard reads from user_api_key_dict / user_obj.
# ---------------------------------------------------------------------------

_USER_LIMIT_SCENARIOS = [
    (
        "user_max_budget/within",
        {"max_budget": 100.0, "models": [], "tpm_limit": None, "rpm_limit": None},
        {"max_budget": 50.0},
        200,
    ),
    (
        "user_max_budget/over",
        {"max_budget": 100.0, "models": [], "tpm_limit": None, "rpm_limit": None},
        {"max_budget": 1000.0},
        400,
    ),
    (
        "user_models/subset",
        {"max_budget": None, "models": ["m-a"], "tpm_limit": None, "rpm_limit": None},
        {"models": ["m-a"]},
        200,
    ),
    (
        "user_models/superset_rejected",
        {"max_budget": None, "models": ["m-a"], "tpm_limit": None, "rpm_limit": None},
        {"models": ["m-a", "m-b"]},
        400,
    ),
    (
        "user_tpm/within",
        {"max_budget": None, "models": [], "tpm_limit": 1000, "rpm_limit": None},
        {"tpm_limit": 500},
        200,
    ),
    (
        "user_tpm/over",
        {"max_budget": None, "models": [], "tpm_limit": 1000, "rpm_limit": None},
        {"tpm_limit": 2000},
        400,
    ),
    (
        "user_rpm/within",
        {"max_budget": None, "models": [], "tpm_limit": None, "rpm_limit": 100},
        {"rpm_limit": 50},
        200,
    ),
    (
        "user_rpm/over",
        {"max_budget": None, "models": [], "tpm_limit": None, "rpm_limit": 100},
        {"rpm_limit": 250},
        400,
    ),
]


@pytest.mark.parametrize(
    "actor_caps,body_extras,expected_status",
    [(b, c, d) for (_id, b, c, d) in _USER_LIMIT_SCENARIOS],
    ids=[s[0] for s in _USER_LIMIT_SCENARIOS],
)
async def test_check_user_team_limits(
    actor_caps,
    body_extras: Dict[str, Any],
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
):
    caller = await _seed_scratch_actor_with_caps(prisma, scratch.prefix, **actor_caps)
    team_id = scratch.tag("team")
    resp = await proxy_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {caller}"},
        json={
            "team_id": team_id,
            "team_alias": scratch.prefix,
            # Standalone team — no organization_id, so user-limit guard fires.
            **body_extras,
        },
    )
    assert (
        resp.status_code == expected_status
    ), f"caps={actor_caps} body={body_extras} → {resp.status_code}: {resp.text}"

    rows = await prisma.db.litellm_teamtable.find_many(where={"team_id": team_id})
    assert len(rows) == (1 if expected_status == 200 else 0)


# ---------------------------------------------------------------------------
# /team/update path — budget authority.
#
# The caller's PERSONAL limits are never applied on update (that compared the
# wrong thing). But raising a team's spend ceiling is reserved for proxy admins:
# a team admin may keep or LOWER the budget, only a proxy admin may RAISE it.
# _check_user_team_limits() only runs on /team/new.
# ---------------------------------------------------------------------------


async def test_team_admin_raise_budget_blocked(proxy_client, prisma, scratch):
    """A team admin cannot raise the team's budget; the block is NOT based on
    their personal budget (which here is higher than the requested value)."""
    caller_cleartext = await _seed_scratch_actor_with_caps(
        prisma,
        scratch.prefix,
        max_budget=100000.0,  # generous personal budget; must not matter
    )
    creator_user_id = f"{scratch.prefix}-team-creator"
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[creator_user_id],
        max_budget=50.0,
    )
    # Raise the team budget 50 -> 999 as a team admin.
    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller_cleartext}"},
        json={"team_id": team_id, "max_budget": 999.0},
    )
    assert resp.status_code == 403, resp.text

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert row.max_budget == 50.0, "team budget must not change on a blocked raise"


async def test_team_admin_lower_budget_allowed(proxy_client, prisma, scratch):
    """A team admin may freely lower (or keep) the team's budget."""
    caller_cleartext = await _seed_scratch_actor_with_caps(
        prisma,
        scratch.prefix,
        max_budget=10.0,  # below both the old and new team budget; must not matter
    )
    creator_user_id = f"{scratch.prefix}-team-creator"
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[creator_user_id],
        max_budget=500.0,
    )
    # Lower the team budget 500 -> 300 as a team admin.
    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller_cleartext}"},
        json={"team_id": team_id, "max_budget": 300.0},
    )
    assert resp.status_code == 200, resp.text

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert row.max_budget == 300.0, "team admin should be able to lower the budget"


async def test_proxy_admin_raise_budget_allowed(proxy_client, prisma, scratch):
    """A proxy admin may raise a team's budget."""
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[f"{scratch.prefix}-team-creator"],
        max_budget=50.0,
    )
    # MASTER_KEY acts as proxy admin.
    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"team_id": team_id, "max_budget": 999.0},
    )
    assert resp.status_code == 200, resp.text

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert row.max_budget == 999.0, "proxy admin should be able to raise the budget"


async def test_team_admin_remove_budget_cap_blocked(proxy_client, prisma, scratch):
    """A team admin cannot strip the team's cap (max_budget=null); removing the
    ceiling is the strongest possible raise -> proxy-admin only."""
    caller_cleartext = await _seed_scratch_actor_with_caps(
        prisma, scratch.prefix, max_budget=100000.0
    )
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[f"{scratch.prefix}-team-creator"],
        max_budget=50.0,
    )
    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller_cleartext}"},
        json={"team_id": team_id, "max_budget": None},
    )
    assert resp.status_code == 403, resp.text

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert row.max_budget == 50.0, "team budget cap must not be removed by a team admin"


async def test_proxy_admin_remove_budget_cap_allowed(proxy_client, prisma, scratch):
    """A proxy admin may remove a team's cap (max_budget=null)."""
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[f"{scratch.prefix}-team-creator"],
        max_budget=50.0,
    )
    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"team_id": team_id, "max_budget": None},
    )
    assert resp.status_code == 200, resp.text

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert row.max_budget is None, "proxy admin should be able to remove the cap"
