from typing import Final
"""Phase 4 F7 — coverage gap-closer scenarios picked from the un-covered
ranges left after F1–F6 landed.

Each scenario cites the file:line range it pins (PR4.M3 requirement).
Scenarios that would only pad the count without pinning observable
behavior are excluded — see `phase4-plan.md` §4-F7 anti-patterns.

Ranges addressed here:
  * team_endpoints.py 455–521  (_check_team_model_specific_limits body)
  * team_endpoints.py 538–566  (_check_team_rpm_tpm_limits body)
  * team_endpoints.py 696–731  (_check_org_team_limits guaranteed-throughput
                                  branch — currently dead because the call
                                  site doesn't include_budget_table=True,
                                  but the inner `find_many` + helper-loop
                                  runs regardless, covering the lines)
  * key_management_endpoints.py 1147–1156 (_check_project_key_limits
                                  project-not-found 404)
  * key_management_endpoints.py 3007–3018 (validate_key_team_change
                                  team-admin-accepts branch)
"""

import uuid
from typing import Any, Dict, Optional

import pytest
from prisma import Json

from litellm.proxy.utils import hash_token

from tests.integration._support.actors import TEAM_ALPHA, Actor
from tests.integration._support.actors import create_scratch_org, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# /team/new — guaranteed_throughput route into _check_org_team_limits's
# throughput branch (lines 696–731), which then calls
# check_org_team_model_specific_limits → _check_team_model_specific_limits
# (lines 442–525). The org metadata supplies the per-model cap; sibling
# teams are loaded from DB to drive the allocation sum.
# ---------------------------------------------------------------------------


async def test_org_team_guaranteed_throughput_model_over_bound_rejected(
    gateway_client, db, resources, actors
):
    org_id = await create_scratch_org(
        db,
        resources.prefix,
        models=["gpt-4"],
        metadata={"model_rpm_limit": {"gpt-4": 30}},
    )
    # A sibling team in the same org already burning 20 rpm on gpt-4 —
    # forces _check_team_model_specific_limits's `model_specific_rpm_limit`
    # accumulator to actually accumulate (covers lines 468–478).
    await create_scratch_team(
        db,
        team_id=resources.tag("sibling"),
        organization_id=org_id,
        metadata={"model_rpm_limit": {"gpt-4": 20}},
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    team_id = resources.tag("new")
    body = {
        "team_id": team_id,
        "team_alias": team_id,
        "organization_id": org_id,
        "models": ["gpt-4"],
        "model_rpm_limit": {"gpt-4": 100},
        "rpm_limit_type": "guaranteed_throughput",
    }
    resp = await gateway_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    # 20 (sibling) + 100 (new) > 30 (org cap) → guard fires.
    assert resp.status_code == 400, resp.text
    assert "RPM" in resp.text, resp.text
    rows = await db.db.litellm_teamtable.find_many(where={"team_id": team_id})
    assert rows == []


async def test_org_team_guaranteed_throughput_model_tpm_over_bound_rejected(
    gateway_client, db, resources, actors
):
    """Mirror of the rpm scenario but for the model_tpm side — pins
    _check_team_model_specific_limits's tpm branch (lines 503–521) which
    the rpm scenario doesn't exercise."""
    org_id = await create_scratch_org(
        db,
        resources.prefix,
        models=["gpt-4"],
        metadata={"model_tpm_limit": {"gpt-4": 500}},
    )
    await create_scratch_team(
        db,
        team_id=resources.tag("sibling"),
        organization_id=org_id,
        metadata={"model_tpm_limit": {"gpt-4": 200}},
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    team_id = resources.tag("new")
    body = {
        "team_id": team_id,
        "team_alias": team_id,
        "organization_id": org_id,
        "models": ["gpt-4"],
        "model_tpm_limit": {"gpt-4": 5000},
        "tpm_limit_type": "guaranteed_throughput",
    }
    resp = await gateway_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert resp.status_code == 400, resp.text
    assert "TPM" in resp.text, resp.text
    rows = await db.db.litellm_teamtable.find_many(where={"team_id": team_id})
    assert rows == []



@pytest.mark.parametrize("requested_tpm,expected_status", [(50, 400), (20, 200)], ids=["over", "within"])
async def test_org_team_guaranteed_throughput_counts_sibling_allocations(
    requested_tpm: int, expected_status: int, gateway_client, db, resources, actors
):
    org_id = await create_scratch_org(db, resources.prefix, models=["m"], tpm_limit=120)
    await create_scratch_team(
        db,
        team_id=resources.tag("sibling"),
        organization_id=org_id,
        models=["m"],
        tpm_limit=100,
    )
    team_id = resources.tag("new")
    resp = await gateway_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "team_id": team_id,
            "team_alias": team_id,
            "organization_id": org_id,
            "models": ["m"],
            "tpm_limit": requested_tpm,
            "tpm_limit_type": "guaranteed_throughput",
        },
    )
    assert resp.status_code == expected_status, resp.text
    rows = await db.db.litellm_teamtable.find_many(where={"team_id": team_id})
    assert len(rows) == (1 if expected_status == 200 else 0)
    if expected_status == 400:
        assert "Allocated TPM limit=100" in resp.text, resp.text

# ---------------------------------------------------------------------------
# /key/generate — _check_project_key_limits project-not-found branch
# (lines 1147–1156). Hitting it requires a project_id that doesn't resolve;
# get_project_object returns None, the handler raises 404.
# ---------------------------------------------------------------------------


async def test_key_generate_with_unknown_project_id_rejected(
    gateway_client, db, resources, actors
):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    body = {
        "key_alias": resources.prefix,
        "project_id": f"{resources.prefix}-ghost-project",
    }
    resp = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    # The unknown-project guard inside _check_project_key_limits raises 404.
    # The route's exception handler may wrap it; pin both shapes.
    assert resp.status_code in (400, 404), resp.text
    assert "project" in resp.text.lower() or "not found" in resp.text.lower(), resp.text
    rows = await db.db.litellm_verificationtoken.find_many(
        where={"key_alias": resources.prefix}
    )
    assert rows == [], "rejected key leaked a row"


# ---------------------------------------------------------------------------
# /key/update — validate_key_team_change team-admin-accepts branch
# (line 3011). PROXY_ADMIN covers line 3006; a non-admin team admin of the
# target team covers 3007–3011. Owner stays a member of the destination
# team to clear the membership guard first.
# ---------------------------------------------------------------------------


async def _seed_key_for_relocation(
    db, scratch_prefix: str, *, user_id: str, team_id: str
) -> str:
    cleartext = "sk-" + uuid.uuid4().hex
    await db.db.litellm_verificationtoken.create(
        data={
            "token": hash_token(cleartext),
            "key_alias": f"{scratch_prefix}-key",
            "key_name": f"{scratch_prefix}-key",
            "user_id": user_id,
            "team_id": team_id,
            "models": [],
        }
    )
    return cleartext


async def test_team_new_with_team_member_budget_creates_budget_row(
    gateway_client, db, resources, actors
):
    """/team/new with `team_member_budget` routes through
    TeamMemberBudgetHandler.create_team_member_budget_table (lines 196–248).
    Observable end-state: a litellm_budgettable row is created and the
    team's metadata.team_member_budget_id points at it."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    team_id = resources.tag("team-with-mbudget")
    resp = await gateway_client.post(
        "/team/new",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "team_alias": resources.tag("alias"),
            "team_member_budget": 10.0,
            "team_member_rpm_limit": 100,
            "team_member_tpm_limit": 1000,
        },
    )
    assert resp.status_code == 200, resp.text
    team_row = await db.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert team_row is not None
    budget_id = (team_row.metadata or {}).get("team_member_budget_id")
    assert (
        budget_id is not None
    ), f"team_member_budget_id not stamped on team metadata: {team_row.metadata!r}"

    # The handler writes the per-member budget row under a non-resources id
    # pattern (`team-<alias>-budget-<uuid>`), so the prefix sweep can't
    # reclaim it. Cleanup must run even when a downstream assertion fires;
    # otherwise orphan rows accumulate across CI re-runs.
    try:
        budget_row = await db.db.litellm_budgettable.find_unique(
            where={"budget_id": budget_id}
        )
        assert budget_row is not None, "team_member_budget row was not created"
        assert budget_row.max_budget == 10.0
        assert budget_row.rpm_limit == 100
        assert budget_row.tpm_limit == 1000
    finally:
        await db.db.litellm_teamtable.update(
            where={"team_id": team_id},
            data={"metadata": Json({})},
        )
        await db.db.litellm_budgettable.delete(where={"budget_id": budget_id})


async def test_team_update_team_member_budget_upserts(
    gateway_client, db, resources, actors
):
    team_id = await create_scratch_team(db, team_id=resources.tag("team"))
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "team_member_budget": 5.0,
        },
    )
    assert resp.status_code == 200, resp.text
    team_row = await db.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert team_row is not None
    budget_id = (team_row.metadata or {}).get("team_member_budget_id")
    assert budget_id is not None, "team_member_budget_id not upserted"
    budget_row = await db.db.litellm_budgettable.find_unique(where={"budget_id": budget_id})
    assert budget_row is not None and budget_row.max_budget == 5.0, budget_row

    try:
        budget_row: Final = await db.db.litellm_budgettable.find_unique(
            where={"budget_id": budget_id}
        )
        assert budget_row is not None, "team_member_budget row was not created"
        assert budget_row.max_budget == 5.0
    finally:
        await db.db.litellm_teamtable.update(
            where={"team_id": team_id},
            data={"metadata": Json({})},
        )
        await db.db.litellm_budgettable.delete(where={"budget_id": budget_id})


async def test_key_team_change_accepted_by_target_team_admin(
    gateway_client, db, resources, actors
):
    """Caller is admin of the destination team AND the key's owner — the
    common_key_access_checks gate requires user_id-match for non-proxy-admin
    callers, so the team-admin-accepts branch of validate_key_team_change
    can only be reached when the team admin is also the key holder.
    Hits validate_key_team_change line 3007–3011."""
    actor_user_id = f"{resources.prefix}-self-admin"
    actor_cleartext = "sk-" + uuid.uuid4().hex
    await db.db.litellm_usertable.create(
        data={"user_id": actor_user_id, "user_role": "internal_user"}
    )
    await db.db.litellm_verificationtoken.create(
        data={
            "token": hash_token(actor_cleartext),
            "key_alias": f"{resources.prefix}-actor-key",
            "key_name": f"{resources.prefix}-actor-key",
            "user_id": actor_user_id,
            "models": [],
            "allowed_routes": ["/key/update"],
        }
    )
    source_team = await create_scratch_team(
        db,
        team_id=resources.tag("source"),
        admin_user_ids=[actor_user_id],
    )
    target_team = await create_scratch_team(
        db,
        team_id=resources.tag("target"),
        admin_user_ids=[actor_user_id],
    )
    key_cleartext = await _seed_key_for_relocation(
        db, resources.prefix, user_id=actor_user_id, team_id=source_team
    )
    resp = await gateway_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {actor_cleartext}"},
        json={"key": key_cleartext, "team_id": target_team},
    )
    assert resp.status_code == 200, resp.text
    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(key_cleartext)}
    )
    assert row is not None
    assert row.team_id == target_team, "key did not move under team-admin initiator"
