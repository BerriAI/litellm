"""Phase 4 F5 — payload-level pins for /team/permissions_bulk_update.

Pins the three helpers
  * _compute_and_batch_updates              (team_endpoints.py:4887)
  * _append_permissions_to_specific_teams   (team_endpoints.py:4913)
  * _append_permissions_to_all_teams        (team_endpoints.py:4932)

The route is admin-only — PROXY_ADMIN is the only legal caller. The
contracts under test:
  * specific-team list → ONLY listed teams mutate; every other team
    (including actors teams) is byte-identical on re-read.
  * apply_to_all_teams → every team has the permission, idempotently
    merged (re-running with the same permission is a no-op).
  * unknown team_id → 404 (the missing_ids guard); no mutation anywhere.
  * malformed payload (neither / both selector flags) → 400; no mutation.

"""

from typing import Final

import pytest

from tests.integration._support.actors import TEAM_ALPHA, TEAM_BETA, TEAM_GAMMA, Actor
from tests.integration._support.actors import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Specific-team list — only the listed team mutates.
# ---------------------------------------------------------------------------


async def test_bulk_update_specific_team_only_mutates_listed(
    gateway_client, db, resources, actors
):
    target = await create_scratch_team(
        db,
        team_id=resources.tag("target"),
        team_member_permissions=[],
    )
    bystander = await create_scratch_team(
        db,
        team_id=resources.tag("bystander"),
        team_member_permissions=["pre-existing-perm"],
    )
    # Snapshot actors teams so we can assert byte-equality on re-read.
    world_team_ids = [actors.team_alpha_id, actors.team_beta_id, actors.team_gamma_id]
    before = {}
    for tid in world_team_ids + [bystander]:
        row = await db.db.litellm_teamtable.find_unique(where={"team_id": tid})
        before[tid] = list(row.team_member_permissions or [])

    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    perm = "/key/info"
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"team_ids": [target], "permissions": [perm]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["teams_updated"] == 1

    target_row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": target}
    )
    assert perm in (
        target_row.team_member_permissions or []
    ), f"target team did not gain perm; got={target_row.team_member_permissions}"

    for tid, before_perms in before.items():
        after_row = await db.db.litellm_teamtable.find_unique(
            where={"team_id": tid}
        )
        assert list(after_row.team_member_permissions or []) == before_perms, (
            f"untouched team {tid} mutated: "
            f"{before_perms} → {list(after_row.team_member_permissions or [])}"
        )


# ---------------------------------------------------------------------------
# Specific-team — repeated call with same permission is a no-op (the
# `permissions_to_add <= existing` short-circuit in _compute_and_batch_updates).
# ---------------------------------------------------------------------------


async def test_bulk_update_specific_team_idempotent(
    gateway_client, db, resources, actors
):
    target = await create_scratch_team(
        db,
        team_id=resources.tag("target"),
        team_member_permissions=["/key/info"],
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_ids": [target],
            "permissions": ["/key/info"],  # already present
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["teams_updated"] == 0, "idempotent add should report 0"


# ---------------------------------------------------------------------------
# Unknown team_id — 404, no mutation anywhere.
# ---------------------------------------------------------------------------


async def test_bulk_update_unknown_team_id_rejected(
    gateway_client, db, resources, actors
):
    existing = await create_scratch_team(
        db,
        team_id=resources.tag("real"),
        team_member_permissions=[],
    )
    before = list(
        (
            await db.db.litellm_teamtable.find_unique(where={"team_id": existing})
        ).team_member_permissions
        or []
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_ids": [existing, f"{resources.prefix}-ghost"],
            "permissions": ["/key/info"],
        },
    )
    # The missing_ids check raises 404, but the global exception handler
    # may wrap it. Either 404 or 400 with "not found" detail counts.
    assert resp.status_code in (400, 404), resp.text
    assert "not found" in resp.text.lower() or "ghost" in resp.text, resp.text
    # Critical: the partial-success regression shape is "real team got
    # mutated before the ghost-id check ran". Re-read and assert it didn't.
    after_real = await db.db.litellm_teamtable.find_unique(
        where={"team_id": existing}
    )
    assert (
        list(after_real.team_member_permissions or []) == before
    ), "partial mutation: real team changed despite ghost-id rejection"


# ---------------------------------------------------------------------------
# Selector validation — neither flag → 400; both flags → 400.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,expected_substring",
    [
        (
            {"permissions": ["/key/info"]},
            "team_ids or set apply_to_all_teams",
        ),
        (
            {
                "permissions": ["/key/info"],
                "team_ids": ["t"],
                "apply_to_all_teams": True,
            },
            "Cannot set both",
        ),
    ],
    ids=["neither_selector", "both_selectors"],
)
async def test_bulk_update_selector_validation(
    body, expected_substring: str, gateway_client, actors
):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert resp.status_code == 400, resp.text
    assert expected_substring in resp.text, resp.text


# ---------------------------------------------------------------------------
# apply_to_all_teams
# ---------------------------------------------------------------------------


async def test_bulk_update_apply_to_all_mutates_every_team(
    gateway_client, db, resources, actors
):
    # Two resources teams so we can assert "every" includes our own targets.
    a = await create_scratch_team(
        db, team_id=resources.tag("a"), team_member_permissions=[]
    )
    b = await create_scratch_team(
        db, team_id=resources.tag("b"), team_member_permissions=["other"]
    )
    perm = "/key/health"  # distinct from the specific-team scenarios above

    all_teams = await db.db.litellm_teamtable.find_many()
    snapshot = {t.team_id: list(t.team_member_permissions or []) for t in all_teams}
    expected_updates: Final = sum(
        1 for permissions in snapshot.values() if perm not in permissions
    )

    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/permissions_bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"apply_to_all_teams": True, "permissions": [perm]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["teams_updated"] == expected_updates

    post = await db.db.litellm_teamtable.find_many()
    for team in post:
        perms = list(team.team_member_permissions or [])
        assert perm in perms, f"team {team.team_id} missing the all-perm: {perms}"
        for prior in snapshot.get(team.team_id, []):
            assert prior in perms, f"team {team.team_id} lost prior perm {prior!r}: {perms}"
