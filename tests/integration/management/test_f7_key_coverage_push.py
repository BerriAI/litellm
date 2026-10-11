from typing import Final
"""Phase 4 F7-extension — additional payload pins pushing
`key_management_endpoints.py` past the 70 % stretch.

Same pattern as `test_f7_coverage_closeout.py`: each scenario cites the
file:line range it pins and asserts observable end-state, not response-body
snapshots. Targets the largest non-deferred un-covered ranges left after
the first F7 pass:

  * 5942–6063  /key/health logging-metadata path (test_key_logging body)
  * 4630–4692  /key/reset_spend happy path + 404 + admin gate
  * 4565–4596  _validate_reset_spend_value branches
  * 4421–4476  /key/regenerate ghost-key 404 + premium gate
  * 6118–6133  _enforce_unique_key_alias duplicate-alias rejection
  * 6148–6169  validate_model_max_budget malformed payload rejection
  * 4708–4789  validate_key_list_check user/team/org/key_hash branches

Excluded: `_rotate_master_key` (lines 3997–4123) — deferred per plan §6.
"""

import uuid
from typing import Any, Dict, Optional

import pytest
from prisma import Json

from litellm.proxy.utils import hash_token

from tests.integration._support.actors import TEAM_ALPHA, TEAM_BETA, Actor
from tests.integration._support.actors import create_scratch_key, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_token(
    db,
    scratch_prefix: str,
    *,
    suffix: str = "tok",
    user_id: str,
    spend: float = 0.0,
    max_budget: Optional[float] = None,
    metadata: Optional[dict] = None,
    team_id: Optional[str] = None,
) -> str:
    cleartext = "sk-" + uuid.uuid4().hex
    data: Dict[str, Any] = {
        "token": hash_token(cleartext),
        "key_alias": f"{scratch_prefix}-{suffix}",
        "key_name": f"{scratch_prefix}-{suffix}",
        "user_id": user_id,
        "models": [],
        "spend": spend,
    }
    if max_budget is not None:
        data["max_budget"] = max_budget
    if metadata is not None:
        data["metadata"] = Json(metadata)
    if team_id is not None:
        data["team_id"] = team_id
    await db.db.litellm_verificationtoken.create(data=data)
    return cleartext


# ---------------------------------------------------------------------------
# /key/health — `metadata.logging` flips the handler into test_key_logging,
# which lives at lines 5990–6067. The healthy-no-logging path is already
# covered by the existing test_key_health.py; this adds the logging-set
# path. The mock_response inside test_key_logging means no real LLM call
# fires — only the callback-name validation and the post-call sweep.
# ---------------------------------------------------------------------------


async def test_key_health_with_logging_metadata_runs_test_logging(
    gateway_client, db, resources, actors
):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.INTERNAL_USER].user_id,
        metadata={
            "logging": [
                {"callback_name": "noop-resources-callback"},
            ]
        },
    )
    resp = await gateway_client.post(
        "/key/health",
        headers={"Authorization": f"Bearer {cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Either healthy or unhealthy — both pin the logging branch.
    assert body["key"] in ("healthy", "unhealthy")
    assert "logging_callbacks" in body
    assert body["logging_callbacks"]["callbacks"] == ["noop-resources-callback"]


async def test_key_health_with_missing_callback_name_rejected(
    gateway_client, db, resources, actors
):
    """test_key_logging raises ValueError if a callback dict lacks
    callback_name — wrapped by the outer try/except into a 500."""
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        metadata={"logging": [{"not_callback_name": "x"}]},
    )
    resp = await gateway_client.post(
        "/key/health",
        headers={"Authorization": f"Bearer {cleartext}"},
    )
    # The outer handler currently wraps the inner ValueError as a 500;
    # accept any rejection envelope so a future 400/422 conversion doesn't
    # trip this test. The named-guard substring is the real pin.
    assert resp.status_code in (400, 422, 500), resp.text
    assert "callback_name" in resp.text


# ---------------------------------------------------------------------------
# /key/reset_spend — 404 + happy path + non-admin reject. Pins
# _validate_reset_spend_value (lines 4565–4596) and
# _check_proxy_or_team_admin_for_key (lines 4536–4562).
# ---------------------------------------------------------------------------


async def test_reset_spend_ghost_key_404(gateway_client, resources, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/sk-{resources.prefix}-ghost/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": 0.0},
    )
    assert resp.status_code == 404, resp.text


async def test_reset_spend_happy_path(gateway_client, db, resources, actors):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        spend=5.0,
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": 1.0},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["spend"] == 1.0
    assert resp.json()["previous_spend"] == 5.0

    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row is not None
    assert row.spend == 1.0


# Pydantic catches non-float at the request layer (422), so the inner
# `isinstance(reset_to, (int, float))` guard at line 4568 is unreachable
# from the HTTP boundary. Only the negative + above-current-spend branches
# fire as the handler's own 400.
@pytest.mark.parametrize(
    "reset_to,expected_detail",
    [
        (-1.0, "must be >= 0"),
        (100.0, "must be <= current spend"),  # current spend = 5.0
    ],
    ids=["negative", "above_current_spend"],
)
async def test_reset_spend_validate_value_branches(
    reset_to, expected_detail: str, gateway_client, db, resources, actors
):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        spend=5.0,
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": reset_to},
    )
    assert resp.status_code == 400, resp.text
    assert expected_detail in resp.text, resp.text
    # Row spend must be unchanged.
    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row.spend == 5.0, "spend mutated despite validation rejection"


async def test_reset_spend_non_admin_caller_rejected(
    gateway_client, db, resources, actors
):
    """_check_proxy_or_team_admin_for_key raises 403 when caller is neither
    proxy admin nor admin of the key's team."""
    # Seed a key against actors.team_beta_id (no actor in our actors is admin of beta).
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.CROSS_ORG_USER].user_id,
        spend=5.0,
        team_id=actors.team_beta_id,
    )
    # Caller is a member of actors.team_alpha_id but not admin of actors.team_beta_id.
    initiator = actors.keys[Actor.INTERNAL_USER].cleartext
    resp = await gateway_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {initiator}"},
        json={"reset_to": 1.0},
    )
    # Route-level admin gate may fire first (401) or the helper's own 403 —
    # both prove the path is guarded; pin either as rejection.
    assert resp.status_code in (401, 403), resp.text
    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row.spend == 5.0, "spend mutated despite rejection"


# ---------------------------------------------------------------------------
# /key/regenerate — ghost key 404 + happy path + new_key override. Pins
# the route handler body lines 4382–4533 and _execute_virtual_key_regeneration
# entry-point lines around 4220–4240.
# ---------------------------------------------------------------------------


async def test_regenerate_ghost_key_404(gateway_client, resources, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/sk-{resources.prefix}-ghost/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={},
    )
    assert resp.status_code == 404, resp.text


async def test_regenerate_no_key_supplied_400(gateway_client, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/key/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={},
    )
    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["code"] == "400", resp.text
    assert "No key passed in." in error["message"], resp.text


async def test_regenerate_happy_path(gateway_client, db, resources, actors):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
    )
    old_hash = hash_token(cleartext)
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["key"].startswith("sk-")
    new_hash = hash_token(body["key"])
    assert new_hash != old_hash

    # Old token should no longer be in active tokens (deleted by regenerate).
    old_row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": old_hash}
    )
    assert old_row is None, "old token still present after regenerate"

    # New token should be active.
    new_row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": new_hash}
    )
    assert new_row is not None, "new token not written after regenerate"


async def test_regenerate_with_explicit_new_key(gateway_client, db, resources, actors):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="explicit",
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    new_key = "sk-" + uuid.uuid4().hex
    resp = await gateway_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"new_key": new_key},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["key"] == new_key
    new_row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(new_key)}
    )
    assert new_row is not None


# ---------------------------------------------------------------------------
# /key/generate duplicate-alias rejection — pins _enforce_unique_key_alias
# (lines 6118–6133). Two keys cannot share an alias.
# ---------------------------------------------------------------------------


async def test_generate_duplicate_alias_rejected(gateway_client, db, resources, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    alias = resources.prefix + "-shared-alias"
    first = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": alias},
    )
    assert first.status_code == 200, first.text
    second = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": alias},
    )
    assert second.status_code == 400, second.text
    assert "already exists" in second.text, second.text


# ---------------------------------------------------------------------------
# /key/generate with malformed model_max_budget → 400 from
# validate_model_max_budget (lines 6148–6169).
# ---------------------------------------------------------------------------


async def test_generate_with_invalid_model_max_budget_rejected(
    gateway_client, resources, actors
):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "key_alias": resources.prefix,
            # Wrong shape — budget_limit should be numeric; passing a dict
            # for the model value bypasses the BudgetConfig parse and trips
            # the validator's exception wrap.
            "model_max_budget": {"gpt-4": {"budget_limit": "not-a-number"}},
        },
    )
    # validate_model_max_budget raises ValueError, which the outer handler
    # currently wraps as a 500. Pin only the named-guard substring; accept
    # 400/422/500 so a future error-envelope improvement doesn't trip this.
    assert resp.status_code in (400, 422, 500), resp.text
    assert "Invalid model_max_budget" in resp.text, resp.text


# ---------------------------------------------------------------------------
# /key/list — pins validate_key_list_check (lines 4695–4790). The handler
# already runs via Phase 1–3's test_key_list.py for the PROXY_ADMIN bypass;
# this adds the non-admin user_id-mismatch + team_id-mismatch +
# organization_id-mismatch branches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_kwarg,expected_substring",
    [
        ({"user_id": "behavior-pin-proxy_admin"}, "check another user"),
        ({"team_id": "behavior-pin-team-beta"}, "check this team"),
        (
            {"organization_id": "behavior-pin-org-b"},
            "check this organization",
        ),
    ],
    ids=["user_mismatch", "team_mismatch", "org_mismatch"],
)
async def test_key_list_non_admin_authz_branches(
    filter_kwarg, expected_substring: str, gateway_client, actors
):
    """Non-admin caller hits validate_key_list_check's three rejection
    branches. INTERNAL_USER (Org A, actors.team_alpha_id member) is the caller; each
    filter targets a foreign user/team/org and trips the matching guard."""
    caller = actors.keys[Actor.INTERNAL_USER].cleartext
    qs = "&".join(f"{k}={v}" for k, v in filter_kwarg.items())
    resp = await gateway_client.get(
        f"/key/list?{qs}",
        headers={"Authorization": f"Bearer {caller}"},
    )
    assert resp.status_code == 403, resp.text
    assert expected_substring in resp.text, resp.text


# ---------------------------------------------------------------------------
# /key/bulk_update — pins the whole admin-only handler body (lines 2622–2733)
# including the per-key try/except branch (2688–2727) via a mixed batch
# of one existing key + one ghost.
# ---------------------------------------------------------------------------


async def test_key_bulk_update_mixed_success_and_failure(
    gateway_client, db, resources, actors
):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    existing = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="bulk1",
    )
    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "keys": [
                {"key": existing, "max_budget": 5.0},
                {"key": "sk-ghost-" + resources.prefix, "max_budget": 5.0},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    assert len(body["successful_updates"]) == 1
    assert len(body["failed_updates"]) == 1
    # Re-read confirms the successful key was actually updated.
    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(existing)}
    )
    assert row.max_budget == 5.0


async def test_key_bulk_update_non_admin_rejected(gateway_client, actors):
    """Lines 2631–2635 — admin-only role gate."""
    caller = actors.keys[Actor.INTERNAL_USER].cleartext
    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {caller}"},
        json={"keys": [{"key": "sk-x", "max_budget": 1.0}]},
    )
    assert resp.status_code in (401, 403), resp.text


async def test_key_bulk_update_empty_keys_rejected(gateway_client, actors):
    """Line 2643–2647 — empty keys list rejected."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"keys": []},
    )
    assert resp.status_code == 400, resp.text
    assert "No keys" in resp.text


async def test_key_bulk_update_exceeds_max_batch_rejected(gateway_client, actors):
    """Lines 2649–2656 — over-batch-size rejection. 501 ghost keys are fine
    here because validation fires before any update runs."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    over_batch = [{"key": f"sk-{i}", "max_budget": 1.0} for i in range(501)]
    resp = await gateway_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"keys": over_batch},
    )
    assert resp.status_code == 400, resp.text
    assert "500" in resp.text or "Maximum" in resp.text


# ---------------------------------------------------------------------------
# /key/update with extended fields — pins prepare_key_update_data
# branches: duration (1794–1802), budget_duration (1804–1815),
# model_max_budget validation (1838–1840), and the reserved-metadata
# immutability check (1718–1731).
# ---------------------------------------------------------------------------


async def test_key_update_with_duration_and_budget_duration(
    gateway_client, db, resources, actors
):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "key": cleartext,
            "duration": "1h",
            "budget_duration": "30d",
        },
    )
    assert resp.status_code == 200, resp.text
    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row is not None
    assert row.budget_duration == "30d"
    assert row.expires is not None
    assert row.budget_reset_at is not None
    info = await gateway_client.get(
        f"/key/info?key={hash_token(cleartext)}",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert info.status_code == 200, info.text
    assert info.json()["info"]["expires"] == row.expires.isoformat()
    assert info.json()["info"]["budget_reset_at"] == row.budget_reset_at.isoformat()


async def test_key_update_with_clear_duration(gateway_client, db, resources, actors):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    expiring = await gateway_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key": cleartext, "duration": "1h"},
    )
    assert expiring.status_code == 200, expiring.text
    expiring_row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert expiring_row.expires is not None, "duration=1h did not set expires"

    resp = await gateway_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key": cleartext, "duration": "-1"},
    )
    assert resp.status_code == 200, resp.text
    row = await db.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row.expires is None, "expires not cleared by duration=-1"


# ---------------------------------------------------------------------------
# /team/key/bulk_update — pins handler body lines 2797–2950 including:
# - missing team_id 400 (2803–2807)
# - over-batch-size 400 (2810–2816)
# - all_keys_in_team scan (2818–2841)
# - explicit key_ids dedupe (2842–2863)
# - non-admin permission gate (2866–2882)
# - per-key loop with mixed success/404 (2902–2944)
# ---------------------------------------------------------------------------


async def test_team_key_bulk_update_no_selector_rejected(gateway_client, actors):
    """Pydantic root-validator catches missing key_ids/all_keys_in_team
    at the request layer → 422."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": actors.team_alpha_id,
            "update_fields": {"max_budget": 5.0},
        },
    )
    assert resp.status_code == 422, resp.text
    assert "key_ids" in resp.text or "all_keys_in_team" in resp.text


async def test_team_key_bulk_update_all_keys_in_team(
    gateway_client, db, resources, actors
):
    """Pin the all_keys_in_team branch (lines 2818–2841) plus the
    per-key success path. Seed two resources keys against the resources team."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    team_id = await create_scratch_team(
        db,
        team_id=resources.tag("team"),
        admin_user_ids=[actors.keys[Actor.PROXY_ADMIN].user_id],
    )
    k1 = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="t1",
        team_id=team_id,
    )
    k2 = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="t2",
        team_id=team_id,
    )
    resp = await gateway_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "all_keys_in_team": True,
            "update_fields": {"max_budget": 7.0},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    # Re-read both keys; max_budget should be set.
    for k in (k1, k2):
        row = await db.db.litellm_verificationtoken.find_unique(
            where={"token": hash_token(k)}
        )
        assert row.max_budget == 7.0


async def test_team_key_bulk_update_explicit_key_ids_mixed(
    gateway_client, db, resources, actors
):
    """Pin the explicit-key_ids dedupe + per-key 404 path (lines 2842–2944).
    Send a real key + a ghost key under the same team_id."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    team_id = await create_scratch_team(
        db,
        team_id=resources.tag("team"),
        admin_user_ids=[actors.keys[Actor.PROXY_ADMIN].user_id],
    )
    real_key = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="real",
        team_id=team_id,
    )
    resp = await gateway_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "key_ids": [real_key, real_key, "sk-ghost-" + resources.prefix],
            "update_fields": {"max_budget": 9.0},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Dedupe collapses the two real_key entries → 2 unique tokens.
    assert body["total_requested"] == 2
    assert len(body["successful_updates"]) == 1
    assert len(body["failed_updates"]) == 1


# ---------------------------------------------------------------------------
# /key/aliases — pins the handler body lines 5108–5207. PROXY_ADMIN hits
# the broad-scope path; a non-admin caller hits the scoped path through
# _apply_non_admin_alias_scope.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /key/info — pins the handler body (lines 3253–3303). PROXY_ADMIN with an
# explicit ghost key 404s; with a real key 200s. Phase 1–3 covered the
# auth matrix; this adds the explicit-key path the matrix doesn't hit.
# ---------------------------------------------------------------------------


async def test_key_info_ghost_key_404(gateway_client, resources, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.get(
        f"/key/info?key=sk-{resources.prefix}-ghost",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 404, resp.text


async def test_key_info_explicit_existing_key(gateway_client, db, resources, actors):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="info-test",
    )
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.get(
        f"/key/info?key={cleartext}",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["key"] == cleartext
    assert "info" in body
    # Token hash is stripped from the response (line 3296).
    assert "token" not in body["info"]


async def test_key_info_no_key_uses_auth_header(gateway_client, actors):
    """Pin line 3260: `key = key or user_api_key_dict.api_key` — caller's
    own key info is returned when no `?key=` is supplied."""
    caller = actors.keys[Actor.INTERNAL_USER].cleartext
    resp = await gateway_client.get(
        "/key/info",
        headers={"Authorization": f"Bearer {caller}"},
    )
    assert resp.status_code == 200, resp.text
    body: Final = resp.json()
    assert body["key"] == hash_token(caller)
    assert body["info"]["user_id"] == actors.keys[Actor.INTERNAL_USER].user_id
    assert body["info"]["status"] == "active"
    assert "token" not in body["info"]


# ---------------------------------------------------------------------------
# /key/generate with budget_limits — pins budget_limits initialization
# inside generate_key_helper_fn (lines 3427–3436).
# ---------------------------------------------------------------------------


async def test_key_generate_with_budget_limits(gateway_client, db, resources, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "key_alias": resources.prefix,
            "budget_limits": [
                {"max_budget": 5.0, "budget_duration": "1d"},
                {"max_budget": 20.0, "budget_duration": "30d"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    cleartext: Final = resp.json()["key"]
    info: Final = await gateway_client.get(
        f"/key/info?key={hash_token(cleartext)}",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert info.status_code == 200, info.text
    windows: Final = info.json()["info"]["budget_limits"]
    assert [
        (window["max_budget"], window["budget_duration"]) for window in windows
    ] == [(5.0, "1d"), (20.0, "30d")]
    cleartext = resp.json()["key"]
    info = await gateway_client.get(
        f"/key/info?key={hash_token(cleartext)}",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert info.status_code == 200, info.text
    assert [
        (window["max_budget"], window["budget_duration"])
        for window in info.json()["info"]["budget_limits"]
    ] == [(5.0, "1d"), (20.0, "30d")]


async def test_key_aliases_proxy_admin_unscoped(gateway_client, db, resources, actors):
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    alias = resources.prefix + "-aliases-test"
    await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": alias},
    )
    resp = await gateway_client.get(
        "/key/aliases?size=10",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "aliases" in body
    assert "total_count" in body
    assert body["current_page"] == 1
    assert body["size"] == 10
    assert (
        alias in body["aliases"]
    ), f"newly created alias not in list: {body['aliases']}"


async def test_key_aliases_with_search_filter(gateway_client, db, resources, actors):
    """Pin the `search` ILIKE branch (lines 5166–5168)."""
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    unique_alias = resources.prefix + "-search-unique-tag"
    await gateway_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": unique_alias},
    )
    resp = await gateway_client.get(
        f"/key/aliases?search={resources.prefix}-search",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 200, resp.text
    assert unique_alias in resp.json()["aliases"]


# ---------------------------------------------------------------------------
# /key/list with extended filters — pins _build_filter_conditions branches
# at lines 5280–5388. Each scenario varies one filter so a different
# branch fires.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_name",
    [
        "created_by",
        "team_keys",
        "full_object",
        "sort",
        "alias_substring",
    ],
    ids=["created_by", "team_keys", "full_object", "sort", "alias_substring"],
)
async def test_key_list_filter_branches(filter_name: str, gateway_client, resources, actors):
    admin: Final = actors.keys[Actor.PROXY_ADMIN]
    key_alias: Final = resources.tag("filter-branch")
    cleartext: Final = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=key_alias,
        team_id=actors.team_alpha_id,
    )
    query: Final = {
        "created_by": "include_created_by_keys=true",
        "team_keys": "include_team_keys=true",
        "full_object": "return_full_object=true",
        "sort": "sort_by=created_at&sort_order=asc",
        "alias_substring": f"key_alias={key_alias[:-5]}&substring_matching=true",
    }[filter_name]
    list_query: Final = (
        query
        if filter_name == "alias_substring"
        else f"{query}&key_alias={key_alias}"
    )
    resp: Final = await gateway_client.get(
        f"/key/list?{list_query}",
        headers={"Authorization": f"Bearer {admin.cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    response_body: Final = resp.json()
    entries: Final = response_body["keys"]
    assert response_body["total_count"] == 1
    returned_hashes: Final = frozenset(
        entry["token"] if isinstance(entry, dict) else entry for entry in entries
    )
    assert returned_hashes == {hash_token(cleartext)}


# ---------------------------------------------------------------------------
# /key/regenerate with grace_period — pins _insert_deprecated_key body
# (lines 4168–4202). The old token gets retained in
# LiteLLM_DeprecatedVerificationToken; assert it lands there.
# ---------------------------------------------------------------------------


async def test_regenerate_with_grace_period_inserts_deprecated_row(
    gateway_client, db, resources, actors
):
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="grace",
    )
    old_hash = hash_token(cleartext)
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"grace_period": "1h"},
    )
    assert resp.status_code == 200, resp.text
    new_key = resp.json()["key"]
    # Old token should now be in the deprecated table.
    deprecated_row = await db.db.litellm_deprecatedverificationtoken.find_unique(
        where={"token": old_hash}
    )
    assert deprecated_row is not None, "old token not retained in deprecated table"
    assert deprecated_row.active_token_id == hash_token(new_key)
    assert deprecated_row.revoke_at is not None
    # Manual cleanup — resources prefix sweep doesn't cover this table.
    await db.db.litellm_deprecatedverificationtoken.delete(
        where={"token": old_hash}
    )


async def test_regenerate_with_invalid_grace_period_format(
    gateway_client, db, resources, actors
):
    """Invalid grace_period format falls through silently (line 4170–4175);
    regenerate still succeeds but no deprecated row is inserted."""
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.PROXY_ADMIN].user_id,
        suffix="grace-bad",
    )
    old_hash = hash_token(cleartext)
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext
    resp = await gateway_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"grace_period": "totally-not-a-duration"},
    )
    assert resp.status_code == 200, resp.text
    deprecated_row = await db.db.litellm_deprecatedverificationtoken.find_unique(
        where={"token": old_hash}
    )
    assert deprecated_row is None, "deprecated row created despite invalid grace_period"


async def test_key_list_key_hash_filter_unauthorized(
    gateway_client, db, resources, actors
):
    """validate_key_list_check's key_hash branch (lines 4766–4789): a
    cross-tenant non-admin caller asks for a key_hash they don't own → 403.

    `user_belongs_to_keys_team` returns True for any team member, so a
    same-team caller is allowed to query peer keys by hash (intentional
    per the helper's policy). The 403 path requires a caller who is neither
    the key owner, team member, nor admin — i.e. CROSS_ORG_USER.
    """
    cleartext = await _seed_token(
        db,
        resources.prefix,
        user_id=actors.keys[Actor.OWNER].user_id,
        team_id=actors.team_alpha_id,
    )
    caller = actors.keys[Actor.CROSS_ORG_USER].cleartext
    resp = await gateway_client.get(
        f"/key/list?key_hash={hash_token(cleartext)}",
        headers={"Authorization": f"Bearer {caller}"},
    )
    assert resp.status_code == 403, resp.text
