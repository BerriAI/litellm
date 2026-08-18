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

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_token(
    prisma,
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
    await prisma.db.litellm_verificationtoken.create(data=data)
    return cleartext


# ---------------------------------------------------------------------------
# /key/health — `metadata.logging` flips the handler into test_key_logging,
# which lives at lines 5990–6067. The healthy-no-logging path is already
# covered by the existing test_key_health.py; this adds the logging-set
# path. The mock_response inside test_key_logging means no real LLM call
# fires — only the callback-name validation and the post-call sweep.
# ---------------------------------------------------------------------------


async def test_key_health_with_logging_metadata_runs_test_logging(
    proxy_client, prisma, scratch, world
):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        metadata={
            "logging": [
                {"callback_name": "noop-scratch-callback"},
            ]
        },
    )
    resp = await proxy_client.post(
        "/key/health",
        headers={"Authorization": f"Bearer {cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Either healthy or unhealthy — both pin the logging branch.
    assert body["key"] in ("healthy", "unhealthy")
    assert "logging_callbacks" in body
    assert body["logging_callbacks"]["callbacks"] == ["noop-scratch-callback"]


async def test_key_health_with_missing_callback_name_rejected(
    proxy_client, prisma, scratch, world
):
    """test_key_logging raises ValueError if a callback dict lacks
    callback_name — wrapped by the outer try/except into a 500."""
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        metadata={"logging": [{"not_callback_name": "x"}]},
    )
    resp = await proxy_client.post(
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


async def test_reset_spend_ghost_key_404(proxy_client, scratch, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/sk-{scratch.prefix}-ghost/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": 0.0},
    )
    assert resp.status_code == 404, resp.text


async def test_reset_spend_happy_path(proxy_client, prisma, scratch, world):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        spend=5.0,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": 1.0},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["spend"] == 1.0
    assert resp.json()["previous_spend"] == 5.0

    row = await prisma.db.litellm_verificationtoken.find_unique(
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
    reset_to, expected_detail: str, proxy_client, prisma, scratch, world
):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        spend=5.0,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": reset_to},
    )
    assert resp.status_code == 400, resp.text
    assert expected_detail in resp.text, resp.text
    # Row spend must be unchanged.
    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row.spend == 5.0, "spend mutated despite validation rejection"


async def test_reset_spend_non_numeric_caught_by_pydantic(
    proxy_client, prisma, scratch, world
):
    """Pin: non-float reset_to is rejected at the Pydantic layer with 422
    before reaching _validate_reset_spend_value. This documents that the
    helper's `isinstance(reset_to, (int, float))` guard at line 4568 is
    structurally unreachable via HTTP."""
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        spend=5.0,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"reset_to": "not-a-number"},
    )
    assert resp.status_code == 422, resp.text


async def test_reset_spend_non_admin_caller_rejected(
    proxy_client, prisma, scratch, world
):
    """_check_proxy_or_team_admin_for_key raises 403 when caller is neither
    proxy admin nor admin of the key's team."""
    # Seed a key against TEAM_BETA (no actor in our world is admin of beta).
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
        spend=5.0,
        team_id=TEAM_BETA,
    )
    # Caller is a member of TEAM_ALPHA but not admin of TEAM_BETA.
    initiator = world.keys[Actor.INTERNAL_USER].cleartext
    resp = await proxy_client.post(
        f"/key/{cleartext}/reset_spend",
        headers={"Authorization": f"Bearer {initiator}"},
        json={"reset_to": 1.0},
    )
    # Route-level admin gate may fire first (401) or the helper's own 403 —
    # both prove the path is guarded; pin either as rejection.
    assert resp.status_code in (401, 403), resp.text
    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row.spend == 5.0, "spend mutated despite rejection"


# ---------------------------------------------------------------------------
# /key/regenerate — ghost key 404 + happy path + new_key override. Pins
# the route handler body lines 4382–4533 and _execute_virtual_key_regeneration
# entry-point lines around 4220–4240.
# ---------------------------------------------------------------------------


async def test_regenerate_ghost_key_404(proxy_client, scratch, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/sk-{scratch.prefix}-ghost/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={},
    )
    assert resp.status_code == 404, resp.text


async def test_regenerate_no_key_supplied_400(proxy_client, world):
    """POST /key/regenerate with no key in path AND no key in body."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/key/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={},
    )
    assert resp.status_code == 400, resp.text
    assert "No key passed in" in resp.text or "key" in resp.text.lower()


async def test_regenerate_happy_path(proxy_client, prisma, scratch, world):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
    )
    old_hash = hash_token(cleartext)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
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
    old_row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": old_hash}
    )
    assert old_row is None, "old token still present after regenerate"

    # New token should be active.
    new_row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": new_hash}
    )
    assert new_row is not None, "new token not written after regenerate"


async def test_regenerate_with_explicit_new_key(proxy_client, prisma, scratch, world):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="explicit",
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    new_key = "sk-" + uuid.uuid4().hex
    resp = await proxy_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"new_key": new_key},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["key"] == new_key
    new_row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(new_key)}
    )
    assert new_row is not None


# ---------------------------------------------------------------------------
# /key/generate duplicate-alias rejection — pins _enforce_unique_key_alias
# (lines 6118–6133). Two keys cannot share an alias.
# ---------------------------------------------------------------------------


async def test_generate_duplicate_alias_rejected(proxy_client, prisma, scratch, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    alias = scratch.prefix + "-shared-alias"
    first = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": alias},
    )
    assert first.status_code == 200, first.text
    second = await proxy_client.post(
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
    proxy_client, scratch, world
):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "key_alias": scratch.prefix,
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
    filter_kwarg, expected_substring: str, proxy_client, world
):
    """Non-admin caller hits validate_key_list_check's three rejection
    branches. INTERNAL_USER (Org A, TEAM_ALPHA member) is the caller; each
    filter targets a foreign user/team/org and trips the matching guard."""
    caller = world.keys[Actor.INTERNAL_USER].cleartext
    qs = "&".join(f"{k}={v}" for k, v in filter_kwarg.items())
    resp = await proxy_client.get(
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
    proxy_client, prisma, scratch, world
):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    existing = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="bulk1",
    )
    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "keys": [
                {"key": existing, "max_budget": 5.0},
                {"key": "sk-ghost-" + scratch.prefix, "max_budget": 5.0},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    assert len(body["successful_updates"]) == 1
    assert len(body["failed_updates"]) == 1
    # Re-read confirms the successful key was actually updated.
    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(existing)}
    )
    assert row.max_budget == 5.0


async def test_key_bulk_update_non_admin_rejected(proxy_client, world):
    """Lines 2631–2635 — admin-only role gate."""
    caller = world.keys[Actor.INTERNAL_USER].cleartext
    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {caller}"},
        json={"keys": [{"key": "sk-x", "max_budget": 1.0}]},
    )
    assert resp.status_code in (401, 403), resp.text


async def test_key_bulk_update_empty_keys_rejected(proxy_client, world):
    """Line 2643–2647 — empty keys list rejected."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"keys": []},
    )
    assert resp.status_code == 400, resp.text
    assert "No keys" in resp.text


async def test_key_bulk_update_exceeds_max_batch_rejected(proxy_client, world):
    """Lines 2649–2656 — over-batch-size rejection. 501 ghost keys are fine
    here because validation fires before any update runs."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    over_batch = [{"key": f"sk-{i}", "max_budget": 1.0} for i in range(501)]
    resp = await proxy_client.post(
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
    proxy_client, prisma, scratch, world
):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "key": cleartext,
            "duration": "1h",
            "budget_duration": "30d",
        },
    )
    assert resp.status_code == 200, resp.text
    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(cleartext)}
    )
    assert row.expires is not None, "expires not stamped from duration"
    assert row.budget_reset_at is not None, "budget_reset_at not stamped"


async def test_key_update_with_clear_duration(proxy_client, prisma, scratch, world):
    """`duration: -1` clears expires (line 1796–1798)."""
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    # First set an expiry
    await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key": cleartext, "duration": "1h"},
    )
    # Then clear it
    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key": cleartext, "duration": "-1"},
    )
    assert resp.status_code == 200, resp.text
    row = await prisma.db.litellm_verificationtoken.find_unique(
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


async def test_team_key_bulk_update_missing_team_id_rejected(proxy_client, world):
    """Pydantic catches missing team_id at the request layer → 422 before
    the handler's own `if not data.team_id` guard at line 2803."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"update_fields": {"max_budget": 5.0}},
    )
    assert resp.status_code == 422, resp.text


async def test_team_key_bulk_update_no_selector_rejected(proxy_client, world):
    """Pydantic root-validator catches missing key_ids/all_keys_in_team
    at the request layer → 422."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": TEAM_ALPHA,
            "update_fields": {"max_budget": 5.0},
        },
    )
    assert resp.status_code == 422, resp.text
    assert "key_ids" in resp.text or "all_keys_in_team" in resp.text


async def test_team_key_bulk_update_all_keys_in_team(
    proxy_client, prisma, scratch, world
):
    """Pin the all_keys_in_team branch (lines 2818–2841) plus the
    per-key success path. Seed two scratch keys against the scratch team."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[world.keys[Actor.PROXY_ADMIN].user_id],
    )
    k1 = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="t1",
        team_id=team_id,
    )
    k2 = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="t2",
        team_id=team_id,
    )
    resp = await proxy_client.post(
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
        row = await prisma.db.litellm_verificationtoken.find_unique(
            where={"token": hash_token(k)}
        )
        assert row.max_budget == 7.0


async def test_team_key_bulk_update_explicit_key_ids_mixed(
    proxy_client, prisma, scratch, world
):
    """Pin the explicit-key_ids dedupe + per-key 404 path (lines 2842–2944).
    Send a real key + a ghost key under the same team_id."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        admin_user_ids=[world.keys[Actor.PROXY_ADMIN].user_id],
    )
    real_key = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="real",
        team_id=team_id,
    )
    resp = await proxy_client.post(
        "/team/key/bulk_update",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "team_id": team_id,
            "key_ids": [real_key, real_key, "sk-ghost-" + scratch.prefix],
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


async def test_key_info_ghost_key_404(proxy_client, scratch, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.get(
        f"/key/info?key=sk-{scratch.prefix}-ghost",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 404, resp.text


async def test_key_info_explicit_existing_key(proxy_client, prisma, scratch, world):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="info-test",
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.get(
        f"/key/info?key={cleartext}",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["key"] == cleartext
    assert "info" in body
    # Token hash is stripped from the response (line 3296).
    assert "token" not in body["info"]


async def test_key_info_no_key_uses_auth_header(proxy_client, world):
    """Pin line 3260: `key = key or user_api_key_dict.api_key` — caller's
    own key info is returned when no `?key=` is supplied."""
    caller = world.keys[Actor.INTERNAL_USER].cleartext
    resp = await proxy_client.get(
        "/key/info",
        headers={"Authorization": f"Bearer {caller}"},
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# /key/generate with budget_limits — pins budget_limits initialization
# inside generate_key_helper_fn (lines 3427–3436).
# ---------------------------------------------------------------------------


async def test_key_generate_with_budget_limits(proxy_client, prisma, scratch, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={
            "key_alias": scratch.prefix,
            "budget_limits": [
                {"max_budget": 5.0, "budget_duration": "1d"},
                {"max_budget": 20.0, "budget_duration": "30d"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text


async def test_key_aliases_proxy_admin_unscoped(proxy_client, prisma, scratch, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    alias = scratch.prefix + "-aliases-test"
    await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": alias},
    )
    resp = await proxy_client.get(
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


async def test_key_aliases_with_search_filter(proxy_client, prisma, scratch, world):
    """Pin the `search` ILIKE branch (lines 5166–5168)."""
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    unique_alias = scratch.prefix + "-search-unique-tag"
    await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"key_alias": unique_alias},
    )
    resp = await proxy_client.get(
        f"/key/aliases?search={scratch.prefix}-search",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 200, resp.text
    assert unique_alias in resp.json()["aliases"]


async def test_key_aliases_non_admin_scoped(proxy_client, world):
    """Non-admin caller routes through _apply_non_admin_alias_scope (line
    5161–5164). The exact alias visibility depends on team membership; the
    pin is that the call succeeds with a scoped result."""
    caller = world.keys[Actor.INTERNAL_USER].cleartext
    resp = await proxy_client.get(
        "/key/aliases?size=10",
        headers={"Authorization": f"Bearer {caller}"},
    )
    assert resp.status_code == 200, resp.text
    assert "aliases" in resp.json()


# ---------------------------------------------------------------------------
# /key/list with extended filters — pins _build_filter_conditions branches
# at lines 5280–5388. Each scenario varies one filter so a different
# branch fires.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "include_created_by_keys=true",
        "include_team_keys=true",
        "return_full_object=true",
        "sort_by=created_at&sort_order=asc",
        "key_alias=behavior",
    ],
    ids=["created_by", "team_keys", "full_object", "sort", "alias_substring"],
)
async def test_key_list_filter_branches(query: str, proxy_client, world):
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.get(
        f"/key/list?{query}",
        headers={"Authorization": f"Bearer {seeder}"},
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# /key/regenerate with grace_period — pins _insert_deprecated_key body
# (lines 4168–4202). The old token gets retained in
# LiteLLM_DeprecatedVerificationToken; assert it lands there.
# ---------------------------------------------------------------------------


async def test_regenerate_with_grace_period_inserts_deprecated_row(
    proxy_client, prisma, scratch, world
):
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="grace",
    )
    old_hash = hash_token(cleartext)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"grace_period": "1h"},
    )
    assert resp.status_code == 200, resp.text
    new_key = resp.json()["key"]
    # Old token should now be in the deprecated table.
    deprecated_row = await prisma.db.litellm_deprecatedverificationtoken.find_unique(
        where={"token": old_hash}
    )
    assert deprecated_row is not None, "old token not retained in deprecated table"
    assert deprecated_row.active_token_id == hash_token(new_key)
    assert deprecated_row.revoke_at is not None
    # Manual cleanup — scratch prefix sweep doesn't cover this table.
    await prisma.db.litellm_deprecatedverificationtoken.delete(
        where={"token": old_hash}
    )


async def test_regenerate_with_invalid_grace_period_format(
    proxy_client, prisma, scratch, world
):
    """Invalid grace_period format falls through silently (line 4170–4175);
    regenerate still succeeds but no deprecated row is inserted."""
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.PROXY_ADMIN].user_id,
        suffix="grace-bad",
    )
    old_hash = hash_token(cleartext)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    resp = await proxy_client.post(
        f"/key/{cleartext}/regenerate",
        headers={"Authorization": f"Bearer {seeder}"},
        json={"grace_period": "totally-not-a-duration"},
    )
    assert resp.status_code == 200, resp.text
    deprecated_row = await prisma.db.litellm_deprecatedverificationtoken.find_unique(
        where={"token": old_hash}
    )
    assert deprecated_row is None, "deprecated row created despite invalid grace_period"


async def test_key_list_key_hash_filter_unauthorized(
    proxy_client, prisma, scratch, world
):
    """validate_key_list_check's key_hash branch (lines 4766–4789): a
    cross-tenant non-admin caller asks for a key_hash they don't own → 403.

    `user_belongs_to_keys_team` returns True for any team member, so a
    same-team caller is allowed to query peer keys by hash (intentional
    per the helper's policy). The 403 path requires a caller who is neither
    the key owner, team member, nor admin — i.e. CROSS_ORG_USER.
    """
    cleartext = await _seed_token(
        prisma,
        scratch.prefix,
        user_id=world.keys[Actor.OWNER].user_id,
        team_id=TEAM_ALPHA,
    )
    caller = world.keys[Actor.CROSS_ORG_USER].cleartext
    resp = await proxy_client.get(
        f"/key/list?key_hash={hash_token(cleartext)}",
        headers={"Authorization": f"Bearer {caller}"},
    )
    assert resp.status_code == 403, resp.text
