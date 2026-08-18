"""Phase 4 F1 — payload-level pins for key budget & rate-limit enforcement.

Pins the five helpers
  * _check_key_model_specific_limits           (key_management_endpoints.py:931)
  * _check_key_rpm_tpm_limits                  (key_management_endpoints.py:1016)
  * _check_team_key_limits                     (key_management_endpoints.py:1096)
  * _check_org_key_limits                      (key_management_endpoints.py:1284)
  * _check_project_key_limits                  (key_management_endpoints.py:1135)

Driven through /key/generate. PROXY_ADMIN is the caller so authz never
short-circuits the payload check — Phase 1–3 already pinned authz cleanly.

`guaranteed_throughput` on either tpm_limit_type or rpm_limit_type is the
trigger that arms `_check_team_key_limits` / `_check_org_key_limits`; without
it both helpers early-return before reading any limit. The project sub-family
covers `_check_project_key_limits`, which has no such gate.

Each scenario asserts BOTH:
  - HTTP status, and
  - the DB row state on re-read (created when accepted; absent when rejected).
A response-body check is deliberately avoided per parent plan's anti-snapshot
rule.
"""

from typing import Any, Dict

import pytest

from .actors import Actor
from .conftest import create_scratch_org, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# _check_team_key_limits — aggregate tpm/rpm guard
# ---------------------------------------------------------------------------

# (id, team_tpm, team_rpm, body_extras, expected_status, detail_substring)
_TEAM_RATE_LIMIT_SCENARIOS = [
    (
        "aggregate/tpm_within_bound",
        1000,
        None,
        {"tpm_limit": 400, "tpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "aggregate/tpm_over_bound",
        1000,
        None,
        {"tpm_limit": 2000, "tpm_limit_type": "guaranteed_throughput"},
        400,
        "TPM limit",
    ),
    (
        "aggregate/rpm_within_bound",
        None,
        100,
        {"rpm_limit": 40, "rpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "aggregate/rpm_over_bound",
        None,
        100,
        {"rpm_limit": 250, "rpm_limit_type": "guaranteed_throughput"},
        400,
        "RPM limit",
    ),
    (
        "aggregate/no_guaranteed_throughput_skips_check",
        None,
        10,
        # No *_limit_type → helper early-returns even though rpm exceeds team's.
        {"rpm_limit": 50},
        200,
        None,
    ),
]


@pytest.mark.parametrize(
    "team_tpm,team_rpm,body_extras,expected_status,detail_substring",
    [(a, b, c, d, e) for (_id, a, b, c, d, e) in _TEAM_RATE_LIMIT_SCENARIOS],
    ids=[s[0] for s in _TEAM_RATE_LIMIT_SCENARIOS],
)
async def test_check_team_key_limits_aggregate(
    team_tpm,
    team_rpm,
    body_extras: Dict[str, Any],
    expected_status: int,
    detail_substring,
    proxy_client,
    prisma,
    scratch,
    world,
):
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        tpm_limit=team_tpm,
        rpm_limit=team_rpm,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    body: Dict[str, Any] = {
        "key_alias": scratch.prefix,
        "team_id": team_id,
        **body_extras,
    }
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{body!r} → {resp.status_code}: {resp.text}"
    if detail_substring is not None:
        assert detail_substring in resp.text, resp.text

    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    if expected_status == 200:
        assert len(rows) == 1
    else:
        assert rows == [], "rejected key leaked a row"


# ---------------------------------------------------------------------------
# _check_team_key_limits — model-specific guard (via team metadata)
# ---------------------------------------------------------------------------

# Body shape: {"model_rpm_limit": {"gpt-4": <n>}, "rpm_limit_type": "guaranteed_throughput"}
# Team metadata supplies the matching per-model cap.
_TEAM_MODEL_LIMIT_SCENARIOS = [
    (
        "model_rpm/within_bound",
        {"model_rpm_limit": {"gpt-4": 30}},
        {"model_rpm_limit": {"gpt-4": 20}, "rpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "model_rpm/over_bound",
        {"model_rpm_limit": {"gpt-4": 30}},
        {"model_rpm_limit": {"gpt-4": 100}, "rpm_limit_type": "guaranteed_throughput"},
        400,
        "RPM",
    ),
    (
        "model_tpm/within_bound",
        {"model_tpm_limit": {"gpt-4": 500}},
        {"model_tpm_limit": {"gpt-4": 200}, "tpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "model_tpm/over_bound",
        {"model_tpm_limit": {"gpt-4": 500}},
        {"model_tpm_limit": {"gpt-4": 5000}, "tpm_limit_type": "guaranteed_throughput"},
        400,
        "TPM",
    ),
]


@pytest.mark.parametrize(
    "team_metadata,body_extras,expected_status,detail_substring",
    [(b, c, d, e) for (_id, b, c, d, e) in _TEAM_MODEL_LIMIT_SCENARIOS],
    ids=[s[0] for s in _TEAM_MODEL_LIMIT_SCENARIOS],
)
async def test_check_team_key_limits_model_specific(
    team_metadata,
    body_extras: Dict[str, Any],
    expected_status: int,
    detail_substring,
    proxy_client,
    prisma,
    scratch,
    world,
):
    team_id = await create_scratch_team(
        prisma,
        team_id=scratch.tag("team"),
        metadata=team_metadata,
    )
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    body: Dict[str, Any] = {
        "key_alias": scratch.prefix,
        "team_id": team_id,
        **body_extras,
    }
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{body!r} → {resp.status_code}: {resp.text}"
    if detail_substring is not None:
        assert detail_substring in resp.text, resp.text
    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    assert len(rows) == (1 if expected_status == 200 else 0)


# ---------------------------------------------------------------------------
# _check_org_key_limits — aggregate tpm/rpm guard via budget table
#
# Behavior pin, not a regression target: /key/generate (line 889) and
# /key/update (line 2310) both call `get_org_object` WITHOUT
# `include_budget_table=True`, so `org_table.litellm_budget_table` is None
# at guard time and the aggregate path silently no-ops. Over-bound payloads
# therefore land as 200, not 400. Documenting that here so a future change
# that flips include_budget_table=True or moves the guard pre-load would
# turn these into reds — exactly the regression-tripwire shape Phase 4 wants.
# The model-specific guard below DOES fire because it reads org metadata,
# which is loaded directly on the org row (no relation include needed).
# ---------------------------------------------------------------------------

_ORG_RATE_LIMIT_SCENARIOS = [
    (
        "org/tpm_within_bound",
        {"max_budget": None, "tpm_limit": 1000, "rpm_limit": None},
        {"tpm_limit": 400, "tpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "org/tpm_over_bound_unenforced_no_include_budget_table",
        {"max_budget": None, "tpm_limit": 1000, "rpm_limit": None},
        {"tpm_limit": 2000, "tpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "org/rpm_within_bound",
        {"max_budget": None, "tpm_limit": None, "rpm_limit": 100},
        {"rpm_limit": 40, "rpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "org/rpm_over_bound_unenforced_no_include_budget_table",
        {"max_budget": None, "tpm_limit": None, "rpm_limit": 100},
        {"rpm_limit": 250, "rpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "org/no_guaranteed_throughput_skips_check",
        {"max_budget": None, "tpm_limit": None, "rpm_limit": 10},
        {"rpm_limit": 50},
        200,
        None,
    ),
]


@pytest.mark.parametrize(
    "org_budget,body_extras,expected_status,detail_substring",
    [(b, c, d, e) for (_id, b, c, d, e) in _ORG_RATE_LIMIT_SCENARIOS],
    ids=[s[0] for s in _ORG_RATE_LIMIT_SCENARIOS],
)
async def test_check_org_key_limits_aggregate(
    org_budget: Dict[str, Any],
    body_extras: Dict[str, Any],
    expected_status: int,
    detail_substring,
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = await create_scratch_org(prisma, scratch.prefix, **org_budget)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    body: Dict[str, Any] = {
        "key_alias": scratch.prefix,
        "organization_id": org_id,
        **body_extras,
    }
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{body!r} → {resp.status_code}: {resp.text}"
    if detail_substring is not None:
        assert detail_substring in resp.text, resp.text
    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    assert len(rows) == (1 if expected_status == 200 else 0)


# ---------------------------------------------------------------------------
# _check_org_key_limits — model-specific guard via org metadata
# ---------------------------------------------------------------------------

_ORG_MODEL_LIMIT_SCENARIOS = [
    (
        "org_model_rpm/within_bound",
        {"model_rpm_limit": {"gpt-4": 30}},
        {"model_rpm_limit": {"gpt-4": 20}, "rpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "org_model_rpm/over_bound",
        {"model_rpm_limit": {"gpt-4": 30}},
        {"model_rpm_limit": {"gpt-4": 100}, "rpm_limit_type": "guaranteed_throughput"},
        400,
        "RPM",
    ),
    (
        "org_model_tpm/within_bound",
        {"model_tpm_limit": {"gpt-4": 500}},
        {"model_tpm_limit": {"gpt-4": 200}, "tpm_limit_type": "guaranteed_throughput"},
        200,
        None,
    ),
    (
        "org_model_tpm/over_bound",
        {"model_tpm_limit": {"gpt-4": 500}},
        {"model_tpm_limit": {"gpt-4": 5000}, "tpm_limit_type": "guaranteed_throughput"},
        400,
        "TPM",
    ),
]


@pytest.mark.parametrize(
    "org_metadata,body_extras,expected_status,detail_substring",
    [(b, c, d, e) for (_id, b, c, d, e) in _ORG_MODEL_LIMIT_SCENARIOS],
    ids=[s[0] for s in _ORG_MODEL_LIMIT_SCENARIOS],
)
async def test_check_org_key_limits_model_specific(
    org_metadata,
    body_extras: Dict[str, Any],
    expected_status: int,
    detail_substring,
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = await create_scratch_org(prisma, scratch.prefix, metadata=org_metadata)
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    body: Dict[str, Any] = {
        "key_alias": scratch.prefix,
        "organization_id": org_id,
        **body_extras,
    }
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{body!r} → {resp.status_code}: {resp.text}"
    if detail_substring is not None:
        assert detail_substring in resp.text, resp.text
    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    assert len(rows) == (1 if expected_status == 200 else 0)


# ---------------------------------------------------------------------------
# Project sub-family deferral — see plan §4-F1 ("project surface proves thin"):
# `get_project_object` requires an LiteLLM_ProjectTable seeder + cache wiring
# that the harness does not yet ship; pinning it would force a wider
# conftest change for a single-helper close-out. Tracked as "deferred" in
# the PR4.M3 follow-up box.
