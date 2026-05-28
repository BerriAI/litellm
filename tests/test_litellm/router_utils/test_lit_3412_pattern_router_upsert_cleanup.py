"""
Regression tests for LIT-3412 / BerriAI/litellm#29064.

`Router.upsert_deployment` did not evict the prior deployment dict from
`PatternMatchRouter.patterns` when a wildcard deployment was updated, so the
pattern dict accumulated stale entries and the router silently round-robined
between the corrected and the broken deployment.

These tests pin the cleanup behaviour for:

* the non-team `pattern_router` (provider-default wildcard),
* the team-scoped `team_pattern_routers[team_id]` (team BYOK wildcard),
* the team-id mutation case (deployment moves from team A to team B on upsert),
* the empty-pattern pruning behaviour of
  `PatternMatchRouter.remove_deployment_by_id`,
* and that ``litellm_params`` actually carry the corrected value after upsert
  (i.e. the pattern list no longer round-robins between old and new).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from litellm import Router
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
from litellm.types.router import Deployment, LiteLLM_Params


# ---------------------------------------------------------------------------
# PatternMatchRouter.remove_deployment_by_id unit tests
# ---------------------------------------------------------------------------


def test_remove_deployment_by_id_evicts_matching_entries():
    pr = PatternMatchRouter()
    pr.add_pattern(
        "openai/*",
        {
            "model_info": {"id": "dep-1"},
            "litellm_params": {"model": "openai/openai/*"},
        },
    )
    pr.add_pattern(
        "openai/*",
        {
            "model_info": {"id": "dep-2"},
            "litellm_params": {"model": "openai/*"},
        },
    )

    removed = pr.remove_deployment_by_id("dep-1")
    assert removed == 1

    regex = pr._pattern_to_regex("openai/*")
    assert len(pr.patterns[regex]) == 1
    assert pr.patterns[regex][0]["model_info"]["id"] == "dep-2"


def test_remove_deployment_by_id_prunes_empty_patterns():
    pr = PatternMatchRouter()
    pr.add_pattern(
        "openai/*",
        {
            "model_info": {"id": "dep-1"},
            "litellm_params": {"model": "openai/*"},
        },
    )

    removed = pr.remove_deployment_by_id("dep-1")
    assert removed == 1
    # The now-empty regex bucket must be dropped — otherwise routing keeps an
    # orphan key in `patterns` and `sorted_patterns` still iterates it.
    assert pr.patterns == {}


def test_remove_deployment_by_id_no_match_is_noop():
    pr = PatternMatchRouter()
    pr.add_pattern(
        "openai/*",
        {
            "model_info": {"id": "dep-1"},
            "litellm_params": {"model": "openai/*"},
        },
    )

    removed = pr.remove_deployment_by_id("does-not-exist")
    assert removed == 0
    regex = pr._pattern_to_regex("openai/*")
    assert len(pr.patterns[regex]) == 1


def test_remove_deployment_by_id_empty_id_is_noop():
    """A missing/blank model_id must short-circuit instead of accidentally
    matching deployments with no `model_info.id`."""
    pr = PatternMatchRouter()
    pr.add_pattern(
        "openai/*",
        {"model_info": {}, "litellm_params": {"model": "openai/*"}},
    )
    assert pr.remove_deployment_by_id("") == 0
    assert pr.remove_deployment_by_id(None) == 0  # type: ignore[arg-type]


def test_remove_deployment_by_id_handles_missing_model_info():
    pr = PatternMatchRouter()
    # entry without a model_info dict at all must not crash the loop
    pr.add_pattern("openai/*", {"litellm_params": {"model": "openai/*"}})
    pr.add_pattern(
        "openai/*",
        {
            "model_info": {"id": "dep-2"},
            "litellm_params": {"model": "openai/*"},
        },
    )
    assert pr.remove_deployment_by_id("dep-2") == 1
    regex = pr._pattern_to_regex("openai/*")
    assert len(pr.patterns[regex]) == 1
    assert pr.patterns[regex][0]["litellm_params"]["model"] == "openai/*"


# ---------------------------------------------------------------------------
# Router.upsert_deployment integration tests
# ---------------------------------------------------------------------------


def _pattern_entries(pr: PatternMatchRouter) -> list:
    """Flatten every deployment dict stored across every pattern."""
    out = []
    for lst in pr.patterns.values():
        out.extend(lst)
    return out


def test_upsert_deployment_evicts_stale_non_team_wildcard_entry():
    """LIT-3412 primary repro — non-team wildcard."""
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/openai/*",  # intentionally double-prefixed
                    "api_key": "fake",
                },
                "model_info": {"id": "deploy-A"},
            }
        ]
    )

    # Sanity: one entry, currently broken
    entries = _pattern_entries(router.pattern_router)
    assert len(entries) == 1
    assert entries[0]["litellm_params"]["model"] == "openai/openai/*"

    fixed = Deployment(
        model_name="openai/*",
        litellm_params=LiteLLM_Params(model="openai/*", api_key="fake"),
        model_info={"id": "deploy-A"},
    )
    router.upsert_deployment(fixed)

    entries = _pattern_entries(router.pattern_router)
    # Before fix: 2 entries — the round-robin bug.
    # After fix: exactly 1 entry, holding the corrected litellm_params.model.
    assert len(entries) == 1
    assert entries[0]["model_info"]["id"] == "deploy-A"
    assert entries[0]["litellm_params"]["model"] == "openai/*"


def test_upsert_deployment_evicts_stale_team_wildcard_entry():
    """LIT-3412 — team BYOK wildcard (`team_pattern_routers` bucket)."""
    router = Router(
        model_list=[
            {
                "model_name": "model-team-A-internal",
                "litellm_params": {
                    "model": "openai/openai/*",
                    "api_key": "fake",
                },
                "model_info": {
                    "id": "deploy-T",
                    "team_id": "team-1",
                    "team_public_model_name": "openai/*",
                },
            }
        ]
    )
    assert "team-1" in router.team_pattern_routers
    entries = _pattern_entries(router.team_pattern_routers["team-1"])
    assert len(entries) == 1
    assert entries[0]["litellm_params"]["model"] == "openai/openai/*"

    fixed = Deployment(
        model_name="model-team-A-internal",
        litellm_params=LiteLLM_Params(model="openai/*", api_key="fake"),
        model_info={
            "id": "deploy-T",
            "team_id": "team-1",
            "team_public_model_name": "openai/*",
        },
    )
    router.upsert_deployment(fixed)

    entries = _pattern_entries(router.team_pattern_routers["team-1"])
    assert len(entries) == 1
    assert entries[0]["model_info"]["id"] == "deploy-T"
    assert entries[0]["litellm_params"]["model"] == "openai/*"


def test_upsert_deployment_clears_old_team_when_team_id_changes():
    """If the deployment moves from team A to team B during an upsert, the
    stale entry in team A's pattern router must also be evicted — otherwise
    team A would silently keep routing through this deployment."""
    router = Router(
        model_list=[
            {
                "model_name": "model-team-A-internal",
                "litellm_params": {"model": "openai/*", "api_key": "fake"},
                "model_info": {
                    "id": "deploy-T",
                    "team_id": "team-A",
                    "team_public_model_name": "openai/*",
                },
            }
        ]
    )
    assert len(_pattern_entries(router.team_pattern_routers["team-A"])) == 1

    fixed = Deployment(
        model_name="model-team-B-internal",
        litellm_params=LiteLLM_Params(model="openai/*", api_key="fake"),
        model_info={
            "id": "deploy-T",
            "team_id": "team-B",
            "team_public_model_name": "openai/*",
        },
    )
    router.upsert_deployment(fixed)

    # Team A bucket must now be empty (pattern dict empty OR bucket removed).
    if "team-A" in router.team_pattern_routers:
        assert _pattern_entries(router.team_pattern_routers["team-A"]) == []
    # Team B bucket carries exactly one entry — the new deployment.
    assert "team-B" in router.team_pattern_routers
    entries_b = _pattern_entries(router.team_pattern_routers["team-B"])
    assert len(entries_b) == 1
    assert entries_b[0]["model_info"]["id"] == "deploy-T"
    assert entries_b[0]["model_info"]["team_id"] == "team-B"


def test_upsert_deployment_non_wildcard_does_not_touch_pattern_router():
    """Upserts on non-wildcard deployments must remain byte-identical: nothing
    enters the pattern router before or after the upsert."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                "model_info": {"id": "deploy-N"},
            }
        ]
    )
    assert router.pattern_router.patterns == {}

    fixed = Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(
            model="openai/gpt-4o-2024-08-06", api_key="fake"
        ),
        model_info={"id": "deploy-N"},
    )
    router.upsert_deployment(fixed)

    assert router.pattern_router.patterns == {}
    assert len(router.model_list) == 1
    assert (
        router.model_list[0]["litellm_params"]["model"]
        == "openai/gpt-4o-2024-08-06"
    )


def test_upsert_deployment_routing_no_longer_round_robins_stale_entry():
    """End-to-end behavioural test — after upsert, every routing decision must
    resolve through the corrected `litellm_params.model`. Before the fix, this
    set contained both `openai/openai/gpt-4o` (stale) and `openai/gpt-4o`."""
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/openai/*",
                    "api_key": "fake",
                },
                "model_info": {"id": "deploy-X"},
            }
        ]
    )

    fixed = Deployment(
        model_name="openai/*",
        litellm_params=LiteLLM_Params(model="openai/*", api_key="fake"),
        model_info={"id": "deploy-X"},
    )
    router.upsert_deployment(fixed)

    seen = set()
    for _ in range(20):
        dep = router.get_available_deployment(model="openai/gpt-4o")
        seen.add(dep["litellm_params"]["model"])

    assert seen == {"openai/gpt-4o"}, seen


def test_evict_stale_pattern_entries_direct_call():
    """Directly invoke :meth:`Router._evict_stale_pattern_entries` to pin its
    contract independent of `upsert_deployment`. Two stale entries seeded into
    `pattern_router` + `team_pattern_routers["team-A"]` for the same
    `model_info.id` must both be evicted after one helper call."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                "model_info": {"id": "stub"},
            }
        ]
    )

    # Seed stale dicts that would normally be planted by an earlier upsert.
    router.pattern_router.add_pattern(
        "openai/*",
        {
            "model_info": {"id": "stale-dep"},
            "litellm_params": {"model": "openai/openai/*"},
        },
    )
    router.team_pattern_routers["team-A"] = PatternMatchRouter()
    router.team_pattern_routers["team-A"].add_pattern(
        "openai/*",
        {
            "model_info": {"id": "stale-dep", "team_id": "team-A"},
            "litellm_params": {"model": "openai/openai/*"},
        },
    )

    prev = Deployment(
        model_name="model-internal",
        litellm_params=LiteLLM_Params(model="openai/openai/*", api_key="fake"),
        model_info={"id": "stale-dep", "team_id": "team-A"},
    )

    router._evict_stale_pattern_entries(
        deployment_id="stale-dep",
        previous_deployment=prev,
    )

    assert _pattern_entries(router.pattern_router) == []
    assert _pattern_entries(router.team_pattern_routers["team-A"]) == []


def test_evict_stale_pattern_entries_no_id_is_safe_noop():
    """Calling :meth:`Router._evict_stale_pattern_entries` with a blank
    deployment_id (e.g. legacy deployment without a populated model_info.id)
    must not touch any pattern bucket."""
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "fake"},
                "model_info": {"id": "keep-me"},
            }
        ]
    )
    snapshot_before = list(_pattern_entries(router.pattern_router))

    router._evict_stale_pattern_entries(
        deployment_id="",
        previous_deployment=None,
    )

    assert _pattern_entries(router.pattern_router) == snapshot_before


# ---------------------------------------------------------------------------
# Router.delete_deployment regression tests (Greptile follow-up)
#
# `delete_deployment` previously suffered from the same stale-pattern-entry
# problem as `upsert_deployment`: wildcard deployments removed via
# `DELETE /model/{id}` stayed registered in `pattern_router` /
# `team_pattern_routers` and continued to receive traffic until the proxy
# was restarted.
# ---------------------------------------------------------------------------


def test_delete_deployment_evicts_non_team_wildcard_entry():
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "fake"},
                "model_info": {"id": "deploy-A"},
            }
        ]
    )
    assert len(_pattern_entries(router.pattern_router)) == 1

    router.delete_deployment(id="deploy-A")

    assert router.delete_deployment(id="deploy-A") is None  # idempotent
    assert _pattern_entries(router.pattern_router) == []
    assert router.model_list == []


def test_delete_deployment_evicts_team_wildcard_entry():
    router = Router(
        model_list=[
            {
                "model_name": "model-team-A-internal",
                "litellm_params": {"model": "openai/*", "api_key": "fake"},
                "model_info": {
                    "id": "deploy-T",
                    "team_id": "team-1",
                    "team_public_model_name": "openai/*",
                },
            }
        ]
    )
    assert len(_pattern_entries(router.team_pattern_routers["team-1"])) == 1

    router.delete_deployment(id="deploy-T")

    entries = (
        _pattern_entries(router.team_pattern_routers["team-1"])
        if "team-1" in router.team_pattern_routers
        else []
    )
    assert entries == []
    assert router.model_list == []


def test_delete_deployment_non_wildcard_does_not_touch_pattern_router():
    """Non-wildcard delete must remain a byte-identical no-op on the pattern
    router buckets."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                "model_info": {"id": "deploy-N"},
            }
        ]
    )
    assert router.pattern_router.patterns == {}

    router.delete_deployment(id="deploy-N")

    assert router.pattern_router.patterns == {}
    assert router.model_list == []


def test_evict_stale_pattern_entries_cleans_all_team_buckets():
    """Veria follow-up — when stale entries for the same deployment id exist
    in multiple team-pattern-router buckets (e.g. legacy pollution from
    pre-fix upserts that moved the deployment across teams), one call to
    `_evict_stale_pattern_entries` must clean every bucket, not just the
    prior team's bucket.

    Without the unconditional sweep, the deployment could keep receiving
    traffic on a team it no longer belongs to (and to which it can be
    deleted from cleanly), and Veria flagged the original scoped-cleanup as
    Medium risk.
    """
    router = Router(
        model_list=[
            {
                "model_name": "stub",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                "model_info": {"id": "stub-keep"},
            }
        ]
    )

    # Seed three team buckets with stale entries for the same deployment id.
    for tid in ("team-A", "team-B", "team-C"):
        router.team_pattern_routers[tid] = PatternMatchRouter()
        router.team_pattern_routers[tid].add_pattern(
            "openai/*",
            {
                "model_info": {"id": "shared-stale", "team_id": tid},
                "litellm_params": {"model": "openai/openai/*"},
            },
        )
    # And one unrelated entry that MUST be preserved.
    router.team_pattern_routers["team-A"].add_pattern(
        "anthropic/*",
        {
            "model_info": {"id": "unrelated", "team_id": "team-A"},
            "litellm_params": {"model": "anthropic/*"},
        },
    )

    prev = Deployment(
        model_name="model-internal",
        litellm_params=LiteLLM_Params(model="openai/openai/*", api_key="fake"),
        # Even though we only pass a hint for team-A, all three buckets
        # must be cleaned.
        model_info={"id": "shared-stale", "team_id": "team-A"},
    )

    router._evict_stale_pattern_entries(
        deployment_id="shared-stale", previous_deployment=prev
    )

    for tid in ("team-A", "team-B", "team-C"):
        bucket = router.team_pattern_routers.get(tid)
        if bucket is None:
            continue
        remaining = [e["model_info"]["id"] for e in _pattern_entries(bucket)]
        assert "shared-stale" not in remaining, (tid, remaining)

    # Unrelated entry must survive.
    bucket_a_remaining = [
        e["model_info"]["id"] for e in _pattern_entries(router.team_pattern_routers["team-A"])
    ]
    assert "unrelated" in bucket_a_remaining

