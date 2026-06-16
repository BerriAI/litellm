"""
Tests for Router._backfill_cost_fields_from_canonical.

Background: the dashboard /model/new form exposes only input/output cost
fields. Without backfill, deployments added via the dashboard end up
registered under their UUID with cache_*_input_token_cost = None, and
the cost calculator's custom-pricing path silently drops cache token
charges. The backfill copies missing fields from
litellm.model_cost[bare_model_name] for known upstream models, while
preserving any value the user supplied explicitly.

This file pins three behaviors:
  1. Known model + no user cache rates  → backfilled from canonical
  2. Known model + explicit user cache rates → user values win
  3. Unknown model (no static entry)    → no backfill, fields stay None
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router

KNOWN_MODEL = "claude-haiku-4-5-20251001"
UNKNOWN_MODEL = "company-private/in-house-llm-not-in-static-map"


@pytest.fixture(autouse=True)
def _ensure_local_model_cost_map(monkeypatch):
    """Use bundled JSON deterministically — never depend on a live fetch."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _build_router(deployment_uuid: str, model: str, extra_params: dict | None = None):
    """Build a single-deployment Router whose litellm_params mimics what
    dashboard /model/new produces — only input/output rates, plus any
    overrides callers want to add via ``extra_params``."""
    litellm_params: dict = {
        "model": model,
        "custom_llm_provider": "anthropic",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
    }
    if extra_params:
        litellm_params.update(extra_params)
    return Router(
        model_list=[
            {
                "model_name": "alias-for-test",
                "litellm_params": litellm_params,
                "model_info": {"id": deployment_uuid},
            }
        ]
    )


def test_known_model_backfills_missing_cache_fields():
    """Dashboard added a known-upstream model with only input/output rates.
    The UUID entry should be backfilled from the static map so cache
    pricing applies on the cost-calc UUID path."""
    deployment_uuid = "backfill-known-model-uuid"
    litellm.model_cost.pop(deployment_uuid, None)

    _build_router(deployment_uuid=deployment_uuid, model=KNOWN_MODEL)

    entry = litellm.model_cost.get(deployment_uuid, {})
    assert (
        entry.get("input_cost_per_token") == 1e-6
    ), "user-supplied input_cost_per_token should be preserved"
    assert (
        entry.get("output_cost_per_token") == 5e-6
    ), "user-supplied output_cost_per_token should be preserved"

    canonical = litellm.model_cost[KNOWN_MODEL]
    assert (
        entry.get("cache_read_input_token_cost")
        == canonical["cache_read_input_token_cost"]
    ), "cache_read_input_token_cost should be backfilled from canonical entry"
    assert (
        entry.get("cache_creation_input_token_cost")
        == canonical["cache_creation_input_token_cost"]
    ), "cache_creation_input_token_cost should be backfilled from canonical entry"


def test_user_supplied_cache_rates_override_backfill():
    """If the user explicitly sets cache rates in litellm_params (e.g.
    they negotiated a discounted gateway rate), those values must win
    over the canonical static values."""
    deployment_uuid = "backfill-user-override-uuid"
    litellm.model_cost.pop(deployment_uuid, None)

    user_cache_read_rate = 2e-7  # 2x the canonical Anthropic rate
    user_cache_create_rate = 9.99e-7

    _build_router(
        deployment_uuid=deployment_uuid,
        model=KNOWN_MODEL,
        extra_params={
            "cache_read_input_token_cost": user_cache_read_rate,
            "cache_creation_input_token_cost": user_cache_create_rate,
        },
    )

    entry = litellm.model_cost.get(deployment_uuid, {})
    assert (
        entry.get("cache_read_input_token_cost") == user_cache_read_rate
    ), "user-supplied cache_read_input_token_cost must win over canonical"
    assert (
        entry.get("cache_creation_input_token_cost") == user_cache_create_rate
    ), "user-supplied cache_creation_input_token_cost must win over canonical"

    canonical = litellm.model_cost[KNOWN_MODEL]
    assert (
        entry.get("cache_read_input_token_cost")
        != canonical["cache_read_input_token_cost"]
    ), "test geometry weak — user value happens to equal canonical"


def test_unknown_model_no_backfill():
    """Custom in-house model with no static map entry: backfill must
    not invent values — cache fields stay None / absent."""
    deployment_uuid = "backfill-unknown-model-uuid"
    litellm.model_cost.pop(deployment_uuid, None)
    litellm.model_cost.pop(UNKNOWN_MODEL, None)
    assert (
        UNKNOWN_MODEL not in litellm.model_cost
    ), "test setup expects UNKNOWN_MODEL to be absent from static map"

    _build_router(deployment_uuid=deployment_uuid, model=UNKNOWN_MODEL)

    entry = litellm.model_cost.get(deployment_uuid, {})
    assert (
        entry.get("input_cost_per_token") == 1e-6
    ), "user-supplied fields still register normally for unknown models"
    assert (
        entry.get("cache_read_input_token_cost") is None
    ), "cache_read must not be invented when no canonical entry exists"
    assert (
        entry.get("cache_creation_input_token_cost") is None
    ), "cache_creation must not be invented when no canonical entry exists"


def test_backfill_skips_when_canonical_lacks_field():
    """If the canonical static entry itself lacks a given field, the
    backfill leaves the UUID entry unchanged for that field (i.e. no
    None-for-None copy that would mask absence)."""
    deployment_uuid = "backfill-canonical-partial-uuid"
    litellm.model_cost.pop(deployment_uuid, None)

    _build_router(deployment_uuid=deployment_uuid, model=KNOWN_MODEL)

    entry = litellm.model_cost.get(deployment_uuid, {})
    # Pick a field the canonical entry definitely doesn't have. The
    # bundled JSON's Anthropic claude-haiku-4-5-20251001 entry doesn't
    # carry input_cost_per_audio_token, for instance.
    canonical = litellm.model_cost[KNOWN_MODEL]
    if canonical.get("input_cost_per_audio_token") is None:
        assert (
            "input_cost_per_audio_token" not in entry
            or entry["input_cost_per_audio_token"] is None
        ), "backfill must not copy None-valued fields from canonical"


def test_backfill_cost_fields_from_canonical_direct_call():
    """Direct unit test on the helper so the AST-based router coverage
    gate (tests/code_coverage_tests/router_code_coverage.py) can see
    ``_backfill_cost_fields_from_canonical`` being called by name. The
    higher-level ``_build_router`` path invokes it via Router internals,
    which the coverage gate cannot trace through indirection.
    """
    router = Router(model_list=[])

    model_info_dict: dict = {
        "id": "direct-unit-uuid",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
    }
    litellm_params: dict = {
        "model": KNOWN_MODEL,
        "custom_llm_provider": "anthropic",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
    }

    router._backfill_cost_fields_from_canonical(
        model_info_dict=model_info_dict,
        litellm_params=litellm_params,
    )

    canonical = litellm.model_cost[KNOWN_MODEL]
    # User-supplied values preserved.
    assert model_info_dict["input_cost_per_token"] == 1e-6
    assert model_info_dict["output_cost_per_token"] == 5e-6
    # Missing cache fields populated from canonical (when canonical has them).
    for field in (
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
    ):
        if field in canonical:
            assert (
                model_info_dict.get(field) == canonical[field]
            ), f"{field} should have been backfilled from canonical entry"


def test_provider_prefixed_model_resolves_to_canonical():
    """Regression for greptile P1 on #30383: dashboard /model/new stores
    the user-facing model as `anthropic/claude-haiku-4-5-20251001` in
    `litellm_params.model`, but `litellm.model_cost` keys are bare
    (no `provider/` prefix). A raw `model_cost.get(litellm_params.model)`
    misses; backfill must fall back to `get_llm_provider` to strip the
    prefix, otherwise the entire fix silently no-ops for the most
    common user-facing model-name shape.
    """
    deployment_uuid = "backfill-prefixed-uuid"
    litellm.model_cost.pop(deployment_uuid, None)

    # Verify the premise: prefixed lookup misses, bare lookup hits
    assert litellm.model_cost.get(f"anthropic/{KNOWN_MODEL}") is None, (
        "test premise broken — litellm.model_cost now contains "
        "`anthropic/...` keys; review whether backfill stripping is "
        "still required"
    )
    assert (
        litellm.model_cost.get(KNOWN_MODEL) is not None
    ), "bundled JSON should contain bare KNOWN_MODEL"

    # litellm_params.model carries the user-facing `provider/model` form
    _build_router(deployment_uuid=deployment_uuid, model=f"anthropic/{KNOWN_MODEL}")

    entry = litellm.model_cost.get(deployment_uuid, {})
    canonical = litellm.model_cost[KNOWN_MODEL]
    # Cache rates must have been backfilled from the bare canonical entry
    backfilled_at_least_one = False
    for field in (
        "cache_creation_input_token_cost",
        "cache_read_input_token_cost",
    ):
        if canonical.get(field) is not None:
            assert entry.get(field) == canonical[field], (
                f"{field} must be backfilled from canonical bare entry "
                f"when litellm_params.model has a provider/ prefix; "
                f"got entry={entry.get(field)!r}, canonical={canonical[field]!r}"
            )
            backfilled_at_least_one = True
    assert backfilled_at_least_one, (
        "test setup expected canonical to carry at least one cache rate "
        "to backfill; bundled JSON may have changed"
    )
