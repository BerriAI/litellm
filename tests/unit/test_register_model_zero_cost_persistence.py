"""
Regression for #30198.

``register_model`` calls ``get_model_info(key)`` to fetch the existing
entry, then ``_update_dictionary`` merges its own ``value`` over it and
the result is written back into ``litellm.model_cost``.

``_get_model_info_helper`` synthesizes ``input_cost_per_token`` and
``output_cost_per_token`` as 0 when the cost keys are missing from the
raw entry (the "price unknown" and "free" cases share the same
representation). So on the SECOND ``register_model`` call against an
already-present sparse entry (e.g. router model id with only
``{"id": ..., "db_model": True}``), the synthesized zeros get written
back, and the entry flips from "no cost keys" → "cost keys = 0".

That defeats ``_is_cost_explicitly_configured`` (added in #24949), which
checks whether the cost keys are present in the raw entry — after the
write-back they are. ``_is_model_cost_zero`` then returns ``True`` and
``common_checks`` skips every tag / key / team / user / org budget check
for the group. Spend keeps recording (cost calc resolves by model name),
so the symptom is silent: requests that should 429 keep returning 200.

Tests below replicate the Router-built-twice scenario from the report
and confirm the sparse entry stays sparse.
"""

import importlib
import os
import sys
from typing import Any, Dict

import pytest


@pytest.fixture(autouse=True)
def _restore_model_cost():
    import litellm

    original = dict(litellm.model_cost)
    try:
        yield
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(original)


def _sparse_router_value(model_cost_key: str) -> Dict[str, Any]:
    # Mirrors what Router builds for a db_model deployment with no custom
    # pricing (litellm/router.py:_create_deployment).
    return {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "gpt-4o-mini",
            "custom_llm_provider": "openai",
            "api_key": "sk-test",
        },
        "model_info": {"id": model_cost_key, "db_model": True},
    }


def test_first_registration_leaves_sparse_entry_without_cost_keys():
    """First ``register_model`` call against an unknown key must NOT add
    cost keys to the entry — otherwise the very first registration would
    already poison the map."""
    import litellm

    key = "fixed-uuid-30198-first"
    litellm.model_cost.pop(key, None)

    litellm.register_model({key: {"litellm_provider": "openai"}})

    entry = litellm.model_cost.get(key, {})
    assert "input_cost_per_token" not in entry, entry
    assert "output_cost_per_token" not in entry, entry


def test_second_registration_does_not_persist_synthesized_zero_costs():
    """The #30198 bug: re-registering the same sparse entry made
    ``get_model_info`` synthesize cost = 0 and write it back. Verify the
    entry stays clean after a second pass."""
    import litellm

    key = "fixed-uuid-30198-double-register"
    litellm.model_cost.pop(key, None)

    payload = {key: {"litellm_provider": "openai"}}
    litellm.register_model(payload)
    litellm.register_model(payload)

    entry = litellm.model_cost.get(key, {})
    assert "input_cost_per_token" not in entry, (
        "second register_model() persisted a synthesized zero "
        "input_cost_per_token; this disables budget enforcement"
    )
    assert "output_cost_per_token" not in entry, (
        "second register_model() persisted a synthesized zero "
        "output_cost_per_token; this disables budget enforcement"
    )


def test_explicit_zero_cost_in_value_is_preserved():
    """If the caller actually wants the model marked free, the explicit
    zero must survive the dedup. The fix must only strip SYNTHESIZED
    zeros, not caller-provided ones."""
    import litellm

    key = "fixed-uuid-30198-explicit-zero"
    litellm.model_cost.pop(key, None)

    litellm.register_model(
        {
            key: {
                "litellm_provider": "openai",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        }
    )

    entry = litellm.model_cost[key]
    assert entry["input_cost_per_token"] == 0
    assert entry["output_cost_per_token"] == 0

    # Re-registering with the same explicit zeros must keep them.
    litellm.register_model(
        {
            key: {
                "litellm_provider": "openai",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        }
    )
    entry = litellm.model_cost[key]
    assert entry["input_cost_per_token"] == 0
    assert entry["output_cost_per_token"] == 0


def test_real_pricing_for_known_model_survives_re_registration():
    """A model with built-in pricing (e.g. gpt-4o-mini) must keep its
    real per-token rates across repeated registrations of an empty
    payload that names the same key."""
    import litellm

    base_in = litellm.model_cost["gpt-4o-mini"]["input_cost_per_token"]
    base_out = litellm.model_cost["gpt-4o-mini"]["output_cost_per_token"]
    assert base_in > 0 and base_out > 0

    litellm.register_model({"gpt-4o-mini": {"litellm_provider": "openai"}})
    litellm.register_model({"gpt-4o-mini": {"litellm_provider": "openai"}})

    assert litellm.model_cost["gpt-4o-mini"]["input_cost_per_token"] == base_in
    assert litellm.model_cost["gpt-4o-mini"]["output_cost_per_token"] == base_out


def test_router_double_init_keeps_db_model_entry_sparse():
    """End-to-end repro from the issue body: building Router twice on
    the same model_list must not flip the per-deployment entry to
    cost=0. This is the exact production symptom (#30198)."""
    import litellm
    from litellm import Router

    deployment_id = "fixed-uuid-30198-router-init"
    litellm.model_cost.pop(deployment_id, None)

    model_list = [_sparse_router_value(deployment_id)]

    Router(model_list=model_list)
    after_first = dict(litellm.model_cost.get(deployment_id, {}))

    Router(model_list=model_list)
    after_second = dict(litellm.model_cost.get(deployment_id, {}))

    # Cost keys must not appear AT ALL on a sparse db_model deployment
    # (matches the pre-bug shape) — the bug rewrites them as 0.
    for snapshot, label in (
        (after_first, "first Router()"),
        (after_second, "second Router()"),
    ):
        assert "input_cost_per_token" not in snapshot, (
            f"{label} persisted input_cost_per_token={snapshot.get('input_cost_per_token')!r} "
            f"on a sparse db_model entry; this disables budget enforcement"
        )
        assert "output_cost_per_token" not in snapshot, (
            f"{label} persisted output_cost_per_token={snapshot.get('output_cost_per_token')!r} "
            f"on a sparse db_model entry; this disables budget enforcement"
        )
