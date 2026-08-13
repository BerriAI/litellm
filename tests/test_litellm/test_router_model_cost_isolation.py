"""
Test that per-deployment custom pricing does not pollute the shared backend
model key in litellm.model_cost.

When two deployments share the same backend model (e.g. vertex_ai/gemini-2.5-flash)
and one has explicit zero-cost pricing in model_info, the other deployment
should still use the built-in pricing.
"""

import copy
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,
    reapply_runtime_model_cost_registrations,
)


def _simulate_price_data_reload(fetched_catalog):
    """Drive what a price data reload does to this process's litellm state.

    Mirrors `litellm.proxy.proxy_server._swap_in_model_cost_map`, which is the
    one place both reload paths adopt a freshly fetched catalog; that wiring is
    covered in the proxy's own tests, so these exercise the replay itself
    without dragging the proxy in. The provider model sets that helper also
    repopulates are left alone, since nothing here reads them and rebuilding
    them from a two-entry catalog would outlive the test.
    """
    litellm.model_cost = fetched_catalog
    _invalidate_model_cost_lowercase_map()
    reapply_runtime_model_cost_registrations()


def _restore_model_cost_entries(original_entries):
    for key, value in original_entries.items():
        if value is None:
            litellm.model_cost.pop(key, None)
        else:
            litellm.model_cost[key] = value
    _invalidate_model_cost_lowercase_map()


def test_should_not_pollute_shared_key_with_zero_cost_pricing():
    """
    When deployment A has input_cost_per_token=0 and deployment B has no
    custom pricing, deployment B should still report the built-in pricing
    (not zero).
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    # Grab built-in pricing before creating any router
    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input_cost = builtin_info["input_cost_per_token"]
    builtin_output_cost = builtin_info["output_cost_per_token"]

    # Sanity: built-in pricing should be non-zero for this model
    assert (
        builtin_input_cost > 0
    ), "Test requires a model with non-zero built-in pricing"
    assert (
        builtin_output_cost > 0
    ), "Test requires a model with non-zero built-in pricing"

    router = Router(
        model_list=[
            # Deployment A: explicit zero-cost pricing
            {
                "model_name": "custom-zero-cost-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-1",
                },
                "model_info": {
                    "id": "deployment-a-zero-cost",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            # Deployment B: no custom pricing, relies on built-in
            {
                "model_name": "standard-cost-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-2",
                },
                "model_info": {
                    "id": "deployment-b-builtin-cost",
                },
            },
        ],
    )

    # Deployment A: should report zero pricing via its unique model_id
    info_a = router.get_deployment_model_info(
        model_id="deployment-a-zero-cost",
        model_name=backend_model,
    )
    assert info_a is not None
    assert info_a["input_cost_per_token"] == 0.0
    assert info_a["output_cost_per_token"] == 0.0

    # Deployment B: should report built-in pricing, NOT zero
    info_b = router.get_deployment_model_info(
        model_id="deployment-b-builtin-cost",
        model_name=backend_model,
    )
    assert info_b is not None
    assert info_b["input_cost_per_token"] == builtin_input_cost, (
        f"Deployment B should use built-in input cost {builtin_input_cost}, "
        f"got {info_b['input_cost_per_token']}"
    )
    assert info_b["output_cost_per_token"] == builtin_output_cost, (
        f"Deployment B should use built-in output cost {builtin_output_cost}, "
        f"got {info_b['output_cost_per_token']}"
    )


def test_should_not_pollute_shared_key_with_custom_nonzero_pricing():
    """
    A deployment with custom (non-zero) pricing should not overwrite
    the shared backend key's built-in pricing.
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input_cost = builtin_info["input_cost_per_token"]

    router = Router(
        model_list=[
            # Deployment with custom high pricing
            {
                "model_name": "expensive-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-3",
                },
                "model_info": {
                    "id": "deployment-expensive",
                    "input_cost_per_token": 0.99,
                    "output_cost_per_token": 0.99,
                },
            },
            # Deployment relying on built-in pricing
            {
                "model_name": "standard-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-4",
                },
                "model_info": {
                    "id": "deployment-standard",
                },
            },
        ],
    )

    # Custom pricing deployment should see its custom values
    info_expensive = router.get_deployment_model_info(
        model_id="deployment-expensive",
        model_name=backend_model,
    )
    assert info_expensive is not None
    assert info_expensive["input_cost_per_token"] == 0.99
    assert info_expensive["output_cost_per_token"] == 0.99

    # Standard deployment should still see built-in pricing
    info_standard = router.get_deployment_model_info(
        model_id="deployment-standard",
        model_name=backend_model,
    )
    assert info_standard is not None
    assert info_standard["input_cost_per_token"] == builtin_input_cost, (
        f"Standard deployment should use built-in pricing {builtin_input_cost}, "
        f"got {info_standard['input_cost_per_token']}"
    )


def test_should_store_full_pricing_under_deployment_model_id():
    """
    Per-deployment pricing (including zero) should be stored and
    retrievable via the unique model_id key in litellm.model_cost.
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    router = Router(
        model_list=[
            {
                "model_name": "zero-cost-model",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-5",
                },
                "model_info": {
                    "id": "deployment-zero-check",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
        ],
    )

    # The model_id entry should exist and have the zero pricing
    entry = litellm.model_cost.get("deployment-zero-check")
    assert entry is not None, "Deployment should be registered by model_id"
    assert entry["input_cost_per_token"] == 0.0
    assert entry["output_cost_per_token"] == 0.0


def test_should_preserve_builtin_pricing_regardless_of_deployment_order():
    """
    The built-in pricing should be preserved no matter which deployment
    is processed first (zero-cost first, or standard first).
    """
    backend_model = "vertex_ai/gemini-2.5-flash"

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input_cost = builtin_info["input_cost_per_token"]
    builtin_output_cost = builtin_info["output_cost_per_token"]

    # Order 1: standard first, then zero-cost
    router1 = Router(
        model_list=[
            {
                "model_name": "standard-first",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-6",
                },
                "model_info": {"id": "order1-standard"},
            },
            {
                "model_name": "zero-cost-second",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-7",
                },
                "model_info": {
                    "id": "order1-zero",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
        ],
    )

    info_std_1 = router1.get_deployment_model_info(
        model_id="order1-standard", model_name=backend_model
    )
    assert info_std_1["input_cost_per_token"] == builtin_input_cost
    assert info_std_1["output_cost_per_token"] == builtin_output_cost

    # Order 2: zero-cost first, then standard
    router2 = Router(
        model_list=[
            {
                "model_name": "zero-cost-first",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-8",
                },
                "model_info": {
                    "id": "order2-zero",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            {
                "model_name": "standard-second",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "fake-key-9",
                },
                "model_info": {"id": "order2-standard"},
            },
        ],
    )

    info_std_2 = router2.get_deployment_model_info(
        model_id="order2-standard", model_name=backend_model
    )
    assert info_std_2["input_cost_per_token"] == builtin_input_cost, (
        f"Order should not matter. Expected {builtin_input_cost}, "
        f"got {info_std_2['input_cost_per_token']}"
    )
    assert info_std_2["output_cost_per_token"] == builtin_output_cost, (
        f"Order should not matter. Expected {builtin_output_cost}, "
        f"got {info_std_2['output_cost_per_token']}"
    )


def test_responses_prefix_stripped_alias_registered_for_model_list():
    """
    Register ``litellm.model_cost`` under the backend key with ``responses/`` and
    under the stripped key (``responses_api_bridge_check`` removes that segment).
    """
    uid = "responses-strip-alias-test-a1b2c3d4"
    Router(
        model_list=[
            {
                "model_name": "azure-responses-strip-test",
                "litellm_params": {
                    "model": "responses/gpt-strip-test-a1b2c3d4",
                    "custom_llm_provider": "azure",
                    "api_key": "fake-key-strip",
                },
                "model_info": {
                    "id": uid,
                    "supports_native_streaming": True,
                },
            }
        ],
    )
    assert "azure/responses/gpt-strip-test-a1b2c3d4" in litellm.model_cost
    assert "azure/gpt-strip-test-a1b2c3d4" in litellm.model_cost
    assert (
        litellm.model_cost["azure/gpt-strip-test-a1b2c3d4"].get(
            "supports_native_streaming"
        )
        is True
    )


def test_responses_prefix_stripped_alias_registered_for_add_deployment():
    """Dynamic ``add_deployment`` must mirror ``_create_deployment`` registration."""
    uid = "add-dep-responses-strip-e5f6a7b8"
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="dyn-responses-strip",
        litellm_params=LiteLLM_Params(
            model="responses/gpt-add-strip-e5f6a7b8",
            custom_llm_provider="azure",
            api_key="fake-key-add",
        ),
        model_info=ModelInfo(id=uid, supports_native_streaming=True),
    )
    router.add_deployment(deployment=deployment)
    assert "azure/responses/gpt-add-strip-e5f6a7b8" in litellm.model_cost
    assert "azure/gpt-add-strip-e5f6a7b8" in litellm.model_cost
    assert (
        litellm.model_cost["azure/gpt-add-strip-e5f6a7b8"].get(
            "supports_native_streaming"
        )
        is True
    )


def test_should_not_downgrade_chatgpt_shared_key_mode_with_alias_override():
    """
    ChatGPT aliases that share the same backend model should not be able to
    downgrade the shared backend key from responses -> chat during router setup.
    """
    from litellm.main import responses_api_bridge_check

    backend_model = "chatgpt/gpt-5.4"
    model_keys = {
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
        "chatgpt-shared-mode-base": copy.deepcopy(
            litellm.model_cost.get("chatgpt-shared-mode-base")
        ),
        "chatgpt-shared-mode-alias": copy.deepcopy(
            litellm.model_cost.get("chatgpt-shared-mode-alias")
        ),
    }

    try:
        backend_entry = copy.deepcopy(model_keys[backend_model]) or {}
        backend_entry["litellm_provider"] = "chatgpt"
        backend_entry["mode"] = "responses"
        litellm.model_cost[backend_model] = backend_entry
        _invalidate_model_cost_lowercase_map()

        router = Router(model_list=[])
        with patch.object(
            Router, "_add_deployment", lambda self, deployment: deployment
        ):
            router._create_deployment(
                deployment_info={},
                _model_name="chatgpt/gpt-5.4",
                _litellm_params={
                    "model": "gpt-5.4",
                    "custom_llm_provider": "chatgpt",
                },
                _model_info={
                    "id": "chatgpt-shared-mode-base",
                    "mode": "responses",
                },
            )
            router._create_deployment(
                deployment_info={},
                _model_name="chatgpt/gpt-5.4-medium",
                _litellm_params={
                    "model": "gpt-5.4",
                    "custom_llm_provider": "chatgpt",
                },
                _model_info={
                    "id": "chatgpt-shared-mode-alias",
                    "mode": "chat",
                },
            )

        assert litellm.model_cost[backend_model]["mode"] == "responses"
        assert "mode" in litellm.model_cost[backend_model]

        bridge_model_info, bridge_model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="chatgpt",
        )
        assert bridge_model == "gpt-5.4"
        assert bridge_model_info["mode"] == "responses"
    finally:
        _restore_model_cost_entries(model_keys)


def test_partial_custom_pricing_inherits_builtin_cache_pricing():
    """A deployment that overrides only input/output cost on a cache-supporting
    model must still bill cache_read and cache_creation tokens. Before the
    fix the deploy-id entry was registered with the user's two fields and
    nothing else, so the cost calculator silently billed cache tokens at 0.
    Regression for the prompt-caching cost dropout reported by the customer.
    """
    backend_model = "anthropic/claude-sonnet-4-5-20250929"
    deploy_id = "claude-deploy-partial-pricing"

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_cache_create = builtin_info["cache_creation_input_token_cost"]
    builtin_cache_read = builtin_info["cache_read_input_token_cost"]
    assert builtin_cache_create is not None and builtin_cache_create > 0
    assert builtin_cache_read is not None and builtin_cache_read > 0

    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "claude-custom",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "input_cost_per_token": 0.000003,
                        "output_cost_per_token": 0.000015,
                    },
                }
            ],
        )

        entry = litellm.model_cost[deploy_id]
        assert entry["input_cost_per_token"] == 0.000003
        assert entry["output_cost_per_token"] == 0.000015
        assert entry.get("cache_creation_input_token_cost") == builtin_cache_create
        assert entry.get("cache_read_input_token_cost") == builtin_cache_read
    finally:
        _restore_model_cost_entries(model_keys)


def test_partial_pricing_does_not_overwrite_explicit_cache_fields():
    """When the user explicitly sets cache_*_input_token_cost on a deployment,
    those values must not be replaced by the built-in fallback.
    """
    backend_model = "anthropic/claude-sonnet-4-5-20250929"
    deploy_id = "claude-deploy-explicit-cache"

    explicit_cache_create = 0.00001
    explicit_cache_read = 0.0000005
    builtin_info = litellm.get_model_info(model=backend_model)
    assert builtin_info["cache_creation_input_token_cost"] != explicit_cache_create
    assert builtin_info["cache_read_input_token_cost"] != explicit_cache_read

    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "claude-custom-explicit",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "input_cost_per_token": 0.000003,
                        "output_cost_per_token": 0.000015,
                        "cache_creation_input_token_cost": explicit_cache_create,
                        "cache_read_input_token_cost": explicit_cache_read,
                    },
                }
            ],
        )

        entry = litellm.model_cost[deploy_id]
        assert entry.get("cache_creation_input_token_cost") == explicit_cache_create
        assert entry.get("cache_read_input_token_cost") == explicit_cache_read
    finally:
        _restore_model_cost_entries(model_keys)


def test_inherit_builtin_cache_pricing_fills_only_missing_fields():
    """Direct unit test of the helper: missing cache fields are filled from the
    backend model's built-in entry, while an explicitly set cache field and the
    user's input/output pricing are left untouched.
    """
    backend_model = "anthropic/claude-sonnet-4-5-20250929"
    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_cache_create = builtin_info["cache_creation_input_token_cost"]
    builtin_cache_read = builtin_info["cache_read_input_token_cost"]
    assert builtin_cache_create is not None and builtin_cache_create > 0
    assert builtin_cache_read is not None and builtin_cache_read > 0

    explicit_cache_read = builtin_cache_read + 1
    model_info = {
        "input_cost_per_token": 0.000003,
        "cache_read_input_token_cost": explicit_cache_read,
    }

    Router._inherit_builtin_cache_pricing(
        model_info=model_info,
        backend_model=backend_model,
        custom_llm_provider="anthropic",
    )

    assert model_info["input_cost_per_token"] == 0.000003
    assert model_info["cache_read_input_token_cost"] == explicit_cache_read
    assert model_info["cache_creation_input_token_cost"] == builtin_cache_create


def test_inherit_builtin_cache_pricing_noop_for_unknown_backend():
    """No canonical entry for the backend model means the helper leaves the
    passed-in dict unchanged rather than raising.
    """
    model_info = {"input_cost_per_token": 0.000003}

    Router._inherit_builtin_cache_pricing(
        model_info=model_info,
        backend_model="this-backend-model-does-not-exist-x9y8z7",
        custom_llm_provider=None,
    )

    assert model_info == {"input_cost_per_token": 0.000003}


def test_custom_pricing_field_denylist_covers_all_builtin_pricing_fields():
    """The shared-backend-key stripping in Router relies on
    CustomPricingLiteLLMParams enumerating every per-deployment pricing field.
    If a new pricing field is added to ModelInfoBase but not mirrored here, a
    deployment override on that field leaks into the shared backend key and
    every sibling deployment reads the wrong rate (LIT-3897). This guard fails
    fast when the two drift apart.
    """
    import typing

    from litellm.types.utils import CustomPricingLiteLLMParams, ModelInfoBase

    pricing_markers = ("cost", "price", "uplift", "vector_size", "tiered_pricing")
    builtin_pricing_fields = {
        name
        for name in typing.get_type_hints(ModelInfoBase)
        if any(marker in name for marker in pricing_markers)
    }
    denylisted_fields = set(CustomPricingLiteLLMParams.model_fields.keys())

    uncovered = sorted(builtin_pricing_fields - denylisted_fields)
    assert not uncovered, (
        "ModelInfoBase pricing fields missing from CustomPricingLiteLLMParams; "
        f"these would leak into shared backend keys: {uncovered}"
    )


def test_tiered_pricing_override_isolated_from_sibling_via_model_info_lookup():
    """LIT-3897: a deployment that overrides a tiered pricing field
    (input_cost_per_token_above_272k_tokens) must not pollute the shared
    backend key, so a sibling sharing the same backend resolves its pricing
    via litellm.get_model_info (the path /model/info uses) without seeing the
    override.
    """
    backend_model = "gemini/gemini-2.5-flash"
    override = 0.000999

    builtin_info = litellm.get_model_info(model=backend_model)
    assert builtin_info.get("input_cost_per_token_above_272k_tokens") != override

    model_keys = {
        "lit3897-tiered-custom": litellm.model_cost.get("lit3897-tiered-custom"),
        "lit3897-tiered-sibling": litellm.model_cost.get("lit3897-tiered-sibling"),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "custom-priced-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-tiered-1",
                    },
                    "model_info": {
                        "id": "lit3897-tiered-custom",
                        "input_cost_per_token_above_272k_tokens": override,
                        "cache_read_input_token_cost_above_272k_tokens": override,
                    },
                },
                {
                    "model_name": "gemini-2.5-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-tiered-2",
                    },
                    "model_info": {"id": "lit3897-tiered-sibling"},
                },
            ],
        )

        shared = litellm.get_model_info(model=backend_model)
        assert shared.get("input_cost_per_token_above_272k_tokens") != override, (
            "Tiered override leaked into the shared backend key; siblings read "
            "the wrong rate via /model/info"
        )
        assert shared.get("cache_read_input_token_cost_above_272k_tokens") != override

        custom_entry = litellm.model_cost["lit3897-tiered-custom"]
        assert custom_entry["input_cost_per_token_above_272k_tokens"] == override
        assert custom_entry["cache_read_input_token_cost_above_272k_tokens"] == override
    finally:
        _restore_model_cost_entries(model_keys)


def test_custom_pricing_isolated_from_sibling_via_proxy_model_info_path():
    """LIT-3897 end to end through the proxy resolution helper: the override
    deployment reports its custom input rate while the sibling keeps the
    canonical gemini rate when /model/info resolves each deployment. Mirrors the
    ticket config where the override is set on litellm_params.
    """
    from litellm.proxy.proxy_server import _get_proxy_model_info

    backend_model = "gemini/gemini-2.5-flash"
    override_input = 5e-05
    override_output = 1e-04

    builtin_info = litellm.get_model_info(model=backend_model)
    builtin_input = builtin_info["input_cost_per_token"]
    assert builtin_input != override_input

    model_keys = {
        "lit3897-proxy-custom": litellm.model_cost.get("lit3897-proxy-custom"),
        "lit3897-proxy-sibling": litellm.model_cost.get("lit3897-proxy-sibling"),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "custom-priced-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-proxy-1",
                        "input_cost_per_token": override_input,
                        "output_cost_per_token": override_output,
                    },
                    "model_info": {"id": "lit3897-proxy-custom"},
                },
                {
                    "model_name": "gemini-2.5-flash",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-proxy-2",
                    },
                    "model_info": {"id": "lit3897-proxy-sibling"},
                },
            ],
        )

        resolved = {
            m["model_name"]: _get_proxy_model_info(model=copy.deepcopy(m))[
                "model_info"
            ]["input_cost_per_token"]
            for m in router.model_list
        }

        assert resolved["custom-priced-flash"] == override_input
        assert resolved["gemini-2.5-flash"] == builtin_input
        assert resolved["gemini-2.5-flash"] != resolved["custom-priced-flash"]
    finally:
        _restore_model_cost_entries(model_keys)


def test_custom_model_info_metadata_not_leaked_to_shared_backend_key():
    """LIT-4544: two deployments share the same backend model but carry
    different custom model_info (arbitrary keys, access_via_team_ids, ids).
    None of that per-deployment metadata may land on the shared backend key in
    litellm.model_cost (served raw by /public/litellm_model_cost_map);
    before the fix it was merged last-write-wins so values flipped randomly.
    """
    backend_model = "openai/gpt-4o-mini"
    shared_keys = ("gpt-4o-mini", backend_model)
    leak_fields = ("id", "additionalProp1", "access_via_team_ids", "db_model")

    model_keys = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (*shared_keys, "lit4544-deploy-a", "lit4544-deploy-b")
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "alias-unrestricted",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-a",
                    },
                    "model_info": {
                        "id": "lit4544-deploy-a",
                        "additionalProp1": {"restricted": False, "model_location": "EU"},
                    },
                },
                {
                    "model_name": "alias-restricted",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key-b",
                    },
                    "model_info": {
                        "id": "lit4544-deploy-b",
                        "additionalProp1": {"restricted": True, "model_location": "US"},
                        "access_via_team_ids": ["team-b-only"],
                    },
                },
            ],
        )

        for shared_key in shared_keys:
            shared_entry = litellm.model_cost.get(shared_key) or {}
            leaked = [field for field in leak_fields if field in shared_entry]
            assert not leaked, (
                f"per-deployment metadata {leaked} leaked onto shared key "
                f"{shared_key}: {shared_entry}"
            )

        entry_a = litellm.model_cost["lit4544-deploy-a"]
        assert entry_a["additionalProp1"] == {"restricted": False, "model_location": "EU"}
        entry_b = litellm.model_cost["lit4544-deploy-b"]
        assert entry_b["additionalProp1"] == {"restricted": True, "model_location": "US"}
        assert entry_b["access_via_team_ids"] == ["team-b-only"]
    finally:
        _restore_model_cost_entries(model_keys)


def test_add_deployment_does_not_leak_custom_metadata_to_shared_backend_key():
    """LIT-4544 dynamic path: deployments added at runtime (e.g. loaded from
    the DB every scheduler cycle) must not re-pollute the shared backend key
    with per-deployment metadata either.
    """
    backend_model = "openai/gpt-4o-mini"
    shared_keys = ("gpt-4o-mini", backend_model)
    deploy_id = "lit4544-add-deployment"

    model_keys = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (*shared_keys, deploy_id)
    }
    try:
        router = Router(model_list=[])
        router.add_deployment(
            deployment=Deployment(
                model_name="alias-dynamic",
                litellm_params=LiteLLM_Params(
                    model=backend_model,
                    api_key="fake-key-dynamic",
                ),
                model_info=ModelInfo(
                    id=deploy_id,
                    additionalProp1={"restricted": True},
                    access_via_team_ids=["team-dynamic"],
                ),
            )
        )

        for shared_key in shared_keys:
            shared_entry = litellm.model_cost.get(shared_key) or {}
            leaked = [
                field
                for field in ("id", "additionalProp1", "access_via_team_ids", "db_model")
                if field in shared_entry
            ]
            assert not leaked, (
                f"per-deployment metadata {leaked} leaked onto shared key "
                f"{shared_key}: {shared_entry}"
            )

        assert litellm.model_cost[deploy_id]["access_via_team_ids"] == ["team-dynamic"]
    finally:
        _restore_model_cost_entries(model_keys)


def test_shared_backend_model_info_keeps_schema_fields_and_drops_the_rest():
    """Unit test of the whitelist helper: cost-map schema fields survive,
    custom pricing overrides and per-deployment metadata do not.
    """
    from litellm.types.utils import shared_backend_model_info

    filtered = shared_backend_model_info(
        {
            "mode": "chat",
            "litellm_provider": "openai",
            "max_tokens": 128000,
            "supports_vision": True,
            "supported_endpoints": ["/v1/responses"],
            "use_openai_responses_path": True,
            "input_cost_per_token": 0.99,
            "output_cost_per_token": 0.99,
            "id": "deploy-a",
            "db_model": False,
            "access_via_team_ids": ["team-a"],
            "additionalProp1": {"restricted": True},
            "base_model": "gpt-4o-mini",
        }
    )

    assert filtered == {
        "mode": "chat",
        "litellm_provider": "openai",
        "max_tokens": 128000,
        "supports_vision": True,
        "supported_endpoints": ["/v1/responses"],
        "use_openai_responses_path": True,
    }


def test_capability_flags_propagate_from_deployment_model_info_to_shared_key():
    """Backend-model capability facts (supported_endpoints,
    use_openai_responses_path) declared in a deployment's model_info must reach
    the shared backend key: the Bedrock Mantle routing gates read them raw off
    litellm.model_cost and document proxy model_info as an override path for
    models missing from the built-in cost map.
    """
    from litellm.llms.bedrock_mantle.common_utils import (
        mantle_base_segment,
        mantle_supports_responses,
    )

    bare_model = "somelab.lit4544-unmapped-model"
    backend_model = f"bedrock_mantle/{bare_model}"
    deploy_id = "lit4544-mantle-deploy"

    model_keys = {
        key: copy.deepcopy(litellm.model_cost.get(key))
        for key in (bare_model, backend_model, deploy_id)
    }
    try:
        Router(
            model_list=[
                {
                    "model_name": "mantle-alias",
                    "litellm_params": {
                        "model": backend_model,
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "supported_endpoints": ["/v1/responses"],
                        "use_openai_responses_path": True,
                    },
                },
            ],
        )

        shared_entry = litellm.model_cost.get(backend_model) or {}
        assert shared_entry.get("supported_endpoints") == ["/v1/responses"]
        assert shared_entry.get("use_openai_responses_path") is True
        assert "id" not in shared_entry
        assert mantle_supports_responses(bare_model, litellm.model_cost) is True
        assert mantle_base_segment(bare_model, litellm.model_cost) == "openai/v1"
    finally:
        _restore_model_cost_entries(model_keys)


def test_wildcard_zero_cost_request_does_not_poison_named_deployment_pricing():
    """LIT-3991 end to end: a proxy has a named text-embedding-3-small
    deployment relying on built-in pricing plus an ``openai/*`` wildcard with
    explicit zero pricing. One embedding call routed through the wildcard must
    not clobber the shared ``openai/text-embedding-3-small`` pricing; requests
    to the named deployment afterwards must still cost non-zero.
    """
    shared_key = "openai/text-embedding-3-small"
    model_keys = {
        shared_key: copy.deepcopy(litellm.model_cost.get(shared_key)),
        "text-embedding-3-small": copy.deepcopy(
            litellm.model_cost.get("text-embedding-3-small")
        ),
        "openai/*": copy.deepcopy(litellm.model_cost.get("openai/*")),
        "lit3991-named": litellm.model_cost.get("lit3991-named"),
        "lit3991-wildcard": litellm.model_cost.get("lit3991-wildcard"),
    }
    builtin_input_cost = litellm.get_model_info(model=shared_key)[
        "input_cost_per_token"
    ]
    assert builtin_input_cost > 0

    try:
        router = Router(
            model_list=[
                {
                    "model_name": "text-embedding-3-small",
                    "litellm_params": {
                        "model": "openai/text-embedding-3-small",
                        "api_key": "fake-key-named",
                    },
                    "model_info": {"id": "lit3991-named"},
                },
                {
                    "model_name": "openai/*",
                    "litellm_params": {
                        "model": "openai/*",
                        "api_key": "fake-key-wildcard",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "lit3991-wildcard"},
                },
            ],
        )

        router.embedding(
            model="openai/text-embedding-3-small",
            input=["hello"],
            mock_response=[0.1, 0.2],
        )

        assert (
            litellm.get_model_info(model=shared_key)["input_cost_per_token"]
            == builtin_input_cost
        ), (
            "one call through the zero-cost wildcard poisoned the shared "
            f"{shared_key} pricing for the named deployment"
        )

        named_response = router.embedding(
            model="text-embedding-3-small",
            input=["hello"],
            mock_response=[0.1, 0.2],
        )
        named_cost = litellm.completion_cost(
            completion_response=named_response, call_type="embedding"
        )
        assert named_cost == pytest.approx(10 * builtin_input_cost)
    finally:
        _restore_model_cost_entries(model_keys)


def test_price_data_reload_preserves_router_registered_model_info(monkeypatch):
    """
    A price-data reload replaces litellm.model_cost wholesale. Deployment
    model_info registered by the Router is not in the fetched catalog, so
    without a replay of runtime registrations the reload silently strips
    max_input_tokens / max_output_tokens from every custom model group and
    /model_group/info starts reporting nulls.
    """
    from litellm import utils as litellm_utils
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "custom-alias",
                "litellm_params": {"model": "hosted_vllm/not-in-the-catalog"},
                "model_info": {
                    "id": "custom-alias-id",
                    "max_input_tokens": 128000,
                    "max_output_tokens": 16384,
                },
            }
        ],
    )

    before = router.get_model_group_info(model_group="custom-alias")
    assert before is not None
    assert before.max_input_tokens == 128000
    assert before.max_output_tokens == 16384

    saved_model_cost = litellm.model_cost
    try:
        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        after = router.get_model_group_info(model_group="custom-alias")
        assert after is not None
        assert after.max_input_tokens == 128000
        assert after.max_output_tokens == 16384
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_price_data_reload_preserves_custom_override_of_a_catalog_model(monkeypatch):
    """
    A deployment whose backend model IS in the catalog is the quieter half of
    the same bug: the reload does not blank the metadata, it reverts the
    operator's model_info override to the upstream catalog values.
    """
    from litellm import utils as litellm_utils
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "capped-gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {
                    "id": "capped-gpt-4o-id",
                    "max_input_tokens": 12345,
                    "max_output_tokens": 678,
                },
            }
        ],
    )

    saved_model_cost = litellm.model_cost
    try:
        _simulate_price_data_reload(
            {
                "openai/gpt-4o": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "max_input_tokens": 999999,
                    "max_output_tokens": 888888,
                }
            },
        )

        after = router.get_model_group_info(model_group="capped-gpt-4o")
        assert after is not None
        assert after.max_input_tokens == 12345
        assert after.max_output_tokens == 678
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_deleted_deployments_are_not_replayed_onto_later_reloads(monkeypatch):
    """
    Runtime registrations are replayed onto every price data reload, so a
    deleted deployment has to be withdrawn or it is re-asserted for the life of
    the process and the registry grows with every create/delete cycle. A backend
    key that another live deployment still points at must survive the same
    deletion.
    """
    from litellm import utils as litellm_utils
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "doomed",
                "litellm_params": {"model": "hosted_vllm/shared-backend"},
                "model_info": {"id": "doomed-id", "max_input_tokens": 111},
            },
            {
                "model_name": "kept",
                "litellm_params": {"model": "hosted_vllm/shared-backend"},
                "model_info": {"id": "kept-id", "max_input_tokens": 222},
            },
            {
                "model_name": "solo",
                "litellm_params": {"model": "hosted_vllm/solo-backend"},
                "model_info": {"id": "solo-id", "max_input_tokens": 333},
            },
        ],
    )

    saved_model_cost = litellm.model_cost
    try:
        assert router.delete_deployment(id="doomed-id") is not None
        assert router.delete_deployment(id="solo-id") is not None

        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        assert "doomed-id" not in litellm.model_cost
        assert "solo-id" not in litellm.model_cost
        assert "hosted_vllm/solo-backend" not in litellm.model_cost

        surviving = litellm.model_cost["kept-id"]
        assert surviving["max_input_tokens"] == 222
        assert "hosted_vllm/shared-backend" in litellm.model_cost
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_deleting_a_deployment_leaves_catalog_pricing_for_its_backend_model(monkeypatch):
    """
    A backend key is shared with the fetched catalog, so withdrawing the entries
    a deleted deployment owns must not take real upstream pricing down with it.
    """
    from litellm import utils as litellm_utils

    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    backend_model = "gemini/gemini-2.5-pro"
    catalog_entry = litellm.get_model_info(model=backend_model)
    catalog_input_cost = catalog_entry["input_cost_per_token"]
    assert catalog_input_cost > 0, "Test requires a catalog model with non-zero pricing"

    saved_catalog = litellm.model_cost
    fetched_catalog = copy.deepcopy(litellm.model_cost)
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "doomed-gemini",
                    "litellm_params": {"model": backend_model, "api_key": "sk-fake"},
                    "model_info": {"id": "doomed-gemini-id"},
                }
            ],
        )

        assert router.delete_deployment(id="doomed-gemini-id") is not None

        _simulate_price_data_reload(
            copy.deepcopy(fetched_catalog),
        )

        assert "doomed-gemini-id" not in litellm.model_cost
        assert litellm.model_cost[backend_model]["input_cost_per_token"] == catalog_input_cost
    finally:
        litellm.model_cost = saved_catalog
        _invalidate_model_cost_lowercase_map()


def test_repointing_a_deployment_drops_its_previous_backend_key(monkeypatch):
    """
    An update that moves a deployment onto a different backend model leaves the
    old backend key behind, and a replayed registry would re-assert it onto every
    later catalog for the life of the process.
    """
    from litellm import utils as litellm_utils
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "moving-target",
                "litellm_params": {"model": "hosted_vllm/old-backend"},
                "model_info": {"id": "moving-target-id"},
            }
        ],
    )

    saved_model_cost = litellm.model_cost
    try:
        router.upsert_deployment(
            deployment=Deployment(
                model_name="moving-target",
                litellm_params=LiteLLM_Params(model="hosted_vllm/new-backend"),
                model_info=ModelInfo(id="moving-target-id"),
            )
        )

        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        assert "hosted_vllm/old-backend" not in litellm.model_cost
        assert "hosted_vllm/new-backend" in litellm.model_cost
        assert "moving-target-id" in litellm.model_cost
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(
    "model, custom_llm_provider, expected",
    [
        ("gpt-4o", None, ("gpt-4o",)),
        ("gpt-4o", "openai", ("openai/gpt-4o",)),
        ("openai/gpt-4o", None, ("openai/gpt-4o",)),
        ("responses/gpt-4o", "openai", ("openai/responses/gpt-4o", "openai/gpt-4o")),
        ("responses/gpt-4o", None, ("responses/gpt-4o", "gpt-4o")),
    ],
)
def test_backend_cost_map_keys_matches_what_registration_writes(model, custom_llm_provider, expected):
    """
    The withdrawal path drops exactly the keys the registration wrote, so the two
    have to agree on the provider prefix and on the responses/ alias. The first
    key is also the one the registration uses as the shared backend key, so its
    position is load-bearing rather than incidental.
    """
    keys = Router._backend_cost_map_keys(model=model, custom_llm_provider=custom_llm_provider)
    assert keys == expected
    assert keys[0] == (model if custom_llm_provider is None else f"{custom_llm_provider}/{model}")


def test_a_discarded_router_stops_contributing_to_later_reloads(monkeypatch):
    """
    `_route_user_config_request` builds a Router per request from caller-supplied
    config and discards it. Nothing can withdraw entries on its behalf afterwards,
    so a rebuild driven off live routers is what keeps a caller from growing the
    cost map one request at a time.
    """
    saved_model_cost = litellm.model_cost
    try:
        kept = Router(
            model_list=[
                {
                    "model_name": "kept",
                    "litellm_params": {"model": "hosted_vllm/kept-backend"},
                    "model_info": {"id": "kept-router-id", "max_input_tokens": 4242},
                }
            ],
        )
        throwaway = Router(
            model_list=[
                {
                    "model_name": "throwaway",
                    "litellm_params": {"model": "hosted_vllm/throwaway-backend"},
                    "model_info": {"id": "throwaway-router-id", "max_input_tokens": 111},
                }
            ],
        )
        throwaway.discard()

        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )

        assert "throwaway-router-id" not in litellm.model_cost
        assert "hosted_vllm/throwaway-backend" not in litellm.model_cost
        assert litellm.model_cost["kept-router-id"]["max_input_tokens"] == 4242
        assert "hosted_vllm/kept-backend" in litellm.model_cost
        assert kept.model_list  # keep the live router referenced for the duration
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_a_reload_rebuilds_exactly_what_a_fresh_boot_registered():
    """
    The rebuild is only correct if it reproduces the entries the original
    registration wrote, including the pieces that are derived rather than stored:
    custom pricing carried on litellm_params, and the cache pricing inherited from
    the built-in cost map.
    """
    saved_catalog = litellm.model_cost
    fetched_catalog = copy.deepcopy(litellm.model_cost)
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "priced",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000123,
                        "output_cost_per_token": 0.000456,
                    },
                    "model_info": {"id": "priced-id", "max_input_tokens": 4242},
                }
            ],
        )
        at_boot = copy.deepcopy(litellm.model_cost["priced-id"])
        assert at_boot["input_cost_per_token"] == 0.000123
        assert at_boot["cache_read_input_token_cost"] is not None

        _simulate_price_data_reload(
            copy.deepcopy(fetched_catalog),
        )

        rebuilt = litellm.model_cost["priced-id"]
        assert at_boot.items() <= rebuilt.items(), (
            f"the rebuild changed or dropped a field the boot registration wrote: "
            f"{ {k: (v, rebuilt.get(k)) for k, v in at_boot.items() if rebuilt.get(k) != v} }"
        )
        # The rebuild goes through the deployment stored in model_list, which also
        # carries the router's own db_model flag; add_deployment already registers it.
        assert set(rebuilt) - set(at_boot) <= {"db_model"}
        assert router.model_list
    finally:
        litellm.model_cost = saved_catalog
        _invalidate_model_cost_lowercase_map()


def test_replay_model_cost_registrations_survives_a_malformed_deployment():
    """
    The rebuild reads whatever dicts are sitting in model_list, so one entry that
    cannot be rebuilt into a Deployment must not stop the rest being restored.
    """
    saved_model_cost = litellm.model_cost
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "healthy",
                    "litellm_params": {"model": "hosted_vllm/healthy-backend"},
                    "model_info": {"id": "healthy-id", "max_input_tokens": 777},
                }
            ],
        )
        router.model_list.insert(0, {"litellm_params": {}})

        litellm.model_cost = {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}}
        _invalidate_model_cost_lowercase_map()
        router._replay_model_cost_registrations()

        assert litellm.model_cost["healthy-id"]["max_input_tokens"] == 777
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_deployment_model_cost_payload_folds_in_litellm_params_pricing():
    """
    Custom pricing is configured on litellm_params but has to land in the
    cost-map entry, and setting it pulls in the built-in cache pricing for the
    backend model. Both are what make the entry reproducible from a deployment.
    """
    payload = Router._deployment_model_cost_payload(
        deployment=Deployment(
            model_name="priced",
            litellm_params=LiteLLM_Params(
                model="gemini/gemini-2.5-pro",
                input_cost_per_token=0.000123,
            ),
            model_info=ModelInfo(id="payload-id", max_input_tokens=4242),
        )
    )

    assert payload["id"] == "payload-id"
    assert payload["max_input_tokens"] == 4242
    assert payload["input_cost_per_token"] == 0.000123
    assert payload["cache_read_input_token_cost"] > 0


def test_register_deployment_in_model_cost_writes_both_key_families():
    """
    A deployment contributes its full model_info under its unique id and the
    cost-map subset under the shared backend key, and the shared key must not
    pick up the deployment's private metadata.
    """
    model_keys = {
        "both-families-id": copy.deepcopy(litellm.model_cost.get("both-families-id")),
        "hosted_vllm/both-families-backend": copy.deepcopy(
            litellm.model_cost.get("hosted_vllm/both-families-backend")
        ),
    }
    try:
        Router._register_deployment_in_model_cost(
            model_id="both-families-id",
            model_info={"id": "both-families-id", "max_input_tokens": 999, "litellm_provider": "hosted_vllm"},
            model="hosted_vllm/both-families-backend",
            custom_llm_provider=None,
        )

        assert litellm.model_cost["both-families-id"]["max_input_tokens"] == 999
        shared = litellm.model_cost["hosted_vllm/both-families-backend"]
        assert shared["max_input_tokens"] == 999
        assert "id" not in shared
    finally:
        _restore_model_cost_entries(model_keys)


def test_reload_keeps_custom_pricing_configured_on_litellm_params_for_a_db_model():
    """
    A deployment added at runtime, which is what /model/new does, configures its
    custom pricing on litellm_params rather than on model_info. A price data
    reload must not revert that to the catalog's pricing.
    """
    saved_catalog = litellm.model_cost
    fetched_catalog = copy.deepcopy(litellm.model_cost)
    try:
        router = Router(model_list=[])
        router.add_deployment(
            deployment=Deployment(
                model_name="db-priced",
                litellm_params=LiteLLM_Params(
                    model="openai/gpt-4o",
                    api_key="sk-fake",
                    input_cost_per_token=0.000123,
                    output_cost_per_token=0.000456,
                ),
                model_info=ModelInfo(id="db-priced-id"),
            )
        )

        assert litellm.model_cost["db-priced-id"]["input_cost_per_token"] == 0.000123

        _simulate_price_data_reload(
            copy.deepcopy(fetched_catalog),
        )

        assert litellm.model_cost["db-priced-id"]["input_cost_per_token"] == 0.000123
        assert litellm.model_cost["db-priced-id"]["output_cost_per_token"] == 0.000456
    finally:
        litellm.model_cost = saved_catalog
        _invalidate_model_cost_lowercase_map()


def test_replay_live_router_model_cost_rebuilds_every_live_router():
    """
    A process can hold more than one Router, so the rebuild has to fan out across
    all of them rather than restoring whichever one happens to be reachable.
    """
    from litellm.router import _replay_live_router_model_cost

    saved_model_cost = litellm.model_cost
    try:
        first = Router(
            model_list=[
                {
                    "model_name": "first",
                    "litellm_params": {"model": "hosted_vllm/first-backend"},
                    "model_info": {"id": "first-id", "max_input_tokens": 111},
                }
            ],
        )
        second = Router(
            model_list=[
                {
                    "model_name": "second",
                    "litellm_params": {"model": "hosted_vllm/second-backend"},
                    "model_info": {"id": "second-id", "max_input_tokens": 222},
                }
            ],
        )

        litellm.model_cost = {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}}
        _invalidate_model_cost_lowercase_map()
        _replay_live_router_model_cost()

        assert litellm.model_cost["first-id"]["max_input_tokens"] == 111
        assert litellm.model_cost["second-id"]["max_input_tokens"] == 222
        assert first.model_list and second.model_list
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()


def test_strategy_router_alias_pricing_never_enters_model_cost(monkeypatch):
    """
    A strategy-router alias is never the deployment actually called or billed,
    so custom pricing configured on it must not be registered under its
    model_id - an explicit zero there makes the budget check treat the alias
    as a genuinely free model while requests bill as a real deployment. The
    strip must also survive a price-data reload, which rebuilds entries by
    walking the live routers.
    """
    from litellm import utils as litellm_utils
    monkeypatch.setattr(
        litellm_utils,
        "_runtime_registered_model_cost",
        dict(litellm_utils._runtime_registered_model_cost),
    )

    router = Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router/smart-router",
                    "complexity_router_default_model": "paid-model",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "complexity_router_config": {"tiers": {"simple": "paid-model"}},
                },
                "model_info": {"id": "strategy-alias-id", "max_input_tokens": 128000},
            },
            {
                "model_name": "paid-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                "model_info": {"id": "strategy-alias-paid-id"},
            },
        ],
    )

    def _assert_alias_unpriced():
        entry = litellm.model_cost.get("strategy-alias-id")
        assert entry is not None, "Alias metadata should still be registered"
        assert entry["max_input_tokens"] == 128000
        assert "input_cost_per_token" not in entry
        assert "output_cost_per_token" not in entry

    _assert_alias_unpriced()

    saved_model_cost = litellm.model_cost
    try:
        _simulate_price_data_reload(
            {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}},
        )
        _assert_alias_unpriced()
        assert router.model_list
    finally:
        litellm.model_cost = saved_model_cost
        _invalidate_model_cost_lowercase_map()
