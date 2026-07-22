"""
Test that per-deployment custom pricing does not pollute the shared backend
model key in litellm.model_cost.

When two deployments share the same backend model (e.g. vertex_ai/gemini-2.5-flash)
and one has explicit zero-cost pricing in model_info, the other deployment
should still use the built-in pricing.
"""

import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import _invalidate_model_cost_lowercase_map


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


def test_openai_compatible_glm_partial_pricing_inherits_cache_rate():
    backend_model = "openai/z-ai/glm-5.2"
    deploy_id = "glm-5-2-partial-pricing"
    model_keys = {
        deploy_id: litellm.model_cost.get(deploy_id),
        backend_model: copy.deepcopy(litellm.model_cost.get(backend_model)),
    }

    try:
        model_prices_path = (
            Path(__file__).parents[2] / "model_prices_and_context_window.json"
        )
        with model_prices_path.open(encoding="utf-8") as model_prices_file:
            model_cost = json.load(model_prices_file)
        litellm.register_model({backend_model: model_cost[backend_model]})

        Router(
            model_list=[
                {
                    "model_name": "glm-5.2",
                    "litellm_params": {
                        "model": backend_model,
                        "api_base": "https://api.example.com/v1",
                        "api_key": "fake-key",
                        "custom_llm_provider": "openai",
                    },
                    "model_info": {
                        "id": deploy_id,
                        "input_cost_per_token": 7.8e-07,
                        "output_cost_per_token": 2.42e-06,
                    },
                }
            ],
        )

        entry = litellm.model_cost[deploy_id]
        assert entry["input_cost_per_token"] == 7.8e-07
        assert entry["output_cost_per_token"] == 2.42e-06
        assert entry["cache_read_input_token_cost"] == 2.6e-07

        response = ModelResponse(
            model=backend_model,
            choices=[],
            usage=Usage(
                prompt_tokens=1000,
                completion_tokens=100,
                total_tokens=1100,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
            ),
        )
        cost = litellm.completion_cost(
            completion_response=response,
            model=backend_model,
            custom_llm_provider="openai",
            custom_pricing=True,
            router_model_id=deploy_id,
        )
        assert cost == pytest.approx(200 * 7.8e-07 + 800 * 2.6e-07 + 100 * 2.42e-06)
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
