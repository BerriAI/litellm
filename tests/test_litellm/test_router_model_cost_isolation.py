"""
Test that per-deployment custom pricing does not pollute the shared backend
model key in litellm.model_cost.

When two deployments share the same backend model (e.g. vertex_ai/gemini-2.5-flash)
and one has explicit zero-cost pricing in model_info, the other deployment
should still use the built-in pricing.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router


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
    assert builtin_input_cost > 0, "Test requires a model with non-zero built-in pricing"
    assert builtin_output_cost > 0, "Test requires a model with non-zero built-in pricing"

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
