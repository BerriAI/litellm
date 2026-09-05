import pytest
import litellm
from litellm.router import Router


def test_router_case_variant_deployments_capability_isolation():
    """
    Issue #39909: Ensure registering case-variant models (e.g. GLM-5.3-Flash vs glm-5.3-flash)
    does not leak one deployment's declared capabilities (max_input_tokens, supports_vision)
    into the other deployment.
    """
    model_list = [
        {
            "model_name": "mixed-group",
            "litellm_params": {
                "model": "anthropic/GLM-5.3-Flash",
                "api_key": "sk-x",
                "api_base": "https://provider-a.example",
            },
            "model_info": {"provider": "provider-a"},
        },
        {
            "model_name": "mixed-group",
            "litellm_params": {
                "model": "anthropic/glm-5.3-flash",
                "api_key": "sk-y",
                "api_base": "https://provider-b.example",
            },
            "model_info": {
                "provider": "provider-b",
                "max_input_tokens": 450000,
                "supports_vision": True,
            },
        },
    ]

    router = Router(model_list=model_list)

    dep_a = next(
        d for d in router.model_list if d["litellm_params"]["model"] == "anthropic/GLM-5.3-Flash"
    )
    dep_b = next(
        d for d in router.model_list if d["litellm_params"]["model"] == "anthropic/glm-5.3-flash"
    )

    info_a = router.get_router_model_info(deployment=dep_a, received_model_name="mixed-group")
    info_b = router.get_router_model_info(deployment=dep_b, received_model_name="mixed-group")

    assert info_a.get("max_input_tokens") is None
    assert info_a.get("supports_vision") is not True

    assert info_b.get("max_input_tokens") == 450000
    assert info_b.get("supports_vision") is True


def test_router_case_variant_deployments_reverse_order():
    """
    Ensure capability isolation holds regardless of registration order
    (Provider B with caps registered first, Provider A registered second).
    """
    model_list = [
        {
            "model_name": "mixed-group",
            "litellm_params": {
                "model": "anthropic/glm-5.3-flash",
                "api_key": "sk-y",
                "api_base": "https://provider-b.example",
            },
            "model_info": {
                "provider": "provider-b",
                "max_input_tokens": 450000,
                "supports_vision": True,
            },
        },
        {
            "model_name": "mixed-group",
            "litellm_params": {
                "model": "anthropic/GLM-5.3-Flash",
                "api_key": "sk-x",
                "api_base": "https://provider-a.example",
            },
            "model_info": {"provider": "provider-a"},
        },
    ]

    router = Router(model_list=model_list)

    dep_a = next(
        d for d in router.model_list if d["litellm_params"]["model"] == "anthropic/GLM-5.3-Flash"
    )
    dep_b = next(
        d for d in router.model_list if d["litellm_params"]["model"] == "anthropic/glm-5.3-flash"
    )

    info_a = router.get_router_model_info(deployment=dep_a, received_model_name="mixed-group")
    info_b = router.get_router_model_info(deployment=dep_b, received_model_name="mixed-group")

    assert info_a.get("max_input_tokens") is None
    assert info_a.get("supports_vision") is not True

    assert info_b.get("max_input_tokens") == 450000
    assert info_b.get("supports_vision") is True


def test_register_model_does_not_merge_case_variants():
    """
    Direct register_model test: registering a model should not inherit or overwrite
    a case-variant runtime entry.
    """
    key_upper = "custom-test-provider/TEST-CAP-MODEL"
    key_lower = "custom-test-provider/test-cap-model"

    # Clean up if previously set
    litellm.model_cost.pop(key_upper, None)
    litellm.model_cost.pop(key_lower, None)
    litellm.utils._invalidate_model_cost_lowercase_map()

    try:
        litellm.register_model({
            key_lower: {
                "litellm_provider": "openai",
                "max_input_tokens": 200000,
                "supports_vision": True,
            }
        })

        litellm.register_model({
            key_upper: {
                "litellm_provider": "openai",
                "max_input_tokens": 50000,
            }
        })

        entry_lower = litellm.model_cost.get(key_lower, {})
        entry_upper = litellm.model_cost.get(key_upper, {})

        assert entry_lower.get("max_input_tokens") == 200000
        assert entry_lower.get("supports_vision") is True

        assert entry_upper.get("max_input_tokens") == 50000
        # supports_vision was not declared for key_upper, should not leak from key_lower
        assert entry_upper.get("supports_vision") is not True
    finally:
        litellm.model_cost.pop(key_upper, None)
        litellm.model_cost.pop(key_lower, None)
        litellm.utils._invalidate_model_cost_lowercase_map()
