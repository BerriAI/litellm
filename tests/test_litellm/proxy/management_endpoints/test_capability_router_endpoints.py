import pytest
from pydantic import ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.auto_router_endpoints import validate_capability_router_config
from litellm.types.management_endpoints.auto_router_endpoints import (
    AutoRouterRoutingTestRequest,
    CapabilityRouterConfigValidationRequest,
)


def config() -> dict:
    return {
        "candidates": [
            {"model": "small", "description": "Reliable for short extraction tasks"},
            {"model": "frontier", "description": "Reliable for ambiguous multi-step tasks"},
        ],
        "classifier": {"model": "classifier"},
        "probability_threshold": 0.7,
        "fallback_model": "frontier",
    }


def test_routing_preview_accepts_exactly_one_router_config() -> None:
    request = AutoRouterRoutingTestRequest.model_validate(
        {"prompt": "Extract the invoice number", "capability_router_config": config()}
    )
    assert request.capability_router_config is not None
    assert request.complexity_router_config is None

    with pytest.raises(ValidationError, match="exactly one router config"):
        AutoRouterRoutingTestRequest.model_validate(
            {
                "prompt": "Extract the invoice number",
                "capability_router_config": config(),
                "complexity_router_config": {
                    "tiers": {"SIMPLE": "small"},
                    "classifier_type": "heuristic",
                },
            }
        )


@pytest.mark.asyncio
async def test_validation_endpoint_uses_runtime_config_contract() -> None:
    admin = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="admin")
    accepted = await validate_capability_router_config(
        CapabilityRouterConfigValidationRequest(capability_router_config=config()),
        admin,
    )
    rejected = await validate_capability_router_config(
        CapabilityRouterConfigValidationRequest(
            capability_router_config={**config(), "fallback_model": "missing"}
        ),
        admin,
    )

    assert accepted.valid is True
    assert rejected.valid is False
    assert rejected.error is not None and "fallback_model" in rejected.error
