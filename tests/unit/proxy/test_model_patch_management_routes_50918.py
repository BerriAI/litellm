import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks


def test_model_patch_update_route_in_management_routes():
    """
    Regression test for #50918:
    PATCH /model/{model_id}/update must be included in management_routes
    so virtual keys restricted to management_routes can partially update models.
    """
    assert "/model/{model_id}/update" in LiteLLMRoutes.management_routes.value
    assert RouteChecks.is_management_route(route="/model/model-abc-123/update") is True


def test_virtual_key_with_management_routes_can_call_model_patch_update():
    valid_token = UserAPIKeyAuth(allowed_routes=["management_routes"])
    assert (
        RouteChecks.virtual_key_can_call_route(
            route="/model/model-abc-123/update",
            valid_token=valid_token,
        )
        is True
    )


def test_virtual_key_with_llm_api_routes_cannot_call_model_patch_update():
    valid_token = UserAPIKeyAuth(allowed_routes=["llm_api_routes"])
    with pytest.raises((HTTPException, Exception)):
        RouteChecks.virtual_key_can_call_route(
            route="/model/model-abc-123/update",
            valid_token=valid_token,
        )
