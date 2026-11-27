from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.finalization import finalize_cost
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import adjusted_cost
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.customer_cost_margin import (
    apply_customer_cost_margin,
    apply_customer_cost_margin_to_estimate,
    customer_cost_margin,
    customer_margin_share_of_spend,
    key_customer_cost_margin,
    request_customer_cost_margin,
    request_has_customer,
)
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth


@pytest.mark.parametrize(("margin", "expected"), [(0.02, 1.02), (10, 11.0)])
def test_customer_margin_is_added_and_recorded_separately(margin: float, expected: float) -> None:
    cost: Final = apply_customer_cost_margin(finalize_cost(1.0, None), margin)
    assert cost.amount == pytest.approx(expected)
    assert cost.logging_breakdown() == {
        "customer_margin_percent": margin,
        "customer_margin_amount": pytest.approx(expected - 1.0),
        "total_cost": pytest.approx(expected),
    }


def test_no_customer_margin_leaves_the_cost_unchanged() -> None:
    cost: Final = finalize_cost(3.0, None)
    assert apply_customer_cost_margin(cost, None) is cost
    assert apply_customer_cost_margin_to_estimate(3.0, None) == 3.0
    assert apply_customer_cost_margin_to_estimate(None, 0.02) is None


@pytest.mark.parametrize("configured", [None, "", -1, 10.5, float("nan"), float("inf"), True, {"percentage": 0.02}])
def test_an_invalid_configured_margin_is_ignored(configured: object) -> None:
    with patch.dict(litellm.cost_margin_config, {"customer": configured}, clear=True):
        assert customer_cost_margin(True) is None


def test_the_configured_margin_only_applies_to_requests_with_a_customer() -> None:
    with patch.dict(litellm.cost_margin_config, {"customer": 0.02}, clear=True):
        assert customer_cost_margin(request_has_customer({"user_api_key_customer_org_id": "org-1"})) == 0.02
        assert customer_cost_margin(request_has_customer({"user_api_key_customer_org_id": None})) is None
        assert customer_cost_margin(request_has_customer({"user_api_key_team_id": "team-1"})) is None
        bound: Final = {"user_api_key_customer_usage": '{"customer_org_id": "org-1"}'}
        assert customer_cost_margin(request_has_customer(bound)) == 0.02
        assert customer_cost_margin(request_has_customer({"user_api_key_customer_usage": "{}"})) is None
        assert customer_cost_margin(request_has_customer({"user_api_key_customer_usage": "not json"})) is None


@pytest.mark.parametrize("provider", ["anthropic", "other"])
def test_customer_margin_composes_after_provider_or_global_margin(provider: str) -> None:
    with patch.dict(litellm.cost_margin_config, {"anthropic": 0.1, "global": 0.2, "customer": 0.02}, clear=True):
        provider_cost: Final = adjusted_cost(5.0, provider).cost
        cost: Final = apply_customer_cost_margin(finalize_cost(provider_cost, None), customer_cost_margin(True))
    assert provider_cost == pytest.approx(5.0 * (1.1 if provider == "anthropic" else 1.2))
    assert cost.amount == pytest.approx(provider_cost * 1.02)
    assert apply_customer_cost_margin(cost, 0.02) is cost


def test_a_free_request_stays_free_with_a_customer_margin() -> None:
    assert apply_customer_cost_margin(finalize_cost(0.0, None), 0.02).amount == 0.0


@pytest.mark.parametrize(
    ("metadata", "spend", "expected"),
    [
        ({"cost_breakdown": {"customer_margin_amount": 0.02}}, 1.02, 0.02),
        ({"cost_breakdown": {"customer_margin_amount": 5.0}}, 1.0, 1.0),
        ({"cost_breakdown": {"customer_margin_amount": -1.0}}, 1.0, 0.0),
        ({"cost_breakdown": {"total_cost": 1.0}}, 1.0, 0.0),
        ({"cost_breakdown": None}, 1.0, 0.0),
        ({}, 1.0, 0.0),
    ],
)
def test_surcharge_share_of_a_spend_row(metadata: dict[str, object], spend: float, expected: float) -> None:
    assert customer_margin_share_of_spend(metadata, spend) == pytest.approx(expected)


def _stamped(app_customer_margin: object, customer_org_id: str | None = "org-1") -> dict[str, object]:
    return {
        "user_api_key_customer_org_id": customer_org_id,
        "user_api_key_customer_hierarchy": {
            "customer_org_id": customer_org_id,
            "app_id": "app-1",
            "app_customer_margin": app_customer_margin,
        },
    }


@pytest.mark.parametrize(
    ("gateway_margin", "app_margin", "expected"),
    [
        (0.02, 0.05, 0.05),
        (0.02, 0, 0.0),
        (None, 0.05, 0.05),
        (0.02, None, 0.02),
        (0.02, 11, 0.02),
        (0.02, "0.05", 0.02),
        (None, None, None),
    ],
    ids=[
        "app replaces gateway",
        "app zero turns it off",
        "app alone",
        "app unset uses gateway",
        "app out of range uses gateway",
        "app not a number uses gateway",
        "neither",
    ],
)
def test_an_apps_surcharge_replaces_the_gateway_one_for_its_customer_requests(
    gateway_margin: float | None, app_margin: object, expected: float | None
) -> None:
    config: Final = {} if gateway_margin is None else {"customer": gateway_margin}
    with patch.dict(litellm.cost_margin_config, config, clear=True):
        assert request_customer_cost_margin(_stamped(app_margin)) == expected


def test_an_apps_surcharge_does_not_apply_to_a_request_with_no_customer() -> None:
    with patch.dict(litellm.cost_margin_config, {"customer": 0.02}, clear=True):
        assert request_customer_cost_margin(_stamped(0.05, customer_org_id=None)) is None
        assert request_customer_cost_margin({"user_api_key_customer_hierarchy": None}) is None


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        (CustomerHierarchyAuth(customer_org_id="org-1", app_id="app-1", app_customer_margin=0.05, scopes=()), 0.05),
        (CustomerHierarchyAuth(customer_org_id="org-1", scopes=()), 0.02),
        (CustomerHierarchyAuth(app_id="app-1", app_customer_margin=0.05, scopes=()), None),
        (None, None),
    ],
    ids=["app customer", "customer without app", "app without customer", "no hierarchy"],
)
def test_the_reservation_uses_the_same_surcharge_the_request_settles_with(
    hierarchy: CustomerHierarchyAuth | None, expected: float | None
) -> None:
    with patch.dict(litellm.cost_margin_config, {"customer": 0.02}, clear=True):
        assert key_customer_cost_margin(UserAPIKeyAuth(token="key"), hierarchy) == expected


def test_a_hierarchy_without_an_app_serializes_without_app_fields() -> None:
    plain: Final = CustomerHierarchyAuth(customer_org_id="org-1", scopes=()).model_dump(mode="json")
    with_app: Final = CustomerHierarchyAuth(
        customer_org_id="org-1", app_id="app-1", app_alias="chat", app_customer_margin=0.05, scopes=()
    ).model_dump(mode="json")

    assert "app_customer_margin" not in plain
    assert with_app["app_customer_margin"] == 0.05


def test_a_zero_surcharge_leaves_the_cost_and_its_breakdown_untouched() -> None:
    cost: Final = finalize_cost(3.0, None)

    assert apply_customer_cost_margin(cost, 0.0) is cost
