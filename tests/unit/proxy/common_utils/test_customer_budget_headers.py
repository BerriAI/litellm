from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_exception_handler import _as_proxy_exception
from litellm.proxy.auth.customer_hierarchy_auth import CustomerBudgetExceededError
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.customer_budget_headers import customer_budget_headers
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerBudgetWindowUsage,
    CustomerHierarchyAuth,
    CustomerScopeAuth,
)

ENABLED: Final[Mapping[str, object]] = {"customer_budget_response_headers": True}
DAILY_RESET: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)
MONTHLY_RESET: Final = datetime(2026, 2, 1, tzinfo=timezone.utc)

HIERARCHY: Final = CustomerHierarchyAuth(
    scopes=(
        CustomerScopeAuth(key="app", entity_id="app-1", spend=50.0, max_budget=1000.0),
        CustomerScopeAuth(
            key="app_customer_org",
            entity_id="app-1:org-1",
            spend=40.0,
            budget_window_usage=(
                CustomerBudgetWindowUsage(budget_duration="1d", spend=4.5, max_budget=5.0, reset_at=DAILY_RESET),
                CustomerBudgetWindowUsage(budget_duration="1mo", spend=40.0, max_budget=100.0, reset_at=MONTHLY_RESET),
            ),
        ),
        CustomerScopeAuth(
            key="app_customer_member",
            entity_id="app-1:member-1",
            spend=7.0,
            max_budget=6.0,
            budget_window_usage=(CustomerBudgetWindowUsage(budget_duration="1d", spend=1.0, max_budget=2.0),),
        ),
    )
)

EXPECTED: Final = {
    "x-litellm-app-customer-org-spend-1d": "4.5",
    "x-litellm-app-customer-org-max-budget-1d": "5.0",
    "x-litellm-app-customer-org-remaining-budget-1d": "0.5",
    "x-litellm-app-customer-org-budget-reset-at-1d": "2026-01-02T00:00:00+00:00",
    "x-litellm-app-customer-org-spend-1mo": "40.0",
    "x-litellm-app-customer-org-max-budget-1mo": "100.0",
    "x-litellm-app-customer-org-remaining-budget-1mo": "60.0",
    "x-litellm-app-customer-org-budget-reset-at-1mo": "2026-02-01T00:00:00+00:00",
    "x-litellm-app-customer-member-spend": "7.0",
    "x-litellm-app-customer-member-max-budget": "6.0",
    "x-litellm-app-customer-member-remaining-budget": "0.0",
    "x-litellm-app-customer-member-spend-1d": "1.0",
    "x-litellm-app-customer-member-max-budget-1d": "2.0",
    "x-litellm-app-customer-member-remaining-budget-1d": "1.0",
}


def test_each_budget_window_of_the_organization_and_member_reports_under_its_own_duration() -> None:
    assert dict(customer_budget_headers(HIERARCHY, ENABLED)) == EXPECTED


def test_a_request_without_a_customer_reports_no_customer_budget() -> None:
    assert dict(customer_budget_headers(None, ENABLED)) == {}
    assert dict(customer_budget_headers(CustomerHierarchyAuth(scopes=()), ENABLED)) == {}


def test_a_reset_time_is_reported_in_utc_whatever_zone_it_was_stored_in() -> None:
    eastern: Final = timezone(timedelta(hours=-5))
    hierarchy: Final = CustomerHierarchyAuth(
        scopes=(
            CustomerScopeAuth(
                key="app_customer_org",
                entity_id="org-1",
                spend=1.0,
                max_budget=10.0,
                budget_reset_at=datetime(2026, 1, 1, 19, tzinfo=eastern),
            ),
            CustomerScopeAuth(
                key="app_customer_member",
                entity_id="member-1",
                spend=1.0,
                max_budget=10.0,
                budget_reset_at=datetime(2026, 1, 2),
            ),
        )
    )
    headers: Final = customer_budget_headers(hierarchy, ENABLED)
    assert headers["x-litellm-app-customer-org-budget-reset-at"] == "2026-01-02T00:00:00+00:00"
    assert headers["x-litellm-app-customer-member-budget-reset-at"] == "2026-01-02T00:00:00+00:00"


def test_budgets_outside_the_calling_app_are_not_reported() -> None:
    hierarchy: Final = CustomerHierarchyAuth(
        scopes=tuple(
            CustomerScopeAuth(key=key, entity_id=f"{key}-1", spend=1.0, max_budget=10.0)
            for key in (
                "customer_org",
                "customer_team",
                "customer_member",
                "app",
                "app_group",
                "app_group_customer_org",
            )
        )
    )
    assert dict(customer_budget_headers(hierarchy, ENABLED)) == {}


@pytest.mark.parametrize("general_settings", ({}, {"customer_budget_response_headers": False}))
def test_customer_budgets_are_not_reported_unless_the_proxy_opts_in(general_settings: Mapping[str, object]) -> None:
    assert dict(customer_budget_headers(HIERARCHY, general_settings)) == {}


def _customer_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {name: value for name, value in headers.items() if "customer" in name}


def test_a_successful_response_carries_the_customer_budget_headers_once_the_proxy_opts_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key: Final = UserAPIKeyAuth(api_key="sk-test", customer_hierarchy=HIERARCHY)
    assert _customer_headers(ProxyBaseLLMRequestProcessing.get_custom_headers(user_api_key_dict=key)) == {}
    monkeypatch.setitem(proxy_server.general_settings, "customer_budget_response_headers", True)
    assert _customer_headers(ProxyBaseLLMRequestProcessing.get_custom_headers(user_api_key_dict=key)) == EXPECTED


def test_a_budget_rejection_carries_the_customer_budget_headers_once_the_proxy_opts_in() -> None:
    error: Final = CustomerBudgetExceededError(
        HIERARCHY, litellm.BudgetExceededError(current_cost=7.0, max_budget=6.0, message="over budget")
    )
    assert _as_proxy_exception(error, {}).headers == {}
    assert _as_proxy_exception(error, ENABLED).headers == EXPECTED


def test_a_budget_rejection_outside_a_customer_carries_no_customer_budget_headers() -> None:
    rejection: Final = _as_proxy_exception(
        litellm.BudgetExceededError(current_cost=7.0, max_budget=6.0, message="over budget"), ENABLED
    )
    assert not rejection.headers


@pytest.mark.parametrize("stored_duration", ("1d ", "1d\n"))
def test_whitespace_a_stored_duration_ends_with_stays_out_of_the_header_name(stored_duration: str) -> None:
    hierarchy: Final = CustomerHierarchyAuth(
        scopes=(
            CustomerScopeAuth(
                key="app_customer_org",
                entity_id="app-1:org-1",
                budget_window_usage=(
                    CustomerBudgetWindowUsage(budget_duration=stored_duration, spend=1.0, max_budget=2.0),
                ),
            ),
        )
    )
    assert dict(customer_budget_headers(hierarchy, ENABLED)) == {
        "x-litellm-app-customer-org-spend-1d": "1.0",
        "x-litellm-app-customer-org-max-budget-1d": "2.0",
        "x-litellm-app-customer-org-remaining-budget-1d": "1.0",
    }
