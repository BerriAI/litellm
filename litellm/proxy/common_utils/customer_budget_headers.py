from collections.abc import Mapping
from datetime import datetime, timezone
from itertools import chain
from typing import Final

from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth, CustomerScopeKey

_HEADER_SCOPES: Final[frozenset[CustomerScopeKey]] = frozenset(("app_customer_org", "app_customer_member"))


def _budget_headers(
    prefix: str, suffix: str, spend: float, max_budget: float, reset_at: datetime | None
) -> Mapping[str, str]:
    return {
        f"{prefix}-spend{suffix}": str(spend),
        f"{prefix}-max-budget{suffix}": str(max_budget),
        f"{prefix}-remaining-budget{suffix}": str(max(max_budget - spend, 0.0)),
        **(
            {
                f"{prefix}-budget-reset-at{suffix}": (
                    reset_at if reset_at.tzinfo is not None else reset_at.replace(tzinfo=timezone.utc)
                )
                .astimezone(timezone.utc)
                .isoformat()
            }
            if reset_at is not None
            else {}
        ),
    }


def _scope_headers(scope: CustomerScopeAuth) -> Mapping[str, str]:
    prefix: Final = f"x-litellm-{scope.key.replace('_', '-')}"
    return dict(
        chain.from_iterable(
            budget.items()
            for budget in (
                *(
                    (_budget_headers(prefix, "", scope.spend, scope.max_budget, scope.budget_reset_at),)
                    if scope.max_budget is not None
                    else ()
                ),
                *(
                    _budget_headers(
                        prefix, f"-{window.budget_duration}", window.spend, window.max_budget, window.reset_at
                    )
                    for window in scope.budget_window_usage
                ),
            )
        )
    )


def customer_budget_headers(hierarchy: object, general_settings: Mapping[str, object]) -> Mapping[str, str]:
    """Spend, max budget, remaining budget and reset time the calling app holds for the request's customer organization
    and member, as auth read them before the request ran. A budget window carries its duration as a suffix
    (``-spend-1d``), so a level with a daily and a monthly cap reports each one under its own name; a budget with no
    window has no suffix. Off unless ``general_settings.customer_budget_response_headers`` is true, because a key that
    is not pinned to one customer organization reads the budget of whichever organization it names."""
    if general_settings.get("customer_budget_response_headers") is not True or not isinstance(
        hierarchy, CustomerHierarchyAuth
    ):
        return {}
    return dict(
        chain.from_iterable(_scope_headers(scope).items() for scope in hierarchy.scopes if scope.key in _HEADER_SCOPES)
    )
