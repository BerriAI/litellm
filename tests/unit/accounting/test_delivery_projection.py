import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final
from urllib.parse import unquote

from litellm.accounting.delivery_projection import entity_transactions, project_deliveries
from litellm.accounting.projection import Contribution
from litellm.repositories.accounting_delivery_repository import DeliveryProjection

_NOW: Final = datetime(2026, 11, 2, tzinfo=timezone.utc)


def _payload(spend: float, started: datetime = _NOW, window_start: datetime = _NOW) -> Mapping[str, object]:
    return {
        "request_id": f"request-{spend}-{started.isoformat()}",
        "startTime": started,
        "api_key": "key",
        "user": "user/1",
        "team_id": "team",
        "organization_id": "org::1",
        "spend": spend,
        "metadata": json.dumps(
            {
                "accounting_entities": {
                    "api_key": "key",
                    "user": "user/1",
                    "team_id": "team",
                    "organization_id": "org::1",
                },
                "accounting_windows": [
                    {
                        "entity_type": "key",
                        "entity_id": "key",
                        "window_duration": "1d",
                        "window_start": window_start.isoformat(),
                    }
                ],
            }
        ),
    }


def _spend(contributions: tuple[Contribution, ...], table: str) -> tuple[float, ...]:
    return tuple(float(dict(item.increments)["spend"]) for item in contributions if item.table == table)


def test_app_charge_is_additional_to_global_customers_and_daily_attribution() -> None:
    row: Final = {
        "request_id": "request",
        "startTime": _NOW,
        "spend": 0.25,
        "api_key": "key",
        "model": "model",
        "custom_llm_provider": "provider",
        "model_group": "model",
        "metadata": {
            "accounting_entities": {
                "app_id": "app",
                "customer_org_id": "org",
                "customer_team_id": "team",
                "customer_member_id": "member",
            }
        },
    }
    contributions: Final = project_deliveries((DeliveryProjection(row),))
    totals: Final = entity_transactions(contributions)
    assert totals["customer_org_list_transactions"] == {"org": 0.25}
    assert totals["customer_team_list_transactions"] == {"team": 0.25}
    assert totals["customer_member_list_transactions"] == {"member": 0.25}
    assert totals["app_list_transactions"] == {"app": 0.25}
    assert totals["app_customer_org_list_transactions"] == {"app:org": 0.25}
    assert totals["app_customer_member_list_transactions"] == {"app:member": 0.25}
    daily: Final = tuple(item for item in contributions if item.table == "LiteLLM_DailyCustomerMemberSpend")
    assert len(daily) == 1 and dict(daily[0].key)["app_id"] == "app"


def _app_delivery(app_id: str, organization_id: str, member_id: str, spend: float) -> DeliveryProjection:
    return DeliveryProjection(
        {
            "request_id": f"{app_id}-{organization_id}-{member_id}",
            "startTime": _NOW,
            "spend": spend,
            "api_key": "key",
            "model": "model",
            "custom_llm_provider": "provider",
            "model_group": "model",
            "metadata": {
                "accounting_entities": {
                    "app_id": app_id,
                    "customer_org_id": organization_id,
                    "customer_member_id": member_id,
                }
            },
        }
    )


def test_one_customers_spend_across_apps_keeps_one_global_total_and_distinct_daily_rows() -> None:
    contributions: Final = project_deliveries(
        (
            _app_delivery("chat", "org", "member", 0.25),
            _app_delivery("search", "org", "member", 0.5),
        )
    )
    totals: Final = entity_transactions(contributions)
    assert totals["customer_org_list_transactions"] == {"org": 0.75}
    assert totals["customer_member_list_transactions"] == {"member": 0.75}
    assert totals["app_list_transactions"] == {"chat": 0.25, "search": 0.5}
    assert totals["app_customer_org_list_transactions"] == {
        "chat:org": 0.25,
        "search:org": 0.5,
    }
    assert totals["app_customer_member_list_transactions"] == {
        "chat:member": 0.25,
        "search:member": 0.5,
    }
    daily: Final = {
        dict(item.key)["app_id"]: dict(item.increments)["spend"]
        for item in contributions
        if item.table == "LiteLLM_DailyCustomerMemberSpend"
    }
    assert daily == {"chat": 0.25, "search": 0.5}
    assert sum(daily.values()) == totals["customer_org_list_transactions"]["org"]


def test_customer_balances_within_one_app_remain_independent() -> None:
    totals: Final = entity_transactions(
        project_deliveries(
            (
                _app_delivery("chat", "first", "member-a", 0.25),
                _app_delivery("chat", "second", "member-b", 0.5),
            )
        )
    )
    assert totals["customer_org_list_transactions"] == {"first": 0.25, "second": 0.5}
    assert totals["customer_member_list_transactions"] == {
        "member-a": 0.25,
        "member-b": 0.5,
    }
    assert totals["app_list_transactions"] == {"chat": 0.75}
    assert totals["app_customer_org_list_transactions"] == {
        "chat:first": 0.25,
        "chat:second": 0.5,
    }
    assert totals["app_customer_member_list_transactions"] == {
        "chat:member-a": 0.25,
        "chat:member-b": 0.5,
    }


def test_deliveries_for_one_destination_sum_and_keep_the_earliest_request_start() -> None:
    later: Final = _NOW + timedelta(hours=1)

    contributions: Final = project_deliveries(
        (DeliveryProjection(_payload(0.5, started=later)), DeliveryProjection(_payload(0.25)))
    )

    assert _spend(contributions, "LiteLLM_VerificationToken") == (0.75,)
    windows: Final = tuple(item for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    assert len(windows) == 1
    assert dict(windows[0].increments) == {"spend": 0.75}
    assert dict(windows[0].values)["started_at"] == str(_NOW)


def test_the_same_window_in_a_new_period_is_charged_to_its_own_row() -> None:
    next_day: Final = _NOW + timedelta(days=1)

    contributions: Final = project_deliveries(
        (DeliveryProjection(_payload(0.25)), DeliveryProjection(_payload(0.5, window_start=next_day)))
    )

    windows: Final = sorted(
        (dict(item.values)["window_start"], dict(item.increments)["spend"])
        for item in contributions
        if item.table == "LiteLLM_BudgetWindowSpend"
    )
    assert windows == [(_NOW.isoformat(), 0.25), (next_day.isoformat(), 0.5)]


def test_entity_transactions_key_memberships_the_way_the_writer_applies_them() -> None:
    transactions: Final = entity_transactions(project_deliveries((DeliveryProjection(_payload(0.25)),)))

    assert transactions["key_list_transactions"] == {"key": 0.25}
    assert transactions["org_list_transactions"] == {"org::1": 0.25}
    assert transactions["team_member_list_transactions"] == {"team_id::team::user_id::user/1": 0.25}
    ((org_member_key, org_member_spend),) = transactions["org_member_list_transactions"].items()
    _, quoted_org_id, _, quoted_user_id = org_member_key.split("::")
    assert (unquote(quoted_org_id), unquote(quoted_user_id), org_member_spend) == ("org::1", "user/1", 0.25)


def test_group_totals_and_history_use_each_requests_snapshot_after_membership_changes() -> None:
    def delivery(group_id: str, app_id: str, organization_id: str, cost: float) -> DeliveryProjection:
        return DeliveryProjection(
            {
                "request_id": f"{group_id}-{app_id}-{organization_id}",
                "startTime": _NOW,
                "spend": cost,
                "api_key": "key",
                "model": "model",
                "model_group": "model",
                "custom_llm_provider": "provider",
                "metadata": {
                    "accounting_entities": {
                        "app_id": app_id,
                        "app_group_id": group_id,
                        "customer_org_id": organization_id,
                    }
                },
            }
        )

    contributions: Final = project_deliveries(
        (
            delivery("first", "chat", "org", 1),
            delivery("first", "search", "org", 2),
            delivery("first", "chat", "other", 4),
            delivery("second", "chat", "org", 8),
        )
    )
    totals: Final = entity_transactions(contributions)
    assert totals["app_group_list_transactions"] == {"first": 7, "second": 8}
    assert totals["app_group_customer_org_list_transactions"] == {"first:org": 3, "first:other": 4, "second:org": 8}
    assert totals["app_list_transactions"] == {"chat": 13, "search": 2}
    assert totals["customer_org_list_transactions"] == {"org": 11, "other": 4}
    daily: Final = {
        (dict(item.key)["app_group_id"], dict(item.key)["app_id"], dict(item.key)["customer_org_id"]): dict(
            item.increments
        )["spend"]
        for item in contributions
        if item.table == "LiteLLM_DailyCustomerMemberSpend"
    }
    assert daily == {
        ("first", "chat", "org"): 1,
        ("first", "search", "org"): 2,
        ("first", "chat", "other"): 4,
        ("second", "chat", "org"): 8,
    }
