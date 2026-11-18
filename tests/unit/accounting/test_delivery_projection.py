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


def test_baseline_entity_spend_from_different_budget_periods_stays_split() -> None:
    old_period: Final = "2026-11-01T00:00:00.000000"
    new_period: Final = "2026-11-02T00:00:00.000000"

    def row(spend: float, period: str) -> Mapping[str, object]:
        payload: Final = dict(_payload(spend))
        metadata: Final = json.loads(str(payload["metadata"]))
        metadata["accounting_budget_periods"] = [
            {
                "table": "LiteLLM_VerificationToken",
                "entity_id": "key",
                "budget_reset_at": period,
            }
        ]
        return {**payload, "metadata": json.dumps(metadata)}

    contributions: Final = project_deliveries(
        (DeliveryProjection(row(2, old_period)), DeliveryProjection(row(3, new_period)))
    )

    assert sorted(_spend(contributions, "LiteLLM_VerificationToken")) == [2, 3]
    transactions: Final = entity_transactions(contributions)
    assert transactions["key_list_transactions"] == {"key": 5}
    assert transactions["baseline_spend"] == [
        {
            "field": "key_list_transactions",
            "entity_id": "key",
            "amount": 2,
            "budget_reset_at": old_period,
        },
        {
            "field": "key_list_transactions",
            "entity_id": "key",
            "amount": 3,
            "budget_reset_at": new_period,
        },
    ]


def test_baseline_entity_spend_in_one_budget_period_still_sums() -> None:
    period: Final = "2026-11-01T00:00:00.000000"

    def row(spend: float) -> Mapping[str, object]:
        payload: Final = dict(_payload(spend))
        metadata: Final = json.loads(str(payload["metadata"]))
        metadata["accounting_budget_periods"] = [
            {"table": "LiteLLM_VerificationToken", "entity_id": "key", "budget_reset_at": period}
        ]
        return {**payload, "metadata": json.dumps(metadata)}

    contributions: Final = project_deliveries((DeliveryProjection(row(2)), DeliveryProjection(row(3))))

    assert _spend(contributions, "LiteLLM_VerificationToken") == (5,)


def test_entity_transactions_key_memberships_the_way_the_writer_applies_them() -> None:
    transactions: Final = entity_transactions(project_deliveries((DeliveryProjection(_payload(0.25)),)))

    assert transactions["key_list_transactions"] == {"key": 0.25}
    assert transactions["org_list_transactions"] == {"org::1": 0.25}
    assert transactions["team_member_list_transactions"] == {"team_id::team::user_id::user/1": 0.25}
    ((org_member_key, org_member_spend),) = transactions["org_member_list_transactions"].items()
    _, quoted_org_id, _, quoted_user_id = org_member_key.split("::")
    assert (unquote(quoted_org_id), unquote(quoted_user_id), org_member_spend) == ("org::1", "user/1", 0.25)
