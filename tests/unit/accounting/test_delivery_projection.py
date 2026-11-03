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
                "accounting_windows": [
                    {
                        "entity_type": "key",
                        "entity_id": "key",
                        "window_duration": "1d",
                        "window_start": window_start.isoformat(),
                    }
                ]
            }
        ),
    }


def _spend(contributions: tuple[Contribution, ...], table: str) -> tuple[float, ...]:
    return tuple(float(dict(item.increments)["spend"]) for item in contributions if item.table == table)


def test_deliveries_for_one_destination_sum_and_keep_the_earliest_request_start() -> None:
    later: Final = _NOW + timedelta(hours=1)

    contributions: Final = project_deliveries(
        (DeliveryProjection(_payload(0.5, started=later), None), DeliveryProjection(_payload(0.25), None))
    )

    assert _spend(contributions, "LiteLLM_VerificationToken") == (0.75,)
    windows: Final = tuple(item for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    assert len(windows) == 1
    assert dict(windows[0].increments) == {"spend": 0.75}
    assert dict(windows[0].values)["started_at"] == str(_NOW)


def test_the_same_window_in_a_new_period_is_charged_to_its_own_row() -> None:
    next_day: Final = _NOW + timedelta(days=1)

    contributions: Final = project_deliveries(
        (DeliveryProjection(_payload(0.25), None), DeliveryProjection(_payload(0.5, window_start=next_day), None))
    )

    windows: Final = sorted(
        (dict(item.values)["window_start"], dict(item.increments)["spend"])
        for item in contributions
        if item.table == "LiteLLM_BudgetWindowSpend"
    )
    assert windows == [(_NOW.isoformat(), 0.25), (next_day.isoformat(), 0.5)]


def test_a_replaced_payload_applies_only_its_difference_from_the_previous_charge() -> None:
    contributions: Final = project_deliveries((DeliveryProjection(_payload(0.5), _payload(0.2)),))

    for table in ("LiteLLM_VerificationToken", "LiteLLM_UserTable", "LiteLLM_BudgetWindowSpend"):
        assert _spend(contributions, table) == (0.5 - 0.2,), table


def test_entity_transactions_key_memberships_the_way_the_writer_applies_them() -> None:
    transactions: Final = entity_transactions(project_deliveries((DeliveryProjection(_payload(0.25), None),)))

    assert transactions["key_list_transactions"] == {"key": 0.25}
    assert transactions["org_list_transactions"] == {"org::1": 0.25}
    assert transactions["team_member_list_transactions"] == {"team_id::team::user_id::user/1": 0.25}
    ((org_member_key, org_member_spend),) = transactions["org_member_list_transactions"].items()
    _, quoted_org_id, _, quoted_user_id = org_member_key.split("::")
    assert (unquote(quoted_org_id), unquote(quoted_user_id), org_member_spend) == ("org::1", "user/1", 0.25)
