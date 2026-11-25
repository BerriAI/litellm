from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Final

from litellm.accounting.delivery_projection import project_deliveries
from litellm.accounting.projection import Contribution
from litellm.proxy.common_utils.customer_usage import CUSTOMER_USAGE_METADATA_KEY, CustomerUsageAttribution
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, merge_by_conflict_key
from litellm.repositories.accounting_delivery_repository import DeliveryProjection

_IDENTIFIER: Final = "\U0001f680" * 256
_ATTRIBUTION: Final = CustomerUsageAttribution(
    customer_org_id=_IDENTIFIER, customer_team_id=_IDENTIFIER, customer_member_id=_IDENTIFIER
)


def contribution(alias: str, provider: str, spend: float) -> Contribution:
    delivery: Final = DeliveryProjection(
        {
            "request_id": "long-customer-id-request",
            "startTime": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "api_key": "hashed-key",
            "model": "shared-model",
            "model_group": alias,
            "custom_llm_provider": provider,
            "spend": spend,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "metadata": {CUSTOMER_USAGE_METADATA_KEY: _ATTRIBUTION.model_dump_json()},
        }
    )
    return next(item for item in project_deliveries((delivery,)) if item.table == "LiteLLM_DailyCustomerMemberSpend")


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _row(item: Contribution) -> Mapping[str, object]:
    return {**dict(item.key), **dict(item.values), **dict(item.increments)}


def test_long_customer_ids_roll_up_without_truncation_and_aliases_and_providers_keep_distinct_rows() -> None:
    first: Final = contribution("alias-a", "provider-a", 1)
    second: Final = replace(contribution("alias-a", "provider-a", 2), key=tuple(reversed(first.key)))
    merged: Final = merge_by_conflict_key(
        DAILY_SPEND_TABLES["customer_member"],
        tuple(
            _row(item)
            for item in (
                first,
                second,
                contribution("alias-b", "provider-a", 4),
                contribution("alias-a", "provider-b", 8),
            )
        ),
    )
    rows: Final = tuple(row for _, row in merged)
    assert len(rows) == 3
    assert all(
        row["customer_org_id"] == row["customer_team_id"] == row["customer_member_id"] == _IDENTIFIER for row in rows
    )
    assert {(row["model_group"], row["custom_llm_provider"]): _number(row["spend"]) for row in rows} == {
        ("alias-a", "provider-a"): 3,
        ("alias-b", "provider-a"): 4,
        ("alias-a", "provider-b"): 8,
    }
    assert sum(_number(row["api_requests"]) for row in rows) == 4
    assert sum(_number(row["prompt_tokens"]) + _number(row["completion_tokens"]) for row in rows) == 60
