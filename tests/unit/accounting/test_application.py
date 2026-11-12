from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final, cast

import pytest
from pydantic import TypeAdapter

from litellm.accounting.application import _apply_customer_usage
from litellm.accounting.delivery_projection import project_deliveries
from litellm.accounting.projection import Contribution
from litellm.proxy.common_utils.customer_usage import CUSTOMER_USAGE_METADATA_KEY, CustomerUsageAttribution
from litellm.repositories.accounting_delivery_repository import DeliveryProjection, LedgerTransaction

_FIELDS: Final = TypeAdapter(Mapping[str, object])
_TEXT: Final = TypeAdapter(str)
_UPDATES: Final = TypeAdapter(Mapping[str, Mapping[str, float]])
_IDENTIFIER: Final = "\U0001f680" * 256
_ATTRIBUTION: Final = CustomerUsageAttribution(
    customer_org_id=_IDENTIFIER, customer_team_id=_IDENTIFIER, customer_member_id=_IDENTIFIER
)


@dataclass(frozen=True, slots=True)
class Bucket:
    values: Mapping[str, object]
    counters: Mapping[str, float]


class Buckets:
    def __init__(self) -> None:
        self.rows: Mapping[str, Bucket] = {}

    async def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        row_id: Final = _TEXT.validate_python(where["id"])
        create: Final = _FIELDS.validate_python(data["create"])
        previous: Final = self.rows.get(row_id)
        increments: Final = (
            {name: fields["increment"] for name, fields in _UPDATES.validate_python(data["update"]).items()}
            if previous is not None
            else {name: float(value) for name, value in create.items() if isinstance(value, (int, float))}
        )
        self.rows = {
            **self.rows,
            row_id: Bucket(
                values=previous.values if previous is not None else create,
                counters={
                    name: (previous.counters.get(name, 0) if previous is not None else 0) + value
                    for name, value in increments.items()
                },
            ),
        }


@dataclass(frozen=True, slots=True)
class Transaction:
    litellm_dailycustomermemberspend: Buckets


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


@pytest.mark.asyncio
async def test_long_customer_ids_roll_up_without_truncation_and_keep_alias_provider_buckets_distinct() -> None:
    buckets: Final = Buckets()
    transaction: Final = cast(LedgerTransaction, Transaction(buckets))  # cast-ok: in-memory Prisma upsert boundary
    first: Final = contribution("alias-a", "provider-a", 1)
    second: Final = replace(contribution("alias-a", "provider-a", 2), key=tuple(reversed(first.key)))

    await _apply_customer_usage(transaction, (first,))
    await _apply_customer_usage(transaction, (second, contribution("alias-b", "provider-a", 4)))
    await _apply_customer_usage(transaction, (contribution("alias-a", "provider-b", 8),))

    assert len(buckets.rows) == 3
    assert all(len(row_id.encode("utf-8")) == 64 for row_id in buckets.rows)
    assert all(
        row.values["customer_org_id"]
        == row.values["customer_team_id"]
        == row.values["customer_member_id"]
        == _IDENTIFIER
        for row in buckets.rows.values()
    )
    assert {
        (row.values["model_group"], row.values["custom_llm_provider"]): row.counters["spend"]
        for row in buckets.rows.values()
    } == {("alias-a", "provider-a"): 3, ("alias-b", "provider-a"): 4, ("alias-a", "provider-b"): 8}
    assert sum(row.counters["api_requests"] for row in buckets.rows.values()) == 4
    assert sum(row.counters["prompt_tokens"] + row.counters["completion_tokens"] for row in buckets.rows.values()) == 60
