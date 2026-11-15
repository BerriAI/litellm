import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.accounting.application import apply_deliveries, apply_delivery_isolated
from litellm.accounting.delivery_projection import project_deliveries
from litellm.accounting.projection import Contribution
from litellm.proxy.common_utils.customer_usage import CUSTOMER_USAGE_METADATA_KEY, CustomerUsageAttribution
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, merge_by_conflict_key
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.utils import PrismaClient, ProxyLogging
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


def _recording_transaction() -> MagicMock:
    transaction: Final = MagicMock()
    transaction.execute_raw = AsyncMock(return_value=1)
    transaction.query_raw = AsyncMock(return_value=[])
    batch: Final = MagicMock()
    batch.__aenter__ = AsyncMock(return_value=transaction.batcher)
    batch.__aexit__ = AsyncMock(return_value=False)
    transaction.batch_.return_value = batch
    return transaction


def _tables_in_write_order(transaction: MagicMock) -> tuple[str, ...]:
    def table(name: str, args: tuple[object, ...]) -> str | None:
        if name.startswith("batcher.litellm_"):
            return name.split(".")[1]
        if name == "execute_raw" and isinstance(args[0], str):
            found: Final = re.search(r'"(LiteLLM_\w+)"', args[0])
            return found.group(1).lower() if found else None
        return None

    return tuple(
        found for found in (table(name, args) for name, args, _ in transaction.mock_calls) if found is not None
    )


@pytest.mark.asyncio
async def test_an_isolated_retry_writes_destinations_in_the_batched_order() -> None:
    delivery: Final = DeliveryProjection(
        {
            "request_id": "without-team",
            "startTime": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "api_key": "hashed-key",
            "user": "user",
            "end_user": "end-user",
            "model": "shared-model",
            "model_group": "shared-model",
            "custom_llm_provider": "openai",
            "spend": 0.25,
            "request_tags": '["first-tag", "second-tag"]',
            "metadata": {"accounting_entities": {"api_key": "hashed-key", "user": "user", "end_user": "end-user"}},
        }
    )
    database: Final = MagicMock(spec=PrismaClient)
    writer: Final = DBSpendUpdateWriter(accounting_clock=lambda: 0.0)
    logger: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    batched: Final = _recording_transaction()
    isolated: Final = _recording_transaction()

    await apply_deliveries(batched, (delivery,), prisma_client=database, writer=writer, proxy_logging_obj=logger)
    applied, rejected = await apply_delivery_isolated(
        isolated, delivery, prisma_client=database, writer=writer, proxy_logging_obj=logger
    )

    assert rejected == ()
    assert {"litellm_usertable", "litellm_dailyuserspend"} <= set(_tables_in_write_order(batched))
    assert tuple(dict.fromkeys(_tables_in_write_order(isolated))) == tuple(
        dict.fromkeys(_tables_in_write_order(batched))
    )
    assert len(applied) == len(project_deliveries((delivery,)))
