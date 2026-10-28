import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import NoReturn

import pytest

from litellm.accounting import InboundUsageEvent, Metric, MetricName, Outcome
from litellm.proxy.db.accounting_ledger import AccountingLedger
from litellm.types.utils import CallTypes


def _no_transaction() -> NoReturn:
    raise AssertionError("Invalid accounting input must not open a database transaction")


@pytest.fixture
def event() -> InboundUsageEvent:
    return InboundUsageEvent(
        event_id="caller",
        request_id="caller",
        call_type=CallTypes.completion,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        outcome=Outcome.SUCCEEDED,
        dimensions=(),
        metrics=(Metric(name=MetricName.REQUESTS, value=Decimal(1)),),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("envelope", ["{}", "[]", '{"event":null}', '{"event":{}}', "not json"])
async def test_invalid_envelopes_fail_before_database_access(event, envelope):
    with pytest.raises(ValueError, match=r"Accounting envelope|validation error"):
        await AccountingLedger(_no_transaction).insert_event(event, (), envelope=envelope)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [0, -1, True, 1.5])
@pytest.mark.parametrize("field", ["version", "schema_version"])
async def test_invalid_versions_fail_before_database_access(event, version, field):
    with pytest.raises(ValueError, match="positive integers"):
        await AccountingLedger(_no_transaction).insert_event(event, (), envelope="{}", **{field: version})


@pytest.mark.asyncio
async def test_constructed_invalid_events_are_revalidated(event):
    invalid = event.model_copy(update={"event_id": ""})
    with pytest.raises(ValueError, match="event_id"):
        await AccountingLedger(_no_transaction).insert_event(invalid, (), envelope="{}")


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
async def test_retention_rejects_invalid_batch_bounds(limit):
    with pytest.raises(ValueError, match="batch size"):
        await AccountingLedger(_no_transaction).purge_completed(limit=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["both", "redelivery", "recovery"])
async def test_retention_requires_both_safety_watermarks(missing):
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    deleted = await AccountingLedger(_no_transaction).purge_completed(
        redelivery_before=cutoff if missing == "recovery" else None,
        recovery_before=cutoff if missing == "redelivery" else None,
    )
    assert deleted == 0


@pytest.mark.asyncio
async def test_retention_rejects_ambiguous_naive_watermarks():
    with pytest.raises(ValueError, match="timezone"):
        await AccountingLedger(_no_transaction).purge_completed(
            redelivery_before=datetime(2026, 1, 1), recovery_before=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )


@pytest.mark.asyncio
async def test_nonfinite_context_is_rejected_before_database_access(event):
    envelope = json.dumps({"event": event.model_dump(mode="json"), "context": {"price": float("nan")}})
    with pytest.raises(ValueError, match="Out of range float"):
        await AccountingLedger(_no_transaction).insert_event(event, (), envelope=envelope)


@pytest.mark.asyncio
async def test_unknown_delivery_target_is_rejected_before_database_access(event):
    envelope = json.dumps({"event": event.model_dump(mode="json")})
    with pytest.raises(ValueError, match="not a valid DeliveryTarget"):
        await AccountingLedger(_no_transaction).insert_event(event, ("unknown",), envelope=envelope)
