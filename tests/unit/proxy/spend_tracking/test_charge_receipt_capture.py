import asyncio
import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy.spend_tracking.charge_receipt_capture import BillingContext, ChargeReceiptRuntime
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptStore
from litellm.types.charge_receipts import CommittedCharge, FailedCharge
from tests.unit.proxy.spend_tracking._charge_receipt_cases import (
    RECEIPT,
    SETTINGS,
    STAMP,
    ScriptedQuery,
    acknowledged,
    priced_event,
)


@pytest.mark.asyncio
async def test_capture_waits_for_database_ack_and_journals_no_request_content() -> None:
    query: Final = ScriptedQuery(hold=True)
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    expected: Final = RECEIPT.model_copy(update={"event_id": capture.event_id})
    query.replies = (acknowledged(expected),)
    saving: Final = asyncio.create_task(capture.capture_event(priced_event()))
    await query.entered.wait()
    waiting: Final = asyncio.create_task(capture.wait())
    await asyncio.sleep(0)
    assert not waiting.done() and not saving.done()
    assert json.loads(str(query.calls[0].parameters[2])) == expected.model_dump()
    query.release.set()
    result: Final = await saving
    assert isinstance(result, CommittedCharge) and result.amount_usd == "0.25"
    assert await waiting
    repeated: Final = await capture.capture_event(priced_event(amount=0.5))
    assert repeated is result and len(query.calls) == 1


@pytest.mark.asyncio
async def test_caller_cancellation_cannot_cancel_a_started_charge_write() -> None:
    query: Final = ScriptedQuery(hold=True)
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    query.replies = (acknowledged(RECEIPT.model_copy(update={"event_id": capture.event_id})),)
    caller: Final = asyncio.create_task(capture.capture_event(priced_event()))
    await query.entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    closing: Final = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not closing.done()
    query.release.set()
    await closing
    assert await capture.wait() and len(query.calls) == 1
    assert not await runtime.ready()


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", (None, -1.0, float("inf"), float("nan")))
async def test_invalid_price_is_unconfirmed_not_a_zero_dollar_receipt(amount: float | None) -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    assert isinstance(await capture.capture_event(priced_event(amount=amount)), FailedCharge)
    assert not await capture.wait() and not query.calls


@pytest.mark.asyncio
async def test_a_known_late_charge_can_follow_an_unpriced_failure() -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    assert isinstance(await capture.capture_failure({}, 0, STAMP, STAMP), FailedCharge)
    query.replies = (acknowledged(RECEIPT.model_copy(update={"event_id": capture.event_id})),)
    assert isinstance(await capture.capture_event(priced_event()), CommittedCharge)
    assert await capture.wait() and len(query.calls) == 1


@pytest.mark.asyncio
async def test_failure_receipt_uses_the_known_partial_charge() -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    expected: Final = RECEIPT.model_copy(update={"event_id": capture.event_id})
    query.replies = (acknowledged(expected),)
    result: Final = await capture.capture_failure(
        {
            "model": RECEIPT.model,
            "litellm_params": {},
            "standard_logging_object": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        0.25,
        STAMP,
        STAMP,
    )
    assert isinstance(result, CommittedCharge) and result.amount_usd == "0.25"
    assert json.loads(str(query.calls[0].parameters[2])) == expected.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("update", ({"model": ""}, {"combined_usage": {"prompt_tokens": -1}}))
async def test_invalid_usage_or_missing_model_cannot_create_a_paid_receipt(update: Mapping[str, object]) -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    assert isinstance(await capture.capture_event(priced_event().model_copy(update=update)), FailedCharge)
    assert not await capture.wait()
    assert not query.calls


@pytest.mark.asyncio
async def test_readiness_is_cached_only_after_a_valid_durability_probe() -> None:
    query: Final = ScriptedQuery((({"installed": True, "durable": True},),))
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    assert await runtime.ready() and await runtime.ready()
    assert len(query.calls) == 1
    await runtime.close()
    assert not await runtime.ready()


@pytest.mark.asyncio
async def test_success_logging_projects_only_the_canonical_charge_and_usage() -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    expected: Final = RECEIPT.model_copy(update={"event_id": capture.event_id})
    query.replies = (acknowledged(expected),)
    result: Final = await capture.capture_success(
        {
            "model": RECEIPT.model,
            "litellm_params": {},
            "response_cost": 0.25,
            "messages": [{"role": "user", "content": "private prompt"}],
            "standard_logging_object": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
                "response_cost": 0.25,
            },
        },
        {"id": "owned-response", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        STAMP,
        STAMP,
    )
    assert isinstance(result, CommittedCharge) and await capture.wait()
    assert json.loads(str(query.calls[0].parameters[2])) == expected.model_dump()


@pytest.mark.asyncio
async def test_unreadable_logging_input_cannot_create_a_success_or_failure_charge() -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    success: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    failure: Final = success.fork()
    assert failure.event_id != success.event_id
    assert isinstance(
        await success.capture_success({"litellm_params": object()}, object(), STAMP, STAMP),
        FailedCharge,
    )
    assert isinstance(await failure.capture_failure({"litellm_params": object()}, 0.25, STAMP, STAMP), FailedCharge)
    assert not await success.wait() and not await failure.wait() and not query.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("committed_before_failure", (False, True))
async def test_optional_accounting_failure_cannot_drop_or_repeat_a_known_charge(committed_before_failure: bool) -> None:
    from litellm.proxy.spend_tracking.charge_receipt_capture import ChargeCapture
    from litellm.proxy.spend_tracking.spend_event import SpendEvent

    async def finalize(details: Mapping[str, object], event: SpendEvent, capture: ChargeCapture) -> None:
        if committed_before_failure:
            await capture.persist()
        raise ConnectionError("accounting dependency unavailable")

    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query), finalize=finalize)
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    query.replies = (acknowledged(RECEIPT.model_copy(update={"event_id": capture.event_id})),)
    result: Final = await capture.capture_event(priced_event(), {})
    assert isinstance(result, CommittedCharge) and result.amount_usd == RECEIPT.amount_usd
    assert await capture.wait() and len(query.calls) == 1


@pytest.mark.asyncio
async def test_committed_response_does_not_wait_for_optional_accounting_to_finish() -> None:
    from litellm.proxy.spend_tracking.charge_receipt_capture import ChargeCapture
    from litellm.proxy.spend_tracking.spend_event import SpendEvent

    release: Final = asyncio.Event()

    async def finalize(details: Mapping[str, object], event: SpendEvent, capture: ChargeCapture) -> None:
        await capture.persist()
        await release.wait()

    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query), finalize=finalize)
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    query.replies = (acknowledged(RECEIPT.model_copy(update={"event_id": capture.event_id})),)
    settling: Final = asyncio.create_task(capture.capture_event(priced_event(), {}))
    assert await capture.wait() and not settling.done()
    release.set()
    assert isinstance(await settling, CommittedCharge) and len(query.calls) == 1
