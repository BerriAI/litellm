import asyncio
import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

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
async def test_cancelled_caller_still_commits_once_and_waits_for_database_ack() -> None:
    query: Final = ScriptedQuery(hold=True)
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    expected: Final = RECEIPT.model_copy(update={"event_id": capture.event_id})
    query.replies = (ConnectionError("lost commit acknowledgement"), acknowledged(expected))
    event: Final = priced_event()
    saving: Final = asyncio.create_task(
        capture.capture_success(
            {
                "model": event.model,
                "litellm_params": {},
                "response_cost": event.response_cost,
                "standard_logging_object": event.standard_logging_object,
                "messages": [{"role": "user", "content": "private"}],
            },
            event.response,
            STAMP,
            STAMP,
        )
    )
    await query.entered.wait()
    waiting: Final = asyncio.create_task(capture.wait())
    await asyncio.sleep(0)
    assert not waiting.done() and not saving.done()
    assert json.loads(str(query.calls[0][2])) == expected.model_dump()
    saving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await saving
    closing: Final = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not closing.done()
    query.release.set()
    await closing
    assert await waiting and len(query.calls) == 2
    assert query.calls[0] == query.calls[1]
    repeated: Final = await capture.capture_event(priced_event(amount=0.5))
    assert isinstance(repeated, CommittedCharge) and repeated.amount_usd == "0.25" and len(query.calls) == 2
    assert not await runtime.ready()


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", (None, -1.0, float("inf"), float("nan")))
async def test_invalid_price_is_unconfirmed_not_a_zero_dollar_receipt(amount: float | None) -> None:
    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    details: Final = {
        "model": RECEIPT.model,
        "litellm_params": {},
        "response_cost": amount,
        "standard_logging_object": {"response_cost": 0 if amount is None else amount},
    }
    result: Final = await capture.capture_success(details, None, STAMP, STAMP)
    assert isinstance(result, FailedCharge)
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
@pytest.mark.parametrize("failure", ("before_commit", "after_commit", "timeout"))
async def test_optional_accounting_failure_cannot_drop_or_repeat_a_known_charge(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.spend_tracking.charge_receipt_capture import ChargeCapture
    from litellm.proxy.spend_tracking.spend_event import SpendEvent

    clock: Final = Mock(return_value=0.0)
    monkeypatch.setattr(asyncio.get_running_loop(), "time", clock)
    finalizing: Final = asyncio.Event()

    async def finalize(details: Mapping[str, object], event: SpendEvent, capture: ChargeCapture) -> None:
        finalizing.set()
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "after_commit":
            await capture.persist()
        raise ConnectionError("accounting dependency unavailable")

    query: Final = ScriptedQuery(hold=failure == "timeout")
    runtime: Final = ChargeReceiptRuntime(
        SETTINGS.model_copy(update={"commit_timeout_seconds": 3}), ChargeReceiptStore(query), finalize=finalize
    )
    capture: Final = runtime.bind(BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash))
    query.replies = (acknowledged(RECEIPT.model_copy(update={"event_id": capture.event_id})),)
    saving: Final = asyncio.create_task(capture.capture_event(priced_event(), {}))
    waiting: Final = asyncio.create_task(capture.wait())
    if failure == "timeout":
        await finalizing.wait()
        clock.return_value = 3.5
        await query.entered.wait()
        clock.return_value = 5.5
        tick: Final = asyncio.Event()
        asyncio.get_running_loop().call_later(0, tick.set)
        await tick.wait()
        assert not waiting.done()
        query.release.set()
    result: Final = await saving
    assert isinstance(result, CommittedCharge) and result.amount_usd == RECEIPT.amount_usd
    assert await waiting and len(query.calls) == 1


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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("classifier", "embedding", "ordinary", "forged"))
async def test_internal_calls_get_distinct_receipts_from_authenticated_parent_only(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm
    from litellm.caching._embedding_router import build_router_embedding_metadata
    from litellm.litellm_core_utils.internal_call_metadata import forwarded_internal_call_metadata
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, EmbeddingResponse, ModelResponse, Usage

    query: Final = ScriptedQuery()
    runtime: Final = ChargeReceiptRuntime(SETTINGS, ChargeReceiptStore(query))
    context: Final = BillingContext(request_id=RECEIPT.request_id, key_hash=RECEIPT.key_hash)
    captures: Final = (runtime.bind(context), runtime.bind(context))
    available: Final = iter(captures)
    auth: Final = UserAPIKeyAuth.model_validate(
        {
            "api_key": RECEIPT.key_hash,
            "budget_reservation": {"parent": True},
            "_charge_receipt_factory": "untrusted",
        }
    )
    if kind != "forged":
        auth.set_charge_receipt_factory(lambda: next(available))
    parent: Final = {
        "user_api_key_auth": auth.model_copy(deep=True),
        "user_api_key_budget_reservation": auth.budget_reservation,
    }
    metadata: Final = (
        build_router_embedding_metadata(parent)
        if kind == "embedding"
        else parent
        if kind == "ordinary"
        else forwarded_internal_call_metadata(parent, AUTOROUTER_CLASSIFIER_CALL_ORIGIN)
    )
    embedding: Final = kind == "embedding"
    monkeypatch.setitem(
        litellm.model_cost,
        RECEIPT.model,
        {
            "input_cost_per_token": 0.0625 if embedding else 0.03125,
            "output_cost_per_token": 0.0625,
            "litellm_provider": "openai",
            "mode": "embedding" if embedding else "chat",
        },
    )
    expected: Final = RECEIPT.model_copy(update={"output_tokens": 0, "total_tokens": 4}) if embedding else RECEIPT
    query.replies = tuple(
        acknowledged(expected.model_copy(update={"event_id": capture.event_id})) for capture in captures
    )
    response: Final = (
        EmbeddingResponse(model=RECEIPT.model, data=[], usage=Usage(prompt_tokens=4, total_tokens=4))
        if embedding
        else ModelResponse(model=RECEIPT.model, usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6))
    )
    for i in range(2):
        call: Final = Logging(
            model=RECEIPT.model,
            messages=[],
            stream=False,
            call_type="aembedding" if embedding else "acompletion",
            start_time=STAMP,
            litellm_call_id=str(i),
            function_id=str(i),
            kwargs={
                "litellm_metadata" if kind == "classifier" else "metadata": metadata,
                "custom_llm_provider": "openai",
            },
        )
        call.update_environment_variables(litellm_params={}, optional_params={}, custom_llm_provider="openai")
        await call.async_success_handler(response, start_time=STAMP, end_time=STAMP, cache_hit=False)
    assert "_charge_receipt_factory" not in auth.model_dump_json()
    if kind in ("ordinary", "forged"):
        assert not query.calls
        return
    child_auth: Final = metadata["user_api_key_auth"]
    assert isinstance(child_auth, UserAPIKeyAuth) and child_auth.budget_reservation is None
    assert auth.budget_reservation == {"parent": True}
    receipts: Final = tuple(json.loads(str(call[2])) for call in query.calls)
    assert len({receipt["event_id"] for receipt in receipts}) == 2
    assert {(receipt["request_id"], receipt["key_hash"], receipt["amount_usd"]) for receipt in receipts} == {
        (RECEIPT.request_id, RECEIPT.key_hash, RECEIPT.amount_usd),
    }
