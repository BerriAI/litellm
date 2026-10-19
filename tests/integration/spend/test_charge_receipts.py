import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Final
from uuid import uuid4

import httpx
import jwt
import pytest
import pytest_asyncio
from fastapi import HTTPException, Request
from prisma import Prisma
from pydantic import SecretStr, TypeAdapter
from starlette.types import Message

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.charge_receipt_capture import BillingContext, ChargeReceiptRuntime
from litellm.proxy.spend_tracking.charge_receipt_delivery import ChargeReceiptPublisher
from litellm.proxy.spend_tracking.charge_receipt_request import authorize_charge_receipt
from litellm.proxy.spend_tracking.charge_receipts import (
    ChargeReceipt,
    ChargeReceiptSettings,
    ChargeReceiptStore,
)
from litellm.proxy.spend_tracking.spend_event import SpendEvent, project_spend_event
from litellm.types.charge_receipts import CommittedCharge, FailedCharge
from litellm.types.llms.base import LiteLLMBaseModel
from tests.integration._support.prisma_clients import connected_prisma

pytestmark = pytest.mark.asyncio(loop_scope="session")


class _Balance(LiteLLMBaseModel):
    spend: float
    total_spend: float


class _Count(LiteLLMBaseModel):
    total: int


_BALANCES: Final = TypeAdapter(tuple[_Balance, ...])
_COUNTS: Final = TypeAdapter(tuple[_Count, ...])


@dataclass(frozen=True, slots=True)
class JournalCase:
    database: Prisma
    store: ChargeReceiptStore
    receipt: ChargeReceipt

    async def balance(self) -> _Balance:
        rows: Final = _BALANCES.validate_python(
            await self.database.query_raw(
                'SELECT spend,total_spend FROM "LiteLLM_VerificationToken" WHERE token=$1',
                self.receipt.key_hash,
            )
        )
        assert len(rows) == 1
        return rows[0]

    async def receipt_count(self) -> int:
        rows: Final = _COUNTS.validate_python(
            await self.database.query_raw(
                'SELECT count(*) AS total FROM "LiteLLM_ChargeReceipt" WHERE key_hash=$1',
                self.receipt.key_hash,
            )
        )
        return rows[0].total


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def database() -> AsyncIterator[Prisma]:
    async with connected_prisma() as client:
        yield client


@pytest_asyncio.fixture(loop_scope="session")
async def journal(database: Prisma) -> AsyncIterator[JournalCase]:
    key_hash: Final = sha256(uuid4().bytes).hexdigest()
    receipt: Final = ChargeReceipt(
        event_id=str(uuid4()),
        request_id=str(uuid4()),
        gateway_scope="a" * 64,
        key_hash=key_hash,
        amount_usd="0.25",
        model="owned-billing-model",
        input_tokens=4,
        output_tokens=2,
        total_tokens=6,
    )
    await database.execute_raw(
        'INSERT INTO "LiteLLM_VerificationToken" (token,models) VALUES ($1,ARRAY[]::text[])', key_hash
    )
    try:
        yield JournalCase(database, ChargeReceiptStore(database.query_raw), receipt)
    finally:
        await database.execute_raw('DELETE FROM "LiteLLM_ChargeReceipt" WHERE key_hash=$1', key_hash)
        await database.execute_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE token=$1', key_hash)


async def test_concurrent_receipt_retries_post_one_key_charge(journal: JournalCase) -> None:
    results: Final = await asyncio.gather(*(journal.store.append(journal.receipt) for _ in range(12)))
    assert all(isinstance(result, CommittedCharge) for result in results), results
    assert sum(isinstance(result, CommittedCharge) and not result.duplicate for result in results) == 1
    balance: Final = await journal.balance()
    assert balance.spend == balance.total_spend == 0.25
    assert await journal.receipt_count() == 1
    conflict: Final = await journal.store.append(journal.receipt.model_copy(update={"amount_usd": "0.5"}))
    assert conflict == FailedCharge(journal.receipt.key_hash, "receipt_conflict")
    assert await journal.balance() == balance


async def test_concurrent_distinct_charges_do_not_lose_key_increments(journal: JournalCase) -> None:
    receipts: Final = tuple(journal.receipt.model_copy(update={"event_id": str(uuid4())}) for _ in range(16))
    results: Final = await asyncio.gather(*(journal.store.append(receipt) for receipt in receipts))
    assert all(isinstance(result, CommittedCharge) and not result.duplicate for result in results), results
    balance: Final = await journal.balance()
    assert balance.spend == balance.total_spend == 16 * 0.25
    assert await journal.receipt_count() == 16


class LostCommitAcknowledgement:
    def __init__(self, case: JournalCase) -> None:
        self.case: Final = case
        self.lost = False

    async def __call__(self, query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
        rows: Final = await self.case.database.query_raw(query, *parameters)
        if not self.lost and await self.case.receipt_count() == 1:
            self.lost = True
            raise ConnectionError("injected lost database acknowledgement")
        return rows


async def test_lost_commit_acknowledgement_does_not_rebill_on_retry(journal: JournalCase) -> None:
    store: Final = ChargeReceiptStore(LostCommitAcknowledgement(journal))
    first: Final = await store.append(journal.receipt)
    assert first == FailedCharge(journal.receipt.key_hash, "database_unavailable")
    balance: Final = await journal.balance()
    assert balance.spend == balance.total_spend == 0.25
    second: Final = await store.append(journal.receipt)
    assert second == CommittedCharge(
        journal.receipt.event_id,
        journal.receipt.key_hash,
        journal.receipt.amount_usd,
        duplicate=True,
        key_counter_updated=True,
    )
    assert await journal.balance() == balance and await journal.receipt_count() == 1


async def test_failed_key_write_rolls_back_the_receipt_in_the_same_transaction(journal: JournalCase) -> None:
    trigger: Final = "receipt_failure_" + uuid4().hex
    await journal.database.execute_raw(
        f"""CREATE FUNCTION "{trigger}"() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'injected key update failure'; END $$"""
    )
    try:
        await journal.database.execute_raw(
            f"""CREATE TRIGGER "{trigger}" BEFORE UPDATE ON "LiteLLM_VerificationToken"
                FOR EACH ROW WHEN (OLD.token='{journal.receipt.key_hash}')
                EXECUTE FUNCTION "{trigger}"()"""
        )
        try:
            outcome: Final = await journal.store.append(journal.receipt)
            assert outcome == FailedCharge(journal.receipt.key_hash, "database_unavailable")
            balance: Final = await journal.balance()
            assert balance.spend == balance.total_spend == 0
            assert await journal.receipt_count() == 0
        finally:
            await journal.database.execute_raw(f'DROP TRIGGER "{trigger}" ON "LiteLLM_VerificationToken"')
    finally:
        await journal.database.execute_raw(f'DROP FUNCTION "{trigger}"()')
    assert isinstance(await journal.store.append(journal.receipt), CommittedCharge)
    assert (await journal.balance()).spend == 0.25 and await journal.receipt_count() == 1


async def test_delivery_lease_expires_and_stale_ack_cannot_remove_new_claim(journal: JournalCase) -> None:
    assert isinstance(await journal.store.append(journal.receipt), CommittedCharge)
    first_owner: Final = str(uuid4())
    next_owner: Final = str(uuid4())
    assert await journal.store.claim(first_owner, 32) == (journal.receipt,)
    assert await journal.store.claim(next_owner, 32) == ()
    await journal.database.execute_raw(
        """UPDATE "LiteLLM_ChargeReceipt" SET lease_until=clock_timestamp()-interval '1 second'
           WHERE event_id=$1""",
        journal.receipt.event_id,
    )
    assert await journal.store.claim(next_owner, 32) == (journal.receipt,)
    await journal.store.acknowledge(first_owner, (journal.receipt.event_id,))
    rows: Final = await journal.database.query_raw(
        'SELECT delivered_at FROM "LiteLLM_ChargeReceipt" WHERE event_id=$1', journal.receipt.event_id
    )
    assert rows[0]["delivered_at"] is None
    await journal.store.acknowledge(next_owner, (journal.receipt.event_id,))
    delivered: Final = await journal.database.query_raw(
        'SELECT delivered_at FROM "LiteLLM_ChargeReceipt" WHERE event_id=$1', journal.receipt.event_id
    )
    assert delivered[0]["delivered_at"] is not None
    assert await journal.store.claim(first_owner, 32) == ()
    assert (await journal.balance()).spend == 0.25


async def test_retry_retains_charge_without_rebilling_and_zero_is_a_fresh_receipt(journal: JournalCase) -> None:
    assert isinstance(await journal.store.append(journal.receipt), CommittedCharge)
    owner: Final = str(uuid4())
    assert await journal.store.claim(owner, 32) == (journal.receipt,)
    await journal.store.retry(owner, (journal.receipt.event_id,), "receiver_unavailable")
    assert await journal.store.claim(owner, 32) == ()
    await journal.database.execute_raw(
        'UPDATE "LiteLLM_ChargeReceipt" SET retry_at=clock_timestamp() WHERE event_id=$1',
        journal.receipt.event_id,
    )
    assert await journal.store.claim(owner, 32) == (journal.receipt,)
    zero: Final = journal.receipt.model_copy(update={"event_id": str(uuid4()), "amount_usd": "0"})
    assert isinstance(await journal.store.append(zero), CommittedCharge)
    assert await journal.receipt_count() == 2
    balance: Final = await journal.balance()
    assert balance.spend == balance.total_spend == 0.25


async def test_key_removal_retains_original_charge_evidence_without_recreating_key(journal: JournalCase) -> None:
    await journal.database.execute_raw(
        'DELETE FROM "LiteLLM_VerificationToken" WHERE token=$1', journal.receipt.key_hash
    )
    outcome: Final = await journal.store.append(journal.receipt)
    assert outcome == CommittedCharge(
        journal.receipt.event_id,
        journal.receipt.key_hash,
        journal.receipt.amount_usd,
        duplicate=False,
        key_counter_updated=False,
    )
    assert await journal.receipt_count() == 1
    keys: Final = await journal.database.query_raw(
        'SELECT token FROM "LiteLLM_VerificationToken" WHERE token=$1', journal.receipt.key_hash
    )
    assert keys == []


def priced_event() -> SpendEvent:
    stamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event: Final = project_spend_event(
        {
            "model": "owned-billing-model",
            "litellm_params": {},
            "response_cost": 0.25,
            "standard_logging_object": {
                "response_cost": 0.25,
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
        {"id": "owned-response", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
        stamp,
        stamp,
        store_bodies=False,
    )
    assert isinstance(event, SpendEvent)
    return event


class HeldDatabaseQuery:
    def __init__(self, database: Prisma) -> None:
        self.database: Final = database
        self.entered: Final = asyncio.Event()
        self.release: Final = asyncio.Event()

    async def __call__(self, query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
        self.entered.set()
        await self.release.wait()
        return await self.database.query_raw(query, *parameters)


async def test_client_cancellation_and_shutdown_drain_preserve_started_charge_capture(journal: JournalCase) -> None:
    query: Final = HeldDatabaseQuery(journal.database)
    runtime: Final = ChargeReceiptRuntime(
        ChargeReceiptSettings(gateway_scope=journal.receipt.gateway_scope),
        ChargeReceiptStore(query),
    )
    capture: Final = runtime.bind(
        BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
    )
    caller: Final = asyncio.create_task(capture.capture_event(priced_event()))
    await asyncio.wait_for(query.entered.wait(), timeout=5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    closing: Final = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not closing.done() and await journal.receipt_count() == 0
    query.release.set()
    await closing
    assert await journal.receipt_count() == 1
    assert (await journal.balance()).spend == 0.25
    assert await capture.wait()


async def test_late_known_usage_can_settle_an_earlier_unpriced_failure(journal: JournalCase) -> None:
    runtime: Final = ChargeReceiptRuntime(
        ChargeReceiptSettings(gateway_scope=journal.receipt.gateway_scope),
        journal.store,
    )
    capture: Final = runtime.bind(
        BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
    )
    stamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert isinstance(await capture.capture_failure({}, 0, stamp, stamp), FailedCharge)
    assert await journal.receipt_count() == 0
    assert isinstance(await capture.capture_event(priced_event()), CommittedCharge)
    assert await journal.receipt_count() == 1
    assert (await journal.balance()).spend == 0.25


SECRET: Final = "receipt-integration-secret-no-production"


def receipt_settings(journal: JournalCase) -> ChargeReceiptSettings:
    return ChargeReceiptSettings(
        enabled=True,
        gateway_scope=journal.receipt.gateway_scope,
        key_hashes=frozenset((journal.receipt.key_hash,)),
        destination_url="https://receiver.example.test/billing/receipts",
        destination_secret=SecretStr(SECRET),
    )


@pytest.mark.parametrize(
    "changed",
    ("none", "signature", "body", "key", "scope", "path", "method", "expired", "background", "unsupported", "custom"),
)
async def test_signed_context_is_bound_to_exact_authorized_request(journal: JournalCase, changed: str) -> None:
    body: Final = json.dumps({"model": "owned-model", "background": changed == "background"}).encode()
    issued: Final = int(time.time()) - (600 if changed == "expired" else 0)
    path: Final = (
        "/v1/embeddings"
        if changed == "unsupported"
        else "/custom-provider-call"
        if changed == "custom"
        else "/v1/responses"
    )
    token: Final = jwt.encode(
        {
            "version": 1,
            "sub": journal.receipt.request_id,
            "aud": "litellm-charge-receipts",
            "iat": issued,
            "exp": issued + 300,
            "gateway_scope": "c" * 64 if changed == "scope" else journal.receipt.gateway_scope,
            "key_hash": "c" * 64 if changed == "key" else journal.receipt.key_hash,
            "body_sha256": sha256(b"changed" if changed == "body" else body).hexdigest(),
            "method": "GET" if changed == "method" else "POST",
            "route": "/v1/chat/completions" if changed == "path" else path,
        },
        "wrong-receipt-secret-no-production" if changed == "signature" else SECRET,
        algorithm="HS256",
    )

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": path,
            "query_string": b"",
            "headers": [(b"x-litellm-billing-context", token.encode())],
            "server": ("gateway.example.test", 443),
        },
        receive,
    )
    runtime: Final = ChargeReceiptRuntime(receipt_settings(journal), journal.store)
    auth: Final = UserAPIKeyAuth(api_key=journal.receipt.key_hash)
    if changed == "none":
        await authorize_charge_receipt(request, auth, runtime, json.loads(body))
        assert request.state.charge_receipt_context.request_id == journal.receipt.request_id
    else:
        with pytest.raises(HTTPException) as error:
            await authorize_charge_receipt(request, auth, runtime, json.loads(body))
        assert error.value.status_code == 400
        assert not hasattr(request.state, "charge_receipt_context")
    assert await journal.receipt_count() == 0 and (await journal.balance()).spend == 0


@pytest.mark.parametrize("ack", ("valid", "unknown", "missing", "unavailable"))
async def test_publisher_retains_every_unacknowledged_charge(journal: JournalCase, ack: str) -> None:
    assert isinstance(await journal.store.append(journal.receipt), CommittedCharge)

    def receiver(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://receiver.example.test/billing/receipts"
        assert request.headers["authorization"] == "Bearer " + SECRET
        assert json.loads(request.content) == {"events": [journal.receipt.model_dump()]}
        return httpx.Response(
            503 if ack == "unavailable" else 200,
            json={
                "acknowledged": (
                    [journal.receipt.event_id] if ack == "valid" else ["unrelated"] if ack == "unknown" else []
                )
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
        publisher: Final = ChargeReceiptPublisher(journal.store, receipt_settings(journal), client)
        delivered: Final = await publisher.deliver_once()
    rows: Final = await journal.database.query_raw(
        'SELECT delivered_at,last_error,lease_owner FROM "LiteLLM_ChargeReceipt" WHERE event_id=$1',
        journal.receipt.event_id,
    )
    assert delivered == (1 if ack == "valid" else 0)
    assert (rows[0]["delivered_at"] is not None) == (ack == "valid")
    assert (rows[0]["last_error"] is None) == (ack == "valid")
    assert rows[0]["lease_owner"] is None
    assert await journal.receipt_count() == 1 and (await journal.balance()).spend == 0.25


async def test_stream_chunk_does_not_commit_zero_before_final_usage(journal: JournalCase) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

    runtime: Final = ChargeReceiptRuntime(receipt_settings(journal), journal.store)
    logging: Final = Logging(
        model="owned-billing-model",
        messages=[{"role": "user", "content": "cached reply"}],
        stream=True,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id=str(uuid4()),
        function_id=str(uuid4()),
    )
    logging.required_charge_capture = runtime.bind(
        BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
    )
    await logging.async_success_handler(
        result=ModelResponseStream(
            model="owned-billing-model",
            choices=[{"index": 0, "delta": {"content": "cached"}, "finish_reason": None}],
        ),
        cache_hit=True,
    )
    assert await journal.receipt_count() == 0
    await logging.async_success_handler(
        result=ModelResponse(
            model="owned-billing-model",
            usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            choices=[
                {"index": 0, "message": {"role": "assistant", "content": "cached reply"}, "finish_reason": "stop"}
            ],
        ),
        cache_hit=True,
    )
    assert await journal.receipt_count() == 1
    assert (await journal.balance()).spend == 0


class ObservedPollingQuery:
    def __init__(self, database: Prisma, stop: asyncio.Event) -> None:
        self.database: Final = database
        self.stop: Final = stop
        self.calls = 0

    async def __call__(self, query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
        rows: Final = await self.database.query_raw(query, *parameters)
        self.calls += 1
        if self.calls >= 2:
            self.stop.set()
        return rows


async def test_idle_publisher_continues_after_poll_timeout(journal: JournalCase) -> None:
    stop: Final = asyncio.Event()
    query: Final = ObservedPollingQuery(journal.database, stop)
    async with httpx.AsyncClient() as client:
        publisher: Final = ChargeReceiptPublisher(ChargeReceiptStore(query), receipt_settings(journal), client)
        await asyncio.wait_for(publisher.run(stop), timeout=5)
    assert query.calls == 2 and stop.is_set()
    assert await journal.receipt_count() == 0 and (await journal.balance()).spend == 0


@pytest.mark.parametrize("amount", (None, -1.0, float("inf"), float("nan")))
async def test_unpriced_or_invalid_cost_never_becomes_a_confirmed_free_charge(
    journal: JournalCase,
    amount: float | None,
) -> None:
    runtime: Final = ChargeReceiptRuntime(receipt_settings(journal), journal.store)
    capture: Final = runtime.bind(
        BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
    )
    result: Final = await capture.capture_event(priced_event().model_copy(update={"response_cost": amount}))
    assert isinstance(result, FailedCharge) and not await capture.wait()
    assert await journal.receipt_count() == 0 and (await journal.balance()).spend == 0


async def test_missing_database_ack_cannot_release_the_response_as_paid(journal: JournalCase) -> None:
    from litellm.proxy.spend_tracking.charge_receipt_request import await_charge_receipt

    query: Final = HeldDatabaseQuery(journal.database)
    runtime: Final = ChargeReceiptRuntime(
        receipt_settings(journal).model_copy(update={"commit_timeout_seconds": 0.02}),
        ChargeReceiptStore(query),
    )
    capture: Final = runtime.bind(
        BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
    )
    outcome: Final = await asyncio.wait_for(capture.capture_event(priced_event()), timeout=5)
    assert isinstance(outcome, FailedCharge) and not runtime.available
    request: Final = Request({"type": "http"})
    request.state.required_charge_capture = capture
    with pytest.raises(HTTPException) as error:
        await await_charge_receipt(request)
    assert error.value.status_code == 503 and error.value.headers == {"x-should-retry": "false"}
    assert await journal.receipt_count() == 0 and (await journal.balance()).spend == 0


async def test_unavailable_cached_budget_counters_do_not_drop_the_durable_charge(journal: JournalCase) -> None:
    async def unavailable_cache(details: Mapping[str, object], event: SpendEvent, charge: object) -> None:
        assert event.response_cost == 0.25
        raise ConnectionError("injected cache outage")

    runtime: Final = ChargeReceiptRuntime(receipt_settings(journal), journal.store, finalize=unavailable_cache)
    capture: Final = runtime.bind(
        BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
    )
    assert isinstance(await capture.capture_event(priced_event(), {}), CommittedCharge)
    assert await capture.wait() and await journal.receipt_count() == 1
    assert (await journal.balance()).spend == 0.25


async def test_required_partial_failure_updates_rollups_once_with_error_logging_disabled(
    journal: JournalCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm import Usage
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.utils import ProxyLogging
    from tests.integration._support.prisma_clients import connected_proxy_prisma

    identity: Final = str(uuid4())
    await journal.database.execute_raw(
        'INSERT INTO "LiteLLM_UserTable" (user_id,models) VALUES ($1,ARRAY[]::text[])', identity
    )
    await journal.database.execute_raw(
        'INSERT INTO "LiteLLM_TeamTable" (team_id,admins,members,models) '
        "VALUES ($1,ARRAY[]::text[],ARRAY[]::text[],ARRAY[]::text[])",
        identity,
    )
    try:
        async with connected_proxy_prisma() as client:
            logging: Final = ProxyLogging(UserApiKeyCache())
            monkeypatch.setattr(proxy_server, "prisma_client", client)
            monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging)
            monkeypatch.setattr(proxy_server, "disable_spend_logs", True)
            monkeypatch.setattr(proxy_server, "general_settings", {"disable_error_logs": True})
            runtime: Final = ChargeReceiptRuntime(
                receipt_settings(journal), journal.store, finalize=ProxyDBLogger().finalize_required_charge
            )
            call: Final = Logging(
                model=journal.receipt.model,
                messages=[],
                stream=True,
                call_type="acompletion",
                start_time=time.time(),
                litellm_call_id=str(uuid4()),
                function_id=str(uuid4()),
            )
            call.required_charge_capture = runtime.bind(
                BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
            )
            auth: Final = UserAPIKeyAuth(
                api_key=journal.receipt.key_hash,
                user_id=identity,
                team_id=identity,
                request_route="/v1/chat/completions",
            )
            for _ in range(2):
                await ProxyDBLogger().async_post_call_failure_hook(
                    request_data={
                        "model": journal.receipt.model,
                        "response_cost": 0.25,
                        "litellm_logging_obj": call,
                        "combined_usage_object": Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                    },
                    original_exception=ConnectionError("provider stream interrupted"),
                    user_api_key_dict=auth,
                )
            deadline: Final = time.monotonic() + 10
            while True:
                await logging.db_spend_update_writer.db_update_spend_transaction_handler(client, 1, logging)
                rows: Final = await journal.database.query_raw(
                    'SELECT u.spend AS user_spend,t.spend AS team_spend FROM "LiteLLM_UserTable" u '
                    'JOIN "LiteLLM_TeamTable" t ON t.team_id=u.user_id WHERE u.user_id=$1',
                    identity,
                )
                if rows == [{"user_spend": 0.25, "team_spend": 0.25}]:
                    break
                assert time.monotonic() < deadline, rows
                await asyncio.sleep(0.01)
            assert (await journal.balance()).spend == 0.25
            cached_spend, _ = await proxy_server.read_spend_counter_cache_value("spend:key:" + journal.receipt.key_hash)
            assert cached_spend == 0.25
            assert await journal.receipt_count() == 1
    finally:
        await journal.database.execute_raw('DELETE FROM "LiteLLM_TeamMembership" WHERE team_id=$1', identity)
        await journal.database.execute_raw('DELETE FROM "LiteLLM_TeamTable" WHERE team_id=$1', identity)
        await journal.database.execute_raw('DELETE FROM "LiteLLM_UserTable" WHERE user_id=$1', identity)
