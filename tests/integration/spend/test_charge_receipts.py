import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma
from pydantic import SecretStr

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.charge_receipt_capture import BillingContext, ChargeReceiptRuntime
from litellm.proxy.spend_tracking.charge_receipts import (
    ChargeReceipt,
    ChargeReceiptSettings,
    ChargeReceiptStore,
)
from litellm.types.charge_receipts import CHARGE_OUTCOME_KEY, CommittedCharge, FailedCharge
from tests.integration._support.prisma_clients import connected_prisma

pytestmark = pytest.mark.asyncio(loop_scope="session")


@dataclass(frozen=True, slots=True)
class JournalCase:
    database: Prisma
    store: ChargeReceiptStore
    receipt: ChargeReceipt

    async def balance(self) -> dict[str, float]:
        rows: Final = await self.database.query_raw(
            'SELECT spend,total_spend FROM "LiteLLM_VerificationToken" WHERE token=$1', self.receipt.key_hash
        )
        assert len(rows) == 1
        return {field: float(rows[0][field]) for field in ("spend", "total_spend")}

    async def receipt_count(self) -> int:
        rows: Final = await self.database.query_raw(
            'SELECT count(*) AS total FROM "LiteLLM_ChargeReceipt" WHERE key_hash=$1', self.receipt.key_hash
        )
        return int(rows[0]["total"])


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
        await database.execute_raw('DELETE FROM "LiteLLM_SpendLogs" WHERE api_key=$1', key_hash)
        await database.execute_raw('DELETE FROM "LiteLLM_ChargeReceipt" WHERE key_hash=$1', key_hash)
        await database.execute_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE token=$1', key_hash)


async def test_concurrent_receipt_retries_post_one_key_charge(journal: JournalCase) -> None:
    results: Final = await asyncio.gather(*(journal.store.append(journal.receipt) for _ in range(12)))
    assert all(isinstance(result, CommittedCharge) for result in results), results
    assert sum(isinstance(result, CommittedCharge) and not result.duplicate for result in results) == 1
    balance: Final = await journal.balance()
    assert balance["spend"] == balance["total_spend"] == 0.25
    assert await journal.receipt_count() == 1
    conflict: Final = await journal.store.append(journal.receipt.model_copy(update={"amount_usd": "0.5"}))
    assert conflict == FailedCharge(journal.receipt.key_hash, "receipt_conflict")
    assert await journal.balance() == balance


async def test_concurrent_distinct_charges_do_not_lose_key_increments(journal: JournalCase) -> None:
    receipts: Final = tuple(journal.receipt.model_copy(update={"event_id": str(uuid4())}) for _ in range(16))
    results: Final = await asyncio.gather(*(journal.store.append(receipt) for receipt in receipts))
    assert all(isinstance(result, CommittedCharge) and not result.duplicate for result in results), results
    balance: Final = await journal.balance()
    assert balance["spend"] == balance["total_spend"] == 16 * 0.25
    assert await journal.receipt_count() == 16


class ReceiptWriteFault:
    def __init__(self, case: JournalCase, failure: str) -> None:
        self.case: Final = case
        self.failure: Final = failure

    async def __call__(self, query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
        if self.failure == "timeout":
            await asyncio.Event().wait()
        if self.failure == "unwritten":
            raise ConnectionError("receipt database unreachable")
        rows: Final = await self.case.database.query_raw(query, *parameters)
        if self.failure == "lost_ack":
            raise ConnectionError("injected lost database acknowledgement")
        return rows


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
            assert balance["spend"] == balance["total_spend"] == 0
            assert await journal.receipt_count() == 0
        finally:
            await journal.database.execute_raw(f'DROP TRIGGER "{trigger}" ON "LiteLLM_VerificationToken"')
    finally:
        await journal.database.execute_raw(f'DROP FUNCTION "{trigger}"()')
    assert isinstance(await journal.store.append(journal.receipt), CommittedCharge)
    assert (await journal.balance())["spend"] == 0.25 and await journal.receipt_count() == 1


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
    assert (await journal.balance())["spend"] == 0.25


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
    assert balance["spend"] == balance["total_spend"] == 0.25


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


SECRET: Final = "receipt-integration-secret-no-production"


def receipt_settings(journal: JournalCase) -> ChargeReceiptSettings:
    return ChargeReceiptSettings(
        enabled=True,
        gateway_scope=journal.receipt.gateway_scope,
        key_hashes=frozenset((journal.receipt.key_hash,)),
        destination_url="https://receiver.example.test/billing/receipts",
        destination_secret=SecretStr(SECRET),
    )


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
    assert (await journal.balance())["spend"] == 0


@pytest.mark.parametrize("failure", ("none", "unwritten", "lost_ack", "timeout"))
async def test_receipt_failure_preserves_other_accounting_without_rebilling_the_key(
    journal: JournalCase, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from litellm import Usage
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.utils import ProxyLogging, update_spend
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
            monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
            monkeypatch.setattr(proxy_server, "general_settings", {"disable_error_logs": True})
            runtime: Final = ChargeReceiptRuntime(
                receipt_settings(journal),
                ChargeReceiptStore(ReceiptWriteFault(journal, failure)),
                finalize=ProxyDBLogger().finalize_required_charge,
            )
            logging.charge_receipts = runtime
            call: Final = Logging(
                model=journal.receipt.model,
                messages=[],
                stream=True,
                call_type="acompletion",
                start_time=time.time(),
                litellm_call_id=str(uuid4()),
                function_id=str(uuid4()),
            )
            capture: Final = runtime.bind(
                BillingContext(request_id=journal.receipt.request_id, key_hash=journal.receipt.key_hash)
            )
            call.required_charge_capture = capture
            auth: Final = UserAPIKeyAuth(
                api_key=journal.receipt.key_hash,
                user_id=identity,
                team_id=identity,
                request_route="/v1/chat/completions",
            )
            details: Final = {
                "model": journal.receipt.model,
                "litellm_call_id": identity,
                "response_cost": 0.25,
                "litellm_logging_obj": call,
                "combined_usage_object": Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            }
            for _ in range(2):
                await ProxyDBLogger().async_post_call_failure_hook(
                    request_data=details,
                    original_exception=ConnectionError("provider stream interrupted"),
                    user_api_key_dict=auth,
                )
            outcome: Final = capture.result
            assert isinstance(outcome, CommittedCharge if failure == "none" else FailedCharge)
            await ProxyDBLogger().async_log_success_event(
                {**details, CHARGE_OUTCOME_KEY: outcome}, None, datetime.now(timezone.utc), datetime.now(timezone.utc)
            )
            deadline: Final = time.monotonic() + 10
            while True:
                await update_spend(client, None, logging)
                rows: Final = await journal.database.query_raw(
                    'SELECT u.spend AS user_spend,t.spend AS team_spend FROM "LiteLLM_UserTable" u '
                    'JOIN "LiteLLM_TeamTable" t ON t.team_id=u.user_id WHERE u.user_id=$1',
                    identity,
                )
                if rows == [{"user_spend": 0.25, "team_spend": 0.25}]:
                    break
                assert time.monotonic() < deadline, rows
                await asyncio.sleep(0.01)
            committed: Final = failure in ("none", "lost_ack")
            assert (await journal.balance())["spend"] == (0.25 if committed else 0)
            logs: Final = await journal.database.query_raw(
                'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=$1', journal.receipt.key_hash
            )
            assert logs == [{"spend": 0.25}]
            cached_spend, _ = await proxy_server.read_spend_counter_cache_value("spend:key:" + journal.receipt.key_hash)
            assert cached_spend == 0.25
            assert await journal.receipt_count() == int(committed)
            receipt: Final = capture.receipt
            assert receipt is not None
            replay: Final = await journal.store.append(receipt)
            assert isinstance(replay, CommittedCharge) and replay.duplicate == committed
            assert (await journal.balance())["spend"] == 0.25
            await logging.db_spend_update_writer.update_database(
                token=journal.receipt.key_hash,
                response_cost=0.25,
                user_id=None,
                end_user_id=None,
                team_id=None,
                org_id=None,
                kwargs={**details, "litellm_call_id": str(uuid4())},
                completion_response=None,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
            )
            while (await journal.balance())["spend"] != 0.5:
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
                await logging.db_spend_update_writer.db_update_spend_transaction_handler(client, 1, logging)
            assert await journal.receipt_count() == 1
    finally:
        await journal.database.execute_raw('DELETE FROM "LiteLLM_TeamMembership" WHERE team_id=$1', identity)
        await journal.database.execute_raw('DELETE FROM "LiteLLM_TeamTable" WHERE team_id=$1', identity)
        await journal.database.execute_raw('DELETE FROM "LiteLLM_UserTable" WHERE user_id=$1', identity)
