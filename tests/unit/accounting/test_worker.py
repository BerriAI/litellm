import json
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeRedis as FakeSyncRedis
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from pydantic import TypeAdapter

from litellm import in_memory_llm_clients_cache
from litellm.accounting.application import apply_deliveries
from litellm.accounting.worker import flush_accounting_deliveries
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.customer_usage import CUSTOMER_USAGE_METADATA_KEY, CustomerUsageAttribution
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.accounting_delivery_repository import (
    AccountingDeliveryRepository,
    DeliveryProjection,
    LedgerTransaction,
)

_FIELDS: Final = TypeAdapter(dict[str, object])
_INCREMENTS: Final = TypeAdapter(dict[str, dict[str, float]])
_PAYLOAD: Final = json.dumps(
    {"request_id": "completed-create", "spend": 0.25, "metadata": {"accounting_entities": {"api_key": "creator"}}}
)


@dataclass
class _Ledger:
    pending: bool
    unavailable: bool
    spend: float = 0.0
    attempts: int = 0

    async def query(self, query: str, *args: object) -> tuple[Mapping[str, object], ...]:
        if query.startswith("SELECT delivery_id, attempts"):
            if not self.pending or self.attempts or args[1] is True:
                return ()
            return ({"delivery_id": "delivery", "attempts": self.attempts, "accounting_payload": _PAYLOAD},)
        if "SET attempts = attempts + 1" in query:
            self.attempts += 1
            return ({"delivery_id": "delivery", "attempts": self.attempts},)
        if "pending_count" in query:
            return ({"pending_count": int(self.pending), "parked_count": 0, "oldest_pending_seconds": 0},)
        return ({"batch_count": 0, "table_bytes": 0},)

    async def execute(self, query: str, *args: object) -> int:
        if "SET state = 'applied'" in query:
            self.pending = False
        return 0

    def update_key(self, *, where: object, data: object) -> None:
        if self.unavailable:
            raise OSError("database connection lost before the update")
        assert _FIELDS.validate_python(where) == {"token": "creator"}
        self.spend += _INCREMENTS.validate_python({"spend": _FIELDS.validate_python(data)["spend"]})["spend"][
            "increment"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("locked", [False, True], ids=["without-redis", "orphaned-redis-lock"])
@pytest.mark.parametrize("scenario", ["charge", "empty", "unavailable"])
async def test_delivery_charges_the_creating_key_despite_an_orphaned_maintenance_lock(
    locked: bool, scenario: Literal["charge", "empty", "unavailable"]
) -> None:
    ledger: Final = _Ledger(pending=scenario != "empty", unavailable=scenario == "unavailable")
    database: Final = MagicMock(spec=PrismaClient)
    transaction: Final = MagicMock()
    transaction.query_raw = AsyncMock(side_effect=ledger.query)
    transaction.execute_raw = AsyncMock(side_effect=ledger.execute)
    batcher: Final = MagicMock()
    batcher.litellm_verificationtoken.update_many.side_effect = ledger.update_key
    transaction.batch_.return_value.__aenter__ = AsyncMock(return_value=batcher)
    transaction.batch_.return_value.__aexit__ = AsyncMock(return_value=False)
    database.tx.return_value.__aenter__ = AsyncMock(return_value=transaction)
    database.tx.return_value.__aexit__ = AsyncMock(return_value=False)
    writer: Final = DBSpendUpdateWriter(accounting_clock=lambda: 0.0)
    logger: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    server: Final = FakeServer()
    redis: Final = FakeRedis(server=server)
    sync_redis: Final = FakeSyncRedis(server=server)
    cache: Final = RedisCache(host="accounting-unit.example", connection_pool=sync_redis.connection_pool)
    cache_key: Final = cache._get_async_client_cache_key()
    in_memory_llm_clients_cache.set_cache(key=cache_key, value=redis)
    await redis.set("cronjob_lock:accounting_delivery", json.dumps("terminated-worker"), ex=60)
    try:
        if locked:
            writer.pod_lock_manager.redis_cache = cache
        await flush_accounting_deliveries(writer, database, logger)
        await flush_accounting_deliveries(writer, database, logger)

        assert ledger.spend == (0.25 if scenario == "charge" else 0.0)
        assert ledger.pending is (scenario == "unavailable")
        assert ledger.attempts == (1 if scenario == "unavailable" else 0)
        assert await redis.get("cronjob_lock:accounting_delivery") == json.dumps("terminated-worker").encode()
    finally:
        in_memory_llm_clients_cache.delete_cache(key=cache_key)
        await redis.aclose()
        sync_redis.close()


class _CustomerLedger:
    def __init__(self, fail_acknowledgement: bool) -> None:
        self.pending = True
        self.attempts = 0
        self.fail_acknowledgement = fail_acknowledgement
        self.daily: Mapping[str, int | float] = {}
        self.entities: Mapping[str, float] = {}

    async def query(self, query: str, *args: object) -> tuple[Mapping[str, object], ...]:
        if query.startswith("SELECT delivery_id, attempts"):
            if not self.pending or (self.attempts > 0) is not args[1]:
                return ()
            return (
                {
                    "delivery_id": "customer-delivery",
                    "attempts": self.attempts,
                    "accounting_payload": _CUSTOMER_PAYLOAD,
                },
            )
        if "SET attempts = attempts + 1" in query:
            self.attempts += 1
            return ({"delivery_id": "customer-delivery", "attempts": self.attempts},)
        return ()

    async def execute(self, query: str, *args: object) -> int:
        if query.startswith('INSERT INTO "LiteLLM_DailyCustomerMemberSpend"'):
            columns: Final = re.findall(r'"([a-z_]+)"', query.split("VALUES", 1)[0])[:-1]
            values: Final = dict(zip(columns, args, strict=True))
            self.daily = {
                name: self.daily.get(name, 0) + value
                for name, value in values.items()
                if isinstance(value, (int, float))
            }
            return 1
        if "SET state = 'applied'" not in query:
            return 0
        if self.fail_acknowledgement:
            self.fail_acknowledgement = False
            raise OSError("database connection lost before delivery acknowledgement")
        self.pending = False
        return 1

    def update_entity(self, entity: str, *, where: object, data: object) -> int:
        spend: Final = _INCREMENTS.validate_python(data)["spend"]["increment"]
        self.entities = {**self.entities, entity: self.entities.get(entity, 0.0) + spend}
        return 1


_CUSTOMER_PAYLOAD: Final = json.dumps(
    {
        "request_id": "customer-completed-request",
        "startTime": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "api_key": "hashed-key",
        "model": "usage-model",
        "model_group": "usage-model",
        "custom_llm_provider": "openai",
        "spend": 0.75,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "metadata": {
            CUSTOMER_USAGE_METADATA_KEY: CustomerUsageAttribution(
                customer_org_id="customer-org", customer_team_id="customer-team", customer_member_id="customer-member"
            ).model_dump_json()
        },
    }
)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_acknowledgement", (False, True))
async def test_customer_daily_and_lifetime_totals_commit_once_after_delivery_replay(
    fail_acknowledgement: bool,
) -> None:
    ledger: Final = _CustomerLedger(fail_acknowledgement)
    transaction: Final = MagicMock()
    transaction.query_raw = AsyncMock(side_effect=ledger.query)
    transaction.execute_raw = AsyncMock(side_effect=ledger.execute)
    batcher: Final = MagicMock()
    batcher.litellm_customerorganizationtable.update_many.side_effect = partial(ledger.update_entity, "organization")
    batcher.litellm_customerteamtable.update_many.side_effect = partial(ledger.update_entity, "team")
    batcher.litellm_customermembertable.update_many.side_effect = partial(ledger.update_entity, "member")
    transaction.batch_.return_value.__aenter__ = AsyncMock(return_value=batcher)
    transaction.batch_.return_value.__aexit__ = AsyncMock(return_value=False)

    @asynccontextmanager
    async def database_transaction(*, timeout: object) -> AsyncIterator[LedgerTransaction]:
        before: Final = (ledger.pending, ledger.daily, ledger.entities)
        try:
            yield transaction
        except Exception:
            ledger.pending, ledger.daily, ledger.entities = before
            raise

    database: Final = MagicMock(spec=PrismaClient)
    database.tx.side_effect = database_transaction
    repository: Final = AccountingDeliveryRepository(database)
    writer: Final = DBSpendUpdateWriter(accounting_clock=lambda: 0.0)
    logger: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())

    async def apply(tx: LedgerTransaction, deliveries: tuple[DeliveryProjection, ...]) -> None:
        await apply_deliveries(tx, deliveries, prisma_client=database, writer=writer, proxy_logging_obj=logger)

    first: Final = await repository.apply_pending(apply)
    if fail_acknowledgement:
        assert first.failed == 1
        assert ledger.daily == {}
        assert ledger.entities == {}
        retried: Final = await repository.apply_pending(apply)
        assert retried.applied == 1
    else:
        assert first.applied == 1
    await repository.apply_pending(apply)
    await repository.apply_pending(apply)
    assert ledger.daily["spend"] == 0.75
    assert ledger.daily["prompt_tokens"] == 10
    assert ledger.daily["completion_tokens"] == 5
    assert ledger.daily["api_requests"] == 1
    assert ledger.entities == {"organization": 0.75, "team": 0.75, "member": 0.75}
    assert ledger.pending is False
