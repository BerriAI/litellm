import json
from dataclasses import dataclass
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeRedis as FakeSyncRedis
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from pydantic import TypeAdapter

from litellm import in_memory_llm_clients_cache
from litellm.accounting.worker import flush_accounting_deliveries
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.utils import PrismaClient, ProxyLogging

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

    async def query(self, query: str, *args: object) -> tuple[dict[str, object], ...]:
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
