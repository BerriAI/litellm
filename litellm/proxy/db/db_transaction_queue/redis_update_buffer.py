"""Buffer SpendLogs and tool discovery rows in Redis."""

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from redis.exceptions import RedisError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    MAX_REDIS_BUFFER_DEQUEUE_COUNT,
    REDIS_SPEND_LOGS_BUFFER_KEY,
    REDIS_SPEND_LOGS_BUFFER_MAX_ROWS,
    REDIS_TOOL_DISCOVERY_BUFFER_KEY,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    ToolDiscoveryQueueItem,
)
from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import (
    ToolDiscoveryQueue,
    aggregate_tool_discoveries,
)
from litellm.proxy.db.spend_log_batching import SpendLogRow
from litellm.secret_managers.main import str_to_bool
from litellm.types.caching import (
    RedisPipelineLpopOperation,
    RedisPipelineRpushOperation,
)

SPEND_QUEUE_TARGET: Final = "spend_queue"

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any

_DECODED_JSON: Final = TypeAdapter(object)


def _zero_for_non_finite_constant(constant: str) -> float:
    verbose_proxy_logger.warning(
        "Redis spend buffer - reading a parked %s spend as 0, the spend it held is lost", constant
    )
    return 0.0


_load_buffered_transaction: Final = partial(json.loads, parse_constant=_zero_for_non_finite_constant)


def _spend_log_json_default(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _encode_spend_log_row(row: SpendLogRow) -> str:
    return json.dumps(row, default=_spend_log_json_default)


def _decode_spend_log_row(encoded: str) -> dict[str, object] | None:
    decoded: Final = _DECODED_JSON.validate_python(_load_buffered_transaction(encoded))
    return decoded if isinstance(decoded, dict) else None


_TOOL_DISCOVERY_ITEMS: Final = TypeAdapter(list[ToolDiscoveryQueueItem])


class RedisUpdateBuffer:
    """
    Handles buffering database `UPDATE` transactions in Redis before committing them to the database

    This is to prevent deadlocks and improve reliability
    """

    def __init__(
        self,
        redis_cache: RedisCache | None = None,
    ):
        self.redis_cache = redis_cache

    @staticmethod
    def should_commit_spend_updates_to_redis() -> bool:
        """
        Checks if the Pod should commit spend updates to Redis

        This setting enables buffering database transactions in Redis
        to improve reliability and reduce database contention
        """
        from litellm.proxy.proxy_server import general_settings

        _use_redis_transaction_buffer: bool | str | None = general_settings.get("use_redis_transaction_buffer", False)
        if isinstance(_use_redis_transaction_buffer, str):
            _use_redis_transaction_buffer = str_to_bool(_use_redis_transaction_buffer)
        if _use_redis_transaction_buffer is None:
            return False
        return _use_redis_transaction_buffer

    _should_commit_spend_updates_to_redis = should_commit_spend_updates_to_redis

    @with_service_target(SPEND_QUEUE_TARGET)
    async def store_tool_discoveries_in_redis(self, tool_discovery_queue: ToolDiscoveryQueue) -> None:
        """Move this worker's pending tool rows to Redis so only the leader writes LiteLLM_ToolTable."""
        if self.redis_cache is None:
            return
        items: Final = tool_discovery_queue.flush()
        if not items:
            return
        try:
            _ = await self.redis_cache.async_rpush_pipeline(
                rpush_list=[
                    RedisPipelineRpushOperation(key=REDIS_TOOL_DISCOVERY_BUFFER_KEY, values=[safe_dumps(items)])
                ]
            )
        except (RedisError, OSError, asyncio.TimeoutError) as e:
            verbose_proxy_logger.error(
                "Tool discovery - failed to push %d tool rows to Redis, restoring them for the next tick. Error: %s",
                len(items),
                str(e),
            )
            for item in items:
                tool_discovery_queue.add_update(item)

    @with_service_target(SPEND_QUEUE_TARGET)
    async def restore_tool_discoveries_to_redis(self, items: Sequence[ToolDiscoveryQueueItem]) -> None:
        """Put drained tool rows back for the next drain after a write that provably did not apply."""
        if self.redis_cache is None or not items:
            return
        _ = await self.redis_cache.async_rpush_pipeline(
            rpush_list=[
                RedisPipelineRpushOperation(key=REDIS_TOOL_DISCOVERY_BUFFER_KEY, values=[safe_dumps(list(items))])
            ]
        )

    @with_service_target(SPEND_QUEUE_TARGET)
    async def get_all_tool_discoveries_from_redis_buffer(self) -> tuple[ToolDiscoveryQueueItem, ...]:
        """Drain every worker's tool rows and fold them to one row per tool_name."""
        if self.redis_cache is None:
            return ()
        popped: Final = await self.redis_cache.async_lpop_pipeline(
            lpop_list=[
                RedisPipelineLpopOperation(key=REDIS_TOOL_DISCOVERY_BUFFER_KEY, count=MAX_REDIS_BUFFER_DEQUEUE_COUNT)
            ]
        )
        entries: Final = popped[0] if popped else None
        if not entries:
            return ()
        batches: Final = tuple(_TOOL_DISCOVERY_ITEMS.validate_json(entry) for entry in entries)
        return aggregate_tool_discoveries(chain.from_iterable(batches))

    @with_service_target(SPEND_QUEUE_TARGET)
    async def store_spend_logs_in_redis(
        self,
        rows: Sequence[SpendLogRow],
        max_rows: int = REDIS_SPEND_LOGS_BUFFER_MAX_ROWS,
    ) -> bool:
        """Park spend-log rows in Redis so they outlive this pod, dropping the oldest past ``max_rows``."""
        if self.redis_cache is None or len(rows) == 0 or not self.should_commit_spend_updates_to_redis():
            return False
        try:
            buffer_size: Final = await self.redis_cache.async_rpush_and_trim(
                key=REDIS_SPEND_LOGS_BUFFER_KEY,
                values=tuple(_encode_spend_log_row(row) for row in rows),
                max_len=max_rows,
            )
            overflow: Final = buffer_size - max_rows
            if overflow > 0:
                verbose_proxy_logger.error(
                    "Spend tracking - Redis spend log buffer is at its %d row cap; dropped the %d oldest spend logs",
                    max_rows,
                    overflow,
                )
        except Exception as e:  # noqa: BLE001  # the caller falls back to the in-memory queue on any Redis fault
            verbose_proxy_logger.error(
                "Spend tracking - failed to park %d spend log rows in Redis. Error: %s", len(rows), str(e)
            )
            return False
        verbose_proxy_logger.info("Spend tracking - parked %d spend log rows in Redis for a later flush", len(rows))
        return True

    @with_service_target(SPEND_QUEUE_TARGET)
    async def get_spend_logs_from_redis_buffer(self, limit: int) -> tuple[dict[str, object], ...]:
        """Atomically take up to ``limit`` parked spend-log rows out of Redis."""
        if self.redis_cache is None or not self.should_commit_spend_updates_to_redis():
            return ()
        popped: Final[str | list[str] | None] = await self.redis_cache.async_lpop(
            key=REDIS_SPEND_LOGS_BUFFER_KEY,
            count=limit,
        )
        if popped is None:
            return ()
        encoded_rows: Final = tuple(popped) if isinstance(popped, list) else (popped,)
        decoded_rows: Final = (_decode_spend_log_row(encoded) for encoded in encoded_rows)
        return tuple(row for row in decoded_rows if row is not None)
