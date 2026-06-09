from typing import TYPE_CHECKING, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.constants import REDIS_SPEND_LOG_BUFFER_KEY
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import SpendLogsPayload
from litellm.secret_managers.main import str_to_bool

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache


class SpendLogRedisBuffer:
    def __init__(self, redis_cache: Optional["RedisCache"] = None):
        self.redis_cache = redis_cache

    def is_enabled(self) -> bool:
        from typing import Union as TypingUnion

        from litellm.proxy.proxy_server import general_settings

        if self.redis_cache is None:
            return False
        _use_redis_transaction_buffer: Optional[TypingUnion[bool, str]] = (
            general_settings.get("use_redis_transaction_buffer", False)
        )
        if isinstance(_use_redis_transaction_buffer, str):
            _use_redis_transaction_buffer = str_to_bool(
                _use_redis_transaction_buffer
            )
        if _use_redis_transaction_buffer is None:
            return False
        return _use_redis_transaction_buffer

    async def buffer_spend_log_row(self, payload: SpendLogsPayload) -> None:
        if not self.is_enabled() or self.redis_cache is None:
            return

        try:
            await self.redis_cache.async_rpush(
                REDIS_SPEND_LOG_BUFFER_KEY,
                [safe_dumps(payload)],
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "SpendLogRedisBuffer: failed to buffer spend log row: %s", e
            )

    async def pop_buffered_spend_log_rows(
        self, max_rows: int
    ) -> List[SpendLogsPayload]:
        if not self.is_enabled() or self.redis_cache is None or max_rows <= 0:
            return []

        rows: List[SpendLogsPayload] = []
        try:
            for _ in range(max_rows):
                serialized_payload = await self.redis_cache.async_lpop(
                    REDIS_SPEND_LOG_BUFFER_KEY
                )
                if serialized_payload is None:
                    break
                parsed_payload = safe_json_loads(serialized_payload)
                if isinstance(parsed_payload, dict):
                    rows.append(parsed_payload)
        except Exception as e:
            verbose_proxy_logger.exception(
                "SpendLogRedisBuffer: failed to pop buffered spend log rows: %s", e
            )
        return rows

    async def requeue_spend_log_rows(self, rows: List[SpendLogsPayload]) -> None:
        if not rows:
            return
        if not self.is_enabled() or self.redis_cache is None:
            return

        try:
            for payload in rows:
                await self.redis_cache.async_rpush(
                    REDIS_SPEND_LOG_BUFFER_KEY,
                    [safe_dumps(payload)],
                )
        except Exception as e:
            verbose_proxy_logger.exception(
                "SpendLogRedisBuffer: failed to requeue spend log rows: %s", e
            )

    async def get_buffered_row_count(self) -> int:
        if not self.is_enabled() or self.redis_cache is None:
            return 0
        try:
            return await self.redis_cache.async_llen(REDIS_SPEND_LOG_BUFFER_KEY)
        except Exception:
            return 0
