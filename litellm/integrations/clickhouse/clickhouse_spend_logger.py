import re
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_logger
from litellm.integrations.clickhouse.clickhouse_batch_logger import ClickHouseBatchLogger
from litellm.integrations.clickhouse.schema import SPEND_LOGS_TABLE
from litellm.rust_bridge.traces import TraceStorage

_CACHE_HIT_SUFFIX: Final = re.compile(r"_cache_hit[0-9.]+$")


class _SpendMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_api_key_hash: str | None = None
    user_api_key_team_id: str | None = None


class _SpendPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    call_type: str = ""
    response_cost: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    startTime: float
    endTime: float
    metadata: _SpendMetadata = _SpendMetadata()
    model: str | None = None
    status: str = ""
    cache_hit: bool | None = None


def spend_log_row_from_payload(payload: _SpendPayload) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "request_id": payload.id,
            "response_id": _CACHE_HIT_SUFFIX.sub("", payload.id),
            "call_type": payload.call_type,
            "api_key": payload.metadata.user_api_key_hash or "",
            "team_id": payload.metadata.user_api_key_team_id or "",
            "model": payload.model or "",
            "spend": payload.response_cost or 0.0,
            "prompt_tokens": payload.prompt_tokens,
            "completion_tokens": payload.completion_tokens,
            "total_tokens": payload.total_tokens,
            "start_time": int(payload.startTime * 1000),
            "end_time": int(payload.endTime * 1000),
            "status": payload.status,
            "cache_hit": payload.cache_hit is True,
        }
    )


class ClickHouseSpendLogger(ClickHouseBatchLogger):
    table = SPEND_LOGS_TABLE

    def __init__(self, storage: TraceStorage) -> None:
        super().__init__(storage=storage)

    async def _log(self, kwargs: Mapping[str, object]) -> None:
        try:
            payload: Final = _SpendPayload.model_validate(kwargs.get("standard_logging_object"))
            if payload.call_type.startswith("/v1/traces"):
                return
            self.enqueue((spend_log_row_from_payload(payload),))
        except (ValidationError, RuntimeError, ValueError) as error:
            verbose_logger.warning("ClickHouse spend logging failed: %s", error)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._log(kwargs)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._log(kwargs)
