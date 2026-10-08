from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter, ValidationError
from typing_extensions import LiteralString

from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MODEL_INSIGHTS_DEFAULT_TASK,
    MODEL_INSIGHTS_TASK_TAG_PREFIX,
)
from litellm.proxy._types import DB_RETRY_SAFE_ERROR_TYPES, SpendLogsPayload
from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks
from litellm.proxy.db.rollup_lock_timeout import ROLLUP_LOCK_TIMEOUT_SQL, rollup_lock_timeout_setting

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_METADATA: Final = TypeAdapter(dict[str, object])
_TAGS: Final = TypeAdapter(list[object])


class _UpsertTable(Protocol):
    def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class _ModelUsageBatch(Protocol):
    litellm_dailymodelusage: _UpsertTable

    def execute_raw(self, query: LiteralString, *args: object) -> None: ...


class _ModelUsageBatchManager(Protocol):
    async def __aenter__(self) -> _ModelUsageBatch: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


@dataclass(frozen=True, slots=True)
class ModelUsageKey:
    date: str
    model_group: str
    model: str
    custom_llm_provider: str
    task_type: str


@dataclass(frozen=True, slots=True)
class ModelUsageTransaction:
    key: ModelUsageKey
    spend: float
    prompt_tokens: int
    completion_tokens: int
    successful: bool


def model_usage_task_type(request_tags: str) -> str:
    try:
        tags: Final = _TAGS.validate_json(request_tags)
    except ValidationError:
        return MODEL_INSIGHTS_DEFAULT_TASK
    return next(
        (
            task
            for tag in tags
            if isinstance(tag, str)
            and tag.startswith(MODEL_INSIGHTS_TASK_TAG_PREFIX)
            and (task := tag.removeprefix(MODEL_INSIGHTS_TASK_TAG_PREFIX)) in load_model_insight_tasks()
        ),
        MODEL_INSIGHTS_DEFAULT_TASK,
    )


def _is_internal_call(metadata: str) -> bool:
    try:
        decoded: Final = _METADATA.validate_json(metadata)
    except ValidationError:
        return False
    return bool(decoded.get(INTERNAL_CALL_ORIGIN_METADATA_KEY))


def _date_from_start_time(start_time: datetime | str) -> str | None:
    if isinstance(start_time, datetime):
        return start_time.date().isoformat()
    return start_time[:10] if len(start_time) >= 10 else None


def build_model_usage_transaction(payload: SpendLogsPayload) -> ModelUsageTransaction | None:
    date: Final = _date_from_start_time(payload["startTime"])
    if date is None or _is_internal_call(payload["metadata"]):
        return None
    model: Final = payload["model"] or "unknown"
    return ModelUsageTransaction(
        key=ModelUsageKey(
            date=date,
            model_group=payload["model_group"] or model,
            model=model,
            custom_llm_provider=payload["custom_llm_provider"] or "unknown",
            task_type=model_usage_task_type(payload["request_tags"]),
        ),
        spend=payload["spend"],
        prompt_tokens=payload["prompt_tokens"],
        completion_tokens=payload["completion_tokens"],
        successful=payload["status"] == "success",
    )


def _model_usage_batch(prisma_client: PrismaClient) -> _ModelUsageBatchManager:
    batch: Final[_ModelUsageBatchManager] = prisma_client.db.batch_()
    return batch


def _sort_key(transaction: ModelUsageTransaction) -> tuple[str, str, str, str, str]:
    key: Final = transaction.key
    return (key.date, key.model_group, key.model, key.custom_llm_provider, key.task_type)


async def flush_model_usage_transactions(
    prisma_client: PrismaClient,
    transactions: Sequence[ModelUsageTransaction],
    n_retry_times: int = 3,
) -> None:
    """One upsert per rollup row for the whole drained batch, in a single transaction and in key order so
    concurrent pods take row locks in the same order. Only ConnectError is retried: it proves nothing reached
    the database, while a retry after an ambiguous post-send failure could double-count the increments."""
    if not transactions:
        return
    ordered: Final = sorted(transactions, key=_sort_key)
    for attempt in range(n_retry_times + 1):
        try:
            async with _model_usage_batch(prisma_client) as batcher:
                batcher.execute_raw(ROLLUP_LOCK_TIMEOUT_SQL, rollup_lock_timeout_setting())
                for key, grouped in groupby(ordered, key=lambda transaction: transaction.key):
                    entries = tuple(grouped)
                    spend = sum(entry.spend for entry in entries)
                    prompt_tokens = sum(entry.prompt_tokens for entry in entries)
                    completion_tokens = sum(entry.completion_tokens for entry in entries)
                    successful = sum(1 for entry in entries if entry.successful)
                    failed = len(entries) - successful
                    key_fields = {
                        "date": key.date,
                        "model_group": key.model_group,
                        "model": key.model,
                        "custom_llm_provider": key.custom_llm_provider,
                        "task_type": key.task_type,
                    }
                    batcher.litellm_dailymodelusage.upsert(
                        where={"date_model_group_model_custom_llm_provider_task_type": key_fields},
                        data={
                            "create": {
                                **key_fields,
                                "spend": spend,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "request_count": len(entries),
                                "successful_requests": successful,
                                "failed_requests": failed,
                            },
                            "update": {
                                "spend": {"increment": spend},
                                "prompt_tokens": {"increment": prompt_tokens},
                                "completion_tokens": {"increment": completion_tokens},
                                "request_count": {"increment": len(entries)},
                                "successful_requests": {"increment": successful},
                                "failed_requests": {"increment": failed},
                            },
                        },
                    )
            return
        except DB_RETRY_SAFE_ERROR_TYPES:
            if attempt >= n_retry_times:
                raise
            await asyncio.sleep(2.0**attempt + random.uniform(0, 1))
