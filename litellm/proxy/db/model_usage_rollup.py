from datetime import datetime
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MODEL_INSIGHTS_DEFAULT_TASK,
    MODEL_INSIGHTS_TASK_TAG_PREFIX,
)
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import DailyModelUsageRepository

_METADATA: Final = TypeAdapter(dict[str, object])
_TAGS: Final = TypeAdapter(list[object])


def model_usage_task_type(request_tags: str) -> str:
    try:
        tags: Final = _TAGS.validate_json(request_tags)
    except ValidationError:
        return MODEL_INSIGHTS_DEFAULT_TASK
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(MODEL_INSIGHTS_TASK_TAG_PREFIX):
            task = tag.removeprefix(MODEL_INSIGHTS_TASK_TAG_PREFIX)
            if task in load_model_insight_tasks():
                return task
    return MODEL_INSIGHTS_DEFAULT_TASK


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


async def increment_daily_model_usage(prisma_client: PrismaClient, payload: SpendLogsPayload) -> None:
    date: Final = _date_from_start_time(payload["startTime"])
    if date is None or _is_internal_call(payload["metadata"]):
        return

    model: Final = payload["model"] or "unknown"
    model_group: Final = payload["model_group"] or model
    provider: Final = payload["custom_llm_provider"] or "unknown"
    task_type: Final = model_usage_task_type(payload["request_tags"])
    successful: Final = 1 if payload["status"] == "success" else 0
    failed: Final = 1 - successful
    key: Final = {
        "date": date,
        "model_group": model_group,
        "model": model,
        "custom_llm_provider": provider,
        "task_type": task_type,
    }
    await DailyModelUsageRepository(prisma_client).table.upsert(
        where={"date_model_group_model_custom_llm_provider_task_type": key},
        data={
            "create": {
                **key,
                "spend": payload["spend"],
                "prompt_tokens": payload["prompt_tokens"],
                "completion_tokens": payload["completion_tokens"],
                "request_count": 1,
                "successful_requests": successful,
                "failed_requests": failed,
            },
            "update": {
                "spend": {"increment": payload["spend"]},
                "prompt_tokens": {"increment": payload["prompt_tokens"]},
                "completion_tokens": {"increment": payload["completion_tokens"]},
                "request_count": {"increment": 1},
                "successful_requests": {"increment": successful},
                "failed_requests": {"increment": failed},
            },
        },
    )
