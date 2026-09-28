from datetime import datetime
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, MODEL_INSIGHTS_TASK_TYPES
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import DailyModelUsageRepository

_TASK_BY_CALL_TYPE: Final = {
    "acompletion": "chat",
    "completion": "chat",
    "aembedding": "embeddings",
    "embedding": "embeddings",
    "aimage_generation": "images",
    "image_generation": "images",
    "aspeech": "audio",
    "speech": "audio",
    "atranscription": "audio",
    "transcription": "audio",
    "arerank": "rerank",
    "rerank": "rerank",
    "aresponses": "responses",
    "responses": "responses",
}
_METADATA: Final = TypeAdapter(dict[str, object])


def model_usage_task_type(call_type: str) -> str:
    task_type: Final = _TASK_BY_CALL_TYPE.get(call_type, "unknown")
    return task_type if task_type in MODEL_INSIGHTS_TASK_TYPES else "unknown"


def _date_from_start_time(start_time: datetime | str) -> str | None:
    if isinstance(start_time, datetime):
        return start_time.date().isoformat()
    return start_time[:10] if len(start_time) >= 10 else None


def _is_internal_call(metadata: str) -> bool:
    try:
        decoded: Final = _METADATA.validate_json(metadata)
    except ValidationError:
        return False
    return bool(decoded.get(INTERNAL_CALL_ORIGIN_METADATA_KEY))


async def increment_daily_model_usage(prisma_client: PrismaClient, payload: SpendLogsPayload) -> None:
    date: Final = _date_from_start_time(payload["startTime"])
    if date is None or _is_internal_call(payload["metadata"]):
        return

    model: Final = payload["model"] or "unknown"
    model_group: Final = payload["model_group"] or model
    provider: Final = payload["custom_llm_provider"] or "unknown"
    task_type: Final = model_usage_task_type(payload["call_type"])
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
