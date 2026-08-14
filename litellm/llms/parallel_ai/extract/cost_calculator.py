from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

PARALLEL_AI_EXTRACT_COST_PER_URL: Final = 0.001
PARALLEL_AI_EXTRACT_MODEL: Final = "parallel_ai/extract"
PARALLEL_AI_EXTRACT_USAGE_SKU: Final = "sku_extract_excerpts"


class _ParallelAIExtractUsageItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: StrictStr
    count: Annotated[StrictInt, Field(ge=0)]


class _ParallelAIExtractUsageName(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: StrictStr


class _ParallelAIExtractBillingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usage: tuple[object, ...] | None = None


class _ParallelAIExtractBillingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    urls: tuple[StrictStr, ...] = ()


def _usage_url_count(response_body: object) -> int | None:
    try:
        parsed: Final = _ParallelAIExtractBillingResponse.model_validate(response_body)
    except ValidationError:
        return None
    if parsed.usage is None:
        return None

    target_items: Final = tuple(item for item in parsed.usage if _usage_name(item) == PARALLEL_AI_EXTRACT_USAGE_SKU)
    if not target_items:
        return 0

    try:
        usage_items: Final = tuple(_ParallelAIExtractUsageItem.model_validate(item) for item in target_items)
    except ValidationError:
        return None
    return sum(item.count for item in usage_items)


def _usage_name(usage_item: object) -> str | None:
    try:
        parsed: Final = _ParallelAIExtractUsageName.model_validate(usage_item)
    except ValidationError:
        return None
    return parsed.name


def _request_url_count(request_body: object) -> int:
    try:
        parsed: Final = _ParallelAIExtractBillingRequest.model_validate(request_body)
    except ValidationError:
        return 0
    return len(parsed.urls)


def parallel_ai_extract_cost(request_body: object, response_body: object) -> float:
    usage_url_count: Final = _usage_url_count(response_body)
    billed_url_count: Final = usage_url_count if usage_url_count is not None else _request_url_count(request_body)
    return billed_url_count * PARALLEL_AI_EXTRACT_COST_PER_URL
