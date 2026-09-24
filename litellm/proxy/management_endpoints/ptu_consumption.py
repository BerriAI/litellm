"""PTU-hours a team's tokens amount to, attached to a daily activity response.

Azure sizes a provisioned deployment in normalized tokens per minute per PTU, so the prompt,
cached, and completion tokens a team sent to a PTU model group convert back to the share of
a PTU-hour it consumed, reported next to the raw token counts.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import Final

from litellm.llms.azure.ptu_capacity import PTUCapacity, normalized_tokens, ptu_hours
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendData,
    MetricWithMetadata,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
)


def _with_ptu_hours(metrics: SpendMetrics, capacity: PTUCapacity) -> SpendMetrics:
    consumed: Final = ptu_hours(
        capacity,
        normalized_tokens(
            capacity,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            cache_read_tokens=metrics.cache_read_input_tokens,
        ),
    )
    return metrics.model_copy(update=MappingProxyType({"ptu_hours": consumed}))


def _model_group_with_ptu_hours(bucket: MetricWithMetadata, capacity: PTUCapacity) -> MetricWithMetadata:
    api_key_breakdown: Final = {  # mutable-ok: pydantic serializes a dict[...] field only from a plain dict
        api_key: key_bucket.model_copy(
            update=MappingProxyType({"metrics": _with_ptu_hours(key_bucket.metrics, capacity)})
        )
        for api_key, key_bucket in bucket.api_key_breakdown.items()
    }
    return bucket.model_copy(
        update=MappingProxyType(
            {"metrics": _with_ptu_hours(bucket.metrics, capacity), "api_key_breakdown": api_key_breakdown}
        )
    )


def _day_with_ptu_hours(
    day: DailySpendData, capacity_for_model_group: Callable[[str], PTUCapacity | None]
) -> DailySpendData:
    priced: Final = MappingProxyType(
        {
            model_group: _model_group_with_ptu_hours(bucket, capacity)
            for model_group, bucket in day.breakdown.model_groups.items()
            if (capacity := capacity_for_model_group(model_group)) is not None
        }
    )
    if not priced:
        return day
    model_groups: Final = {  # mutable-ok: pydantic serializes a dict[...] field only from a plain dict
        **day.breakdown.model_groups,
        **priced,
    }
    return day.model_copy(
        update=MappingProxyType(
            {
                "metrics": day.metrics.model_copy(
                    update=MappingProxyType({"ptu_hours": sum(bucket.metrics.ptu_hours for bucket in priced.values())})
                ),
                "breakdown": day.breakdown.model_copy(update=MappingProxyType({"model_groups": model_groups})),
            }
        )
    )


def attach_ptu_hours(
    response: SpendAnalyticsPaginatedResponse, capacity_for_model_group: Callable[[str], PTUCapacity | None]
) -> SpendAnalyticsPaginatedResponse:
    """The response with ``ptu_hours`` filled in on every PTU model group and its api keys,
    on each day, and on the total, from the tokens already on the page.

    A model group the resolver has no sizing row for keeps ``ptu_hours`` at zero.
    """
    days: Final = tuple(_day_with_ptu_hours(day, capacity_for_model_group) for day in response.results)
    results: Final = list(days)  # mutable-ok: pydantic serializes a list[...] field only from a plain list
    return response.model_copy(
        update=MappingProxyType(
            {
                "results": results,
                "metadata": response.metadata.model_copy(
                    update=MappingProxyType({"total_ptu_hours": sum(day.metrics.ptu_hours for day in days)})
                ),
            }
        )
    )
