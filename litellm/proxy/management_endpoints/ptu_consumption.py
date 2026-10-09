"""PTU-hours a team's tokens amount to, attached to a daily activity response.

Azure sizes a provisioned deployment in normalized tokens per minute per PTU, so the prompt,
cached, and completion tokens a team sent to a PTU model group convert back to the share of
a PTU-hour it consumed, reported next to the raw token counts.
"""

from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.ptu_pricing import is_ptu_cost_attribution_enabled
from litellm.llms.azure.ptu_capacity import PTUCapacity, normalized_tokens, ptu_hours
from litellm.router import Router
from litellm.router_utils.common_utils import resolve_model_group_alias
from litellm.router_utils.ptu_shares import (
    model_group_ptu_capacity,
    routed_deployments,
    team_servable_deployments,
)
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
    api_key_breakdown: Final = {
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
    model_groups: Final = {
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
    results: Final = list(days)
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


def capacity_by_requested_name(llm_router: Router, team_id: str | None) -> Callable[[str], PTUCapacity | None]:
    """The sizing row behind the name a usage row is keyed by, resolved the way the ceiling
    resolves a request: an alias to its group, then a group, a routing group, a deployment id,
    or a provider model to the deployments it is served from. For one team those are narrowed
    to the deployments it can be served from, its shared one first, so a team with no share on
    a name reads no PTU-hours for it; a page spanning teams keeps every deployment behind the name."""
    listed_rows: Final = llm_router.get_model_list() or ()
    aliases: Final = llm_router.model_group_alias

    def capacity_for(requested_model: str) -> PTUCapacity | None:
        model_group: Final = resolve_model_group_alias(aliases, requested_model) or requested_model
        routed: Final = routed_deployments(
            listed_rows,
            llm_router.model_list,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # Router.model_list is a bare list
            model_group,
        )
        return model_group_ptu_capacity(routed if team_id is None else team_servable_deployments(routed, team_id))

    return capacity_for


def single_team_id(team_ids: Sequence[str] | None) -> str | None:
    """The one team a usage page is scoped to, else None for a page spanning several or all teams."""
    return team_ids[0] if team_ids is not None and len(team_ids) == 1 else None


def with_ptu_consumption(
    activity: SpendAnalyticsPaginatedResponse, llm_router: Router | None, team_id: str | None
) -> SpendAnalyticsPaginatedResponse:
    """``activity`` with PTU-hours attached from the deployments ``team_id`` is served from (every
    sized deployment when the page spans teams), untouched while PTU cost attribution is off or no
    router is loaded."""
    if llm_router is None or not is_ptu_cost_attribution_enabled():
        return activity
    return attach_ptu_hours(activity, capacity_by_requested_name(llm_router, team_id))
