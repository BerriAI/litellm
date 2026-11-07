from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, Query, Security

from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, bad_request, service_unavailable
from litellm.proxy.control_plane.common.fanout import Fanout, Gathered, describe_missed
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient
from litellm.proxy.control_plane.dataplanes.credentials import Caller
from litellm.proxy.control_plane.dataplanes.dependencies import (
    get_caller,
    get_control_plane_settings,
    get_dataplane_client,
    get_fanout,
    selected_dataplanes,
)
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.proxy.control_plane.usage.service import (
    DataplaneActivity,
    DataplaneDays,
    UsageWindow,
    daily_rows,
    dataplane_totals,
    fetch_daily_activity,
    ranked,
    totals,
)
from litellm.types.proxy.control_plane.dataplanes import UnreachableDataplane
from litellm.types.proxy.control_plane.usage import (
    DailyUsageResponse,
    DailyUsageRow,
    DataplaneUsage,
    TopUsageResponse,
    UsageGroupBy,
    UsageRank,
    UsageSummaryResponse,
    UsageWindowResponse,
)

DEFAULT_WINDOW_DAYS: Final = 30

router: Final = APIRouter(prefix="/usage", tags=["usage"])

UsageRead: TypeAlias = Annotated[Principal, Security(require_scopes, scopes=[Scope.USAGE_READ.value])]
Selected: TypeAlias = Annotated[tuple[Dataplane, ...], Depends(selected_dataplanes)]
GroupBy: TypeAlias = Annotated[UsageGroupBy, Query(description="Dimension to split the fleet's usage by.")]


def usage_window(
    settings: Annotated[ControlPlaneSettings, Depends(get_control_plane_settings)],
    start_date: Annotated[date | None, Query(description="First day, inclusive. Defaults to 30 days ago.")] = None,
    end_date: Annotated[date | None, Query(description="Last day, inclusive. Defaults to today (UTC).")] = None,
) -> UsageWindow:
    end: Final = end_date if end_date is not None else datetime.now(UTC).date()
    default_window_days: Final = min(DEFAULT_WINDOW_DAYS, settings.usage_max_window_days)
    start: Final = start_date if start_date is not None else end - timedelta(days=default_window_days - 1)
    if start > end:
        raise ControlPlaneException(bad_request(f"start_date {start} is after end_date {end}."))
    if (end - start).days + 1 > settings.usage_max_window_days:
        raise ControlPlaneException(bad_request(f"Usage windows may not exceed {settings.usage_max_window_days} days."))
    return UsageWindow(start=start, end=end)


Window: TypeAlias = Annotated[UsageWindow, Depends(usage_window)]


async def _gather_days(
    fanout: Fanout, client: DataplaneClient, caller: Caller, dataplanes: tuple[Dataplane, ...], window: UsageWindow
) -> Gathered[DataplaneActivity]:
    gathered: Final = await fanout.gather(
        dataplanes, lambda dataplane: fetch_daily_activity(client, dataplane, caller, window)
    )
    if gathered.nothing_answered:
        raise ControlPlaneException(service_unavailable(f"No dataplane answered: {describe_missed(gathered.missed)}"))
    return gathered


def _fleet(gathered: Gathered[DataplaneActivity]) -> tuple[DataplaneDays, ...]:
    return tuple(DataplaneDays(reached.dataplane.id, reached.value.days) for reached in gathered.reached)


def _unreachable(gathered: Gathered[DataplaneActivity]) -> tuple[UnreachableDataplane, ...]:
    return tuple(UnreachableDataplane(id=miss.dataplane.id, reason=miss.reason) for miss in gathered.missed)


def _truncated_dataplanes(gathered: Gathered[DataplaneActivity]) -> tuple[str, ...]:
    return tuple(reached.dataplane.id for reached in gathered.reached if reached.value.keys_truncated)


def _window(window: UsageWindow) -> UsageWindowResponse:
    return UsageWindowResponse(start_date=window.start, end_date=window.end)


@router.get("/summary", response_model=UsageSummaryResponse)
async def usage_summary(
    _principal: UsageRead,
    dataplanes: Selected,
    window: Window,
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    fanout: Annotated[Fanout, Depends(get_fanout)],
) -> UsageSummaryResponse:
    """Spend and token totals across every dataplane the caller is bound to."""
    gathered: Final = await _gather_days(fanout, client, caller, dataplanes, window)
    fleet: Final = _fleet(gathered)
    truncated: Final = _truncated_dataplanes(gathered)
    return UsageSummaryResponse(
        window=_window(window),
        totals=totals(fleet),
        by_dataplane=tuple(DataplaneUsage(id=days.dataplane_id, metrics=dataplane_totals(days)) for days in fleet),
        partial=gathered.partial or bool(truncated),
        unreachable=_unreachable(gathered),
        truncated_dataplanes=truncated,
    )


@router.get("/daily", response_model=DailyUsageResponse)
async def usage_daily(
    _principal: UsageRead,
    dataplanes: Selected,
    window: Window,
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    fanout: Annotated[Fanout, Depends(get_fanout)],
    group_by: GroupBy = "dataplane",
) -> DailyUsageResponse:
    """Per-day usage across the fleet, split by dataplane, model, provider, key, team or endpoint."""
    gathered: Final = await _gather_days(fanout, client, caller, dataplanes, window)
    truncated: Final = _truncated_dataplanes(gathered)
    return DailyUsageResponse(
        window=_window(window),
        group_by=group_by,
        rows=tuple(
            DailyUsageRow(date=day, group=group, metrics=metrics)
            for (day, group), metrics in daily_rows(_fleet(gathered), group_by)
        ),
        partial=gathered.partial or bool(truncated),
        unreachable=_unreachable(gathered),
        truncated_dataplanes=truncated,
    )


@router.get("/top", response_model=TopUsageResponse)
async def usage_top(
    _principal: UsageRead,
    dataplanes: Selected,
    window: Window,
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    fanout: Annotated[Fanout, Depends(get_fanout)],
    group_by: GroupBy = "model",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> TopUsageResponse:
    """The biggest spenders in the window along one dimension, summed across the fleet."""
    gathered: Final = await _gather_days(fanout, client, caller, dataplanes, window)
    truncated: Final = _truncated_dataplanes(gathered)
    return TopUsageResponse(
        window=_window(window),
        group_by=group_by,
        limit=limit,
        items=tuple(
            UsageRank(group=group, metrics=metrics) for group, metrics in ranked(_fleet(gathered), group_by, limit)
        ),
        partial=gathered.partial or bool(truncated),
        unreachable=_unreachable(gathered),
        truncated_dataplanes=truncated,
    )
