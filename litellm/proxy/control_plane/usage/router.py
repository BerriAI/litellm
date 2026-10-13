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
    get_dataplane_client,
    get_fanout,
    selected_dataplanes,
)
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.proxy.control_plane.usage.service import (
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
from litellm.types.proxy.management_endpoints.common_daily_activity import DailySpendData

DEFAULT_WINDOW_DAYS: Final = 30

router: Final = APIRouter(prefix="/usage", tags=["usage"])

UsageRead: TypeAlias = Annotated[Principal, Security(require_scopes, scopes=[Scope.USAGE_READ.value])]
Selected: TypeAlias = Annotated[tuple[Dataplane, ...], Depends(selected_dataplanes)]
GroupBy: TypeAlias = Annotated[UsageGroupBy, Query(description="Dimension to split the fleet's usage by.")]


def usage_window(
    start_date: Annotated[date | None, Query(description="First day, inclusive. Defaults to 30 days ago.")] = None,
    end_date: Annotated[date | None, Query(description="Last day, inclusive. Defaults to today (UTC).")] = None,
) -> UsageWindow:
    end: Final = end_date if end_date is not None else datetime.now(UTC).date()
    start: Final = start_date if start_date is not None else end - timedelta(days=DEFAULT_WINDOW_DAYS - 1)
    if start > end:
        raise ControlPlaneException(bad_request(f"start_date {start} is after end_date {end}."))
    return UsageWindow(start=start, end=end)


Window: TypeAlias = Annotated[UsageWindow, Depends(usage_window)]


async def _gather_days(
    fanout: Fanout, client: DataplaneClient, caller: Caller, dataplanes: tuple[Dataplane, ...], window: UsageWindow
) -> Gathered[tuple[DailySpendData, ...]]:
    gathered: Final = await fanout.gather(
        dataplanes, lambda dataplane: fetch_daily_activity(client, dataplane, caller, window)
    )
    if gathered.nothing_answered:
        raise ControlPlaneException(service_unavailable(f"No dataplane answered: {describe_missed(gathered.missed)}"))
    return gathered


def _fleet(gathered: Gathered[tuple[DailySpendData, ...]]) -> tuple[DataplaneDays, ...]:
    return tuple(DataplaneDays(reached.dataplane.id, reached.value) for reached in gathered.reached)


def _unreachable(gathered: Gathered[tuple[DailySpendData, ...]]) -> tuple[UnreachableDataplane, ...]:
    return tuple(UnreachableDataplane(id=miss.dataplane.id, reason=miss.reason) for miss in gathered.missed)


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
    return UsageSummaryResponse(
        window=_window(window),
        totals=totals(fleet),
        by_dataplane=tuple(DataplaneUsage(id=days.dataplane_id, metrics=dataplane_totals(days)) for days in fleet),
        partial=gathered.partial,
        unreachable=_unreachable(gathered),
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
    return DailyUsageResponse(
        window=_window(window),
        group_by=group_by,
        rows=tuple(
            DailyUsageRow(date=day, group=group, metrics=metrics)
            for (day, group), metrics in daily_rows(_fleet(gathered), group_by)
        ),
        partial=gathered.partial,
        unreachable=_unreachable(gathered),
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
    return TopUsageResponse(
        window=_window(window),
        group_by=group_by,
        limit=limit,
        items=tuple(
            UsageRank(group=group, metrics=metrics) for group, metrics in ranked(_fleet(gathered), group_by, limit)
        ),
        partial=gathered.partial,
        unreachable=_unreachable(gathered),
    )
