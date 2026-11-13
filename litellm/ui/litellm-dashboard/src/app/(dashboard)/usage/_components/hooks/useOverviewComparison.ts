import { useMemo } from "react";
import { formatDate } from "@/components/networking";
import type { DailyActivityAggregatedResponse, DailyActivityRequest } from "@/components/UsagePage/dailyActivityApi";
import { ENTITY_API } from "../components/EntityUsage/entityFetchFns";
import { statComparison, type ComparisonStatus, type StatComparison } from "../components/overview/usageComparison";
import { previousPeriod, type DayRange } from "../components/overview/usageTrendData";
import type { UsageOption } from "../components/UsageViewSelect/UsageViewSelect";
import { useAggregatedDailyActivity } from "./useAggregatedDailyActivity";

const OVERVIEW_VIEWS: readonly UsageOption[] = ["global", "my-usage"];

interface OverviewComparisonInput {
  request: DailyActivityRequest | null;
  current: DailyActivityAggregatedResponse;
  view: UsageOption;
  tab: string;
  requestsComparable: boolean;
}

interface OverviewComparison {
  range: DayRange | null;
  comparison: StatComparison | undefined;
}

const localDayStart = (date: string): Date => {
  const [year, month, day] = date.split("-").map(Number);
  return new Date(year, month - 1, day);
};

const localDayEnd = (date: string): Date => {
  const [year, month, day] = date.split("-").map(Number);
  return new Date(year, month - 1, day, 23, 59, 59, 999);
};

const statusOf = (loading: boolean, failed: boolean): ComparisonStatus => {
  if (loading) return "loading";
  return failed ? "failed" : "ready";
};

const previousRequest = (request: DailyActivityRequest, range: DayRange): DailyActivityRequest => ({
  ...request,
  startTime: localDayStart(range.start),
  endTime: localDayEnd(range.end),
  apiKeyLimit: 1,
});

export function useOverviewComparison({
  request,
  current,
  view,
  tab,
  requestsComparable,
}: OverviewComparisonInput): OverviewComparison {
  const range = useMemo<DayRange | null>(
    () => (request ? { start: formatDate(request.startTime), end: formatDate(request.endTime) } : null),
    [request],
  );
  const previous = useMemo(() => (range ? previousPeriod(range) : null), [range]);
  const shown = OVERVIEW_VIEWS.includes(view) && tab === "cost";
  const comparisonRequest = useMemo(
    () => (shown && request && previous ? previousRequest(request, previous) : null),
    [shown, request, previous],
  );
  const fetched = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.user.aggregated(comparisonRequest as DailyActivityRequest),
    enabled: comparisonRequest !== null,
    deps: [comparisonRequest?.accessToken, previous?.start, previous?.end, comparisonRequest?.entityIds?.join(",")],
  });
  const comparison = useMemo(() => {
    if (comparisonRequest === null || previous === null) return undefined;
    const input = {
      current: current.metadata ?? null,
      previous: fetched.data.metadata ?? null,
      previousRange: previous,
      status: statusOf(fetched.loading, fetched.failed),
      requestsComparable,
    };
    return statComparison(input);
  }, [comparisonRequest, previous, current, fetched.data, fetched.loading, fetched.failed, requestsComparable]);
  return { range, comparison };
}
