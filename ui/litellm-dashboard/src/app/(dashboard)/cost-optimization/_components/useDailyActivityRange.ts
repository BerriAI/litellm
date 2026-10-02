import { isQueryPending } from "@/app/(dashboard)/hooks/common/queryReadiness";
import { useMemo, useState } from "react";

import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  toDailyData,
  type DailyActivityMetadata,
  type DailyActivityRequest,
} from "@/components/UsagePage/dailyActivityApi";
import { DailyData } from "@/components/UsagePage/types";
import { spendScopeUserId } from "@/utils/roles";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/hooks/dailyActivity/dailyActivityQueries";

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

export interface DateRange {
  from?: Date;
  to?: Date;
}

export interface DailyActivityScope {
  accessToken: string | null;
  startTime: Date | null;
  endTime: Date | null;
  userId: string | null;
  apiKey: string | null;
}

export interface DailyActivityRange {
  dateValue: DateRange;
  onDateChange: (value: DateRange) => void;
  results: DailyData[];
  metadata: DailyActivityMetadata;
  loading: boolean;
  failed: boolean;
  scope: DailyActivityScope;
}

export type ActivityDateRange = Pick<DailyActivityRange, "dateValue" | "onDateChange">;

export const useActivityDateRange = (): ActivityDateRange => {
  const initialFrom = useMemo(() => new Date(new Date().getTime() - THIRTY_DAYS_MS), []);
  const initialTo = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRange>({ from: initialFrom, to: initialTo });
  return { dateValue, onDateChange: setDateValue };
};

export interface ScopedActivityInput {
  userId: string | null;
  apiKey?: string | null;
}

export const buildDailyActivityRequest = (scope: DailyActivityScope): DailyActivityRequest | null =>
  scope.accessToken && scope.startTime && scope.endTime
    ? {
        accessToken: scope.accessToken,
        startTime: scope.startTime,
        endTime: scope.endTime,
        entityIds: scope.userId ? [scope.userId] : null,
        apiKey: scope.apiKey,
        includeCurrentUtcDay: true,
      }
    : null;

export const useScopedDailyActivityRange = (
  accessToken: string | null,
  scope: ScopedActivityInput,
  { dateValue, onDateChange }: ActivityDateRange,
): DailyActivityRange => {
  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;
  const { userId, apiKey = null } = scope;

  const activityScope = useMemo<DailyActivityScope>(
    () => ({ accessToken, startTime, endTime, userId, apiKey }),
    [accessToken, startTime, endTime, userId, apiKey],
  );

  const request = useMemo<DailyActivityRequest | null>(() => buildDailyActivityRequest(activityScope), [activityScope]);

  const query = useAggregatedDailyActivity("user", request);
  const data = query.data ?? EMPTY_DAILY_ACTIVITY_RESPONSE;

  return {
    dateValue,
    onDateChange,
    results: toDailyData(data),
    metadata: data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA,
    loading: isQueryPending(query),
    failed: query.isError,
    scope: activityScope,
  };
};

export const useDailyActivityRange = (
  accessToken: string | null,
  userId: string | null,
  userRole: string,
): DailyActivityRange => {
  const dateRange = useActivityDateRange();
  return useScopedDailyActivityRange(accessToken, { userId: spendScopeUserId(userRole, userId) }, dateRange);
};
