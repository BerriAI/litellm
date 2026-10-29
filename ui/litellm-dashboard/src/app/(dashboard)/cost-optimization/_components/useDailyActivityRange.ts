import { useMemo, useState } from "react";

import { dailyActivityAggregatedCall } from "@/components/networking";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  toDailyData,
  type DailyActivityMetadata,
  type DailyActivityRequest,
} from "@/components/UsagePage/dailyActivityApi";
import { DailyData } from "@/components/UsagePage/types";
import { spendScopeUserId } from "@/utils/roles";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity";

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

export const useScopedDailyActivityRange = (
  accessToken: string | null,
  scope: ScopedActivityInput,
  { dateValue, onDateChange }: ActivityDateRange,
): DailyActivityRange => {
  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;
  const { userId, apiKey = null } = scope;

  const request = useMemo<DailyActivityRequest | null>(
    () =>
      accessToken && startTime && endTime
        ? {
            accessToken,
            startTime,
            endTime,
            entityIds: userId ? [userId] : null,
            apiKey,
            includeCurrentUtcDay: true,
          }
        : null,
    [accessToken, startTime, endTime, userId, apiKey],
  );

  const { data, loading, failed } = useAggregatedDailyActivity({
    fetch: () => dailyActivityAggregatedCall("user", request as DailyActivityRequest),
    enabled: request !== null,
    deps: [accessToken, startTime, endTime, userId, apiKey],
  });

  return {
    dateValue,
    onDateChange,
    results: toDailyData(data),
    metadata: data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA,
    loading,
    failed,
    scope: { accessToken, startTime, endTime, userId, apiKey },
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
