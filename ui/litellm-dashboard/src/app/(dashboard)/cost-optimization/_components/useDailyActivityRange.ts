import { useMemo, useState } from "react";

import { userDailyActivityAggregatedCall, userDailyActivityCall } from "@/components/networking";
import { DailyData } from "@/components/UsagePage/types";
import { spendScopeUserId } from "@/utils/roles";
import { usePaginatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity";

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

export interface DateRange {
  from?: Date;
  to?: Date;
}

export interface DailyActivityRange {
  dateValue: DateRange;
  onDateChange: (value: DateRange) => void;
  results: DailyData[];
  loading: boolean;
  isFetchingMore: boolean;
  progress: { currentPage: number; totalPages: number };
  cancelled: boolean;
  cancel: () => void;
}

/**
 * Which slice of daily activity to read. Both fields are passed straight through to the
 * endpoint as filters, so the caller — not this hook — decides what the viewer may see.
 *
 * `userId: null` asks for the whole proxy, which the backend only honours for admins;
 * a non-admin must send its own id or the request is rejected. That role decision lives in
 * `useDailyActivityRange` below rather than in here, so a caller scoping to one key is not
 * silently re-scoped to a user as well.
 */
export interface DailyActivityScope {
  userId: string | null;
  apiKey?: string | null;
}

export const useScopedDailyActivityRange = (
  accessToken: string | null,
  scope: DailyActivityScope,
): DailyActivityRange => {
  const initialFrom = useMemo(() => new Date(new Date().getTime() - THIRTY_DAYS_MS), []);
  const initialTo = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRange>({ from: initialFrom, to: initialTo });

  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;
  const { userId, apiKey = null } = scope;

  const activityQueryOptions = {
    fetchFn: userDailyActivityCall,
    aggregatedFetchFn: userDailyActivityAggregatedCall,
    // Positional, and read by two functions whose signatures diverge at index 3: the paginated
    // call takes `page` there (injected by the hook) and the aggregated one does not. Anything
    // appended here must therefore be appended to BOTH networking signatures, in this order.
    args: [accessToken, startTime, endTime, userId, true, apiKey],
    enabled: !!accessToken && !!startTime && !!endTime,
  };
  const { data, loading, isFetchingMore, progress, cancelled, cancel } =
    usePaginatedDailyActivity(activityQueryOptions);

  return {
    dateValue,
    onDateChange: setDateValue,
    results: data.results as DailyData[],
    loading,
    isFetchingMore,
    progress,
    cancelled,
    cancel,
  };
};

export const useDailyActivityRange = (
  accessToken: string | null,
  userId: string | null,
  userRole: string,
): DailyActivityRange =>
  useScopedDailyActivityRange(accessToken, {
    userId: spendScopeUserId(userRole, userId),
  });
