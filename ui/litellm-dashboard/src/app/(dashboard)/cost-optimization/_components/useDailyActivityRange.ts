import moment from "moment";
import { useMemo, useState } from "react";
import { useDailyUsageTimezone } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { usageCalendarDate } from "@/utils/usageTimezone";

import { userDailyActivityAggregatedCall, userDailyActivityCall } from "@/components/networking";
import { ApiKeyTruncation, getApiKeyTruncation } from "@/components/EntityUsageExport/exportBlockedReason";
import { DailyData } from "@/components/UsagePage/types";
import { spendScopeUserId } from "@/utils/roles";
import { usePaginatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity";

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
  failed: boolean;
  cancel: () => void;
  apiKeyTruncation?: ApiKeyTruncation;
  reportingTimezone?: string;
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

export type ActivityDateRange = Pick<DailyActivityRange, "dateValue" | "onDateChange" | "reportingTimezone">;

export const useActivityDateRange = (): ActivityDateRange => {
  const reportingTimezone = useDailyUsageTimezone();
  const initialTo = useMemo(() => usageCalendarDate(new Date(), reportingTimezone), [reportingTimezone]);
  const initialFrom = useMemo(() => moment(initialTo).subtract(30, "days").toDate(), [initialTo]);
  const [selectedRange, setDateValue] = useState<DateRange | null>(null);
  const dateValue = useMemo(
    () => selectedRange ?? { from: initialFrom, to: initialTo },
    [selectedRange, initialFrom, initialTo],
  );
  return { dateValue, onDateChange: setDateValue, reportingTimezone };
};

export const useScopedDailyActivityRange = (
  accessToken: string | null,
  scope: DailyActivityScope,
  { dateValue, onDateChange, reportingTimezone }: ActivityDateRange,
): DailyActivityRange => {
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
  const { data, loading, isFetchingMore, progress, cancelled, failed, coversRange, cancel } =
    usePaginatedDailyActivity(activityQueryOptions);
  const readUnavailable = failed || cancelled;
  const waitingForRange = activityQueryOptions.enabled && !coversRange && !readUnavailable;

  return {
    dateValue,
    onDateChange,
    reportingTimezone,
    results: data.results as DailyData[],
    loading: loading || waitingForRange,
    isFetchingMore,
    progress,
    cancelled,
    failed,
    cancel,
    apiKeyTruncation: getApiKeyTruncation(data.metadata?.api_key_limit, data.metadata?.total_api_keys),
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
