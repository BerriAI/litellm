import { useMemo, useState } from "react";

import { userDailyActivityCall } from "@/components/networking";
import { DailyData } from "@/components/UsagePage/types";
import { all_admin_roles, isProxyAdminViewRole } from "@/utils/roles";
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
  // Whether the caller may read deployment-wide savings, i.e. the hourly
  // endpoint, which refuses anything below proxy admin. Distinct from the
  // all_admin_roles check that scopes the daily rollup below.
  canViewGlobalSavings: boolean;
}

export const useDailyActivityRange = (
  accessToken: string | null,
  userId: string | null,
  userRole: string,
): DailyActivityRange => {
  const initialFrom = useMemo(() => new Date(new Date().getTime() - THIRTY_DAYS_MS), []);
  const initialTo = useMemo(() => new Date(), []);
  const [dateValue, setDateValue] = useState<DateRange>({ from: initialFrom, to: initialTo });

  const startTime = dateValue.from ?? null;
  const endTime = dateValue.to ?? null;
  const isAdmin = all_admin_roles.includes(userRole);
  const effectiveUserId = isAdmin ? null : userId;

  const { data, loading, isFetchingMore } = usePaginatedDailyActivity({
    fetchFn: userDailyActivityCall,
    args: [accessToken, startTime, endTime, effectiveUserId],
    enabled: !!accessToken && !!startTime && !!endTime,
  });

  return {
    dateValue,
    onDateChange: setDateValue,
    results: data.results as DailyData[],
    loading,
    isFetchingMore,
    canViewGlobalSavings: isProxyAdminViewRole(userRole),
  };
};
