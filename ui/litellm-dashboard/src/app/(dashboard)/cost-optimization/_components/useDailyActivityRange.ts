import { useMemo, useState } from "react";

import { userDailyActivityCall } from "@/components/networking";
import { DailyData } from "@/components/UsagePage/types";
import { all_admin_roles } from "@/utils/roles";
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
  const effectiveUserId = all_admin_roles.includes(userRole) ? null : userId;

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
  };
};
