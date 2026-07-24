import { useEffect, useState } from "react";

import { getHourlySavings, HourlySavingsResponse } from "@/components/networking";
import { localIsoDay, shouldUseHourlySavings } from "./costOptimizationUtils";

/**
 * Hour-resolution savings for date ranges the daily rollup renders as a point
 * or two. Returns null whenever the daily rollup should be charted instead:
 * long ranges, non-admins (the endpoint reads across the deployment), a failed
 * request, or a proxy running with spend logs off, which is the only source of
 * sub-day detail and would otherwise chart a flat zero line.
 */
export const useHourlySavings = (
  accessToken: string | null,
  from: Date | undefined,
  to: Date | undefined,
  canViewGlobalSavings: boolean,
): HourlySavingsResponse | null => {
  const eligible = !!accessToken && canViewGlobalSavings && shouldUseHourlySavings(from, to);
  const rangeKey = eligible && from && to ? `${localIsoDay(from)}|${localIsoDay(to)}` : "";
  const [state, setState] = useState<{ key: string; data: HourlySavingsResponse | null } | null>(null);

  useEffect(() => {
    if (!eligible) return;
    if (!accessToken || !from || !to) return;
    let cancelled = false;
    getHourlySavings(accessToken, localIsoDay(from), localIsoDay(to), Intl.DateTimeFormat().resolvedOptions().timeZone)
      .then((data) => {
        if (!cancelled) setState({ key: rangeKey, data });
      })
      .catch(() => {
        if (!cancelled) setState({ key: rangeKey, data: null });
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, eligible, rangeKey, from, to]);

  if (!eligible || state?.key !== rangeKey) return null;
  return state.data?.spend_logs_disabled ? null : state.data;
};
