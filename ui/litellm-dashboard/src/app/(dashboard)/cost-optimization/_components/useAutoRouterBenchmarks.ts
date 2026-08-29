import { formatDate } from "@/components/networking";
import { $api } from "@/lib/http/api";

import type { DateRange } from "./useDailyActivityRange";

/**
 * The endpoint cuts on UTC days but the picker hands back local dates, so west of UTC a
 * range ending today would hide sessions started after UTC midnight. Extend it to the
 * current UTC day (only the empty future is added), mirroring include_current_utc_day.
 */
export const benchmarksWindow = (
  range: DateRange,
  now: Date,
  toLocalDay: (d: Date) => string = formatDate,
): { start_date: string; end_date: string } | Record<string, never> => {
  if (!range.from || !range.to) return {};
  const end_date = toLocalDay(range.to);
  const utcToday = now.toISOString().slice(0, 10);
  const endsToday = end_date >= toLocalDay(now);
  return {
    start_date: toLocalDay(range.from),
    end_date: endsToday && utcToday > end_date ? utcToday : end_date,
  };
};

export const useAutoRouterBenchmarks = (accessToken: string | null, range: DateRange) =>
  $api.useQuery(
    "get",
    "/auto_router/benchmarks",
    { params: { query: benchmarksWindow(range, new Date()) } },
    { enabled: Boolean(accessToken && range.from && range.to), retry: false },
  );
