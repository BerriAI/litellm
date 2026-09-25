import { formatDate } from "@/components/networking";
import { $api } from "@/lib/http/api";

import type { DateRange } from "./useDailyActivityRange";

export const useCacheLeakageKeys = (accessToken: string | null, range: DateRange, scopeUserId: string | null) =>
  $api.useQuery(
    "get",
    "/user/daily/activity/cache_leakage",
    {
      params: {
        query: {
          start_date: range.from ? formatDate(range.from) : undefined,
          end_date: range.to ? formatDate(range.to) : undefined,
          timezone: new Date().getTimezoneOffset(),
          include_current_utc_day: true,
          user_id: scopeUserId ?? undefined,
        },
      },
    },
    { enabled: Boolean(accessToken && range.from && range.to), retry: false },
  );
