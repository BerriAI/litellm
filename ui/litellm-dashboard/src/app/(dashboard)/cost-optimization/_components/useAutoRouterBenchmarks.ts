import { formatDate } from "@/components/networking";
import { $api } from "@/lib/http/api";

import type { DateRange } from "./useDailyActivityRange";

export const useAutoRouterBenchmarks = (accessToken: string | null, range: DateRange) =>
  $api.useQuery(
    "get",
    "/auto_router/benchmarks",
    {
      params: {
        query: range.from && range.to ? { start_date: formatDate(range.from), end_date: formatDate(range.to) } : {},
      },
    },
    { enabled: Boolean(accessToken && range.from && range.to), retry: false },
  );
