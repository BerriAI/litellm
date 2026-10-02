import { useQuery } from "@tanstack/react-query";

import type { CacheLeakageKeysResponse } from "@/components/UsagePage/dailyActivityApi";
import { cacheLeakageKeysQueryOptions } from "@/app/(dashboard)/hooks/dailyActivity/dailyActivityQueries";
import { buildDailyActivityRequest, type DailyActivityScope } from "./useDailyActivityRange";

const selectApiKeys = (response: CacheLeakageKeysResponse) => response.api_keys;

export const useCacheLeakageKeys = (scope: DailyActivityScope, enabled: boolean) => {
  const request = enabled ? buildDailyActivityRequest(scope) : null;
  return useQuery({ ...cacheLeakageKeysQueryOptions(request), select: selectApiKeys });
};
