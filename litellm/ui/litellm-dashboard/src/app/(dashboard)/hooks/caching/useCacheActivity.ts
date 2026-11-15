import { $api } from "@/lib/http/api";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

export type CacheActivityResponse = components["schemas"]["CacheActivityResponse"];
export type CacheActivityGroup = components["schemas"]["CacheActivityGroup"];

export interface CacheActivityParams {
  startDate: string | undefined;
  endDate: string | undefined;
  keyAliases: string[];
  models: string[];
}

export const useCacheActivity = ({ startDate, endDate, keyAliases, models }: CacheActivityParams) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    "/global/activity/cache_hits",
    {
      params: {
        query: {
          start_date: startDate ?? "",
          end_date: endDate ?? "",
          key_aliases: keyAliases,
          models,
        },
      },
    },
    { enabled: Boolean(accessToken && startDate && endDate) },
  );
};
