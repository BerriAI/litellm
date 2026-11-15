import type { components } from "@/lib/http/schema";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { backgroundFetchClient } from "@/lib/http/api";
import { getProxyBaseUrl } from "@/components/networking";
import { AlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { listedAlertType, type AlertFilter, type ListedAlertType } from "./alertPresentation";

function useSavedAlerts(
  scope: string,
  enabled: boolean,
  status: AlertStatus,
  key: string,
  type: ListedAlertType | null,
) {
  const options = {
    queryKey: ["alerts", getProxyBaseUrl(), scope, status, key],
    enabled: enabled && type !== null,
    initialPageParam: undefined as string | undefined,
    queryFn: async ({ pageParam, signal }: { pageParam: string | undefined; signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts", {
        params: { query: { status, type: type ?? undefined, limit: 25, cursor: pageParam } },
        signal,
      });
      if (!data) throw new Error("Could not load alerts");
      return data;
    },
    getNextPageParam: (page: components["schemas"]["AlertViewPage"]) => page.next_cursor ?? undefined,
    staleTime: 60_000,
    retry: false,
  };
  return useInfiniteQuery(options);
}
export function useAlerts(scope: string, enabled: boolean, status: AlertStatus, filter: AlertFilter) {
  return useSavedAlerts(scope, enabled, status, filter, listedAlertType(filter));
}
export function usePromptCacheAlerts(scope: string, enabled: boolean, status: AlertStatus) {
  return useSavedAlerts(scope, enabled, status, "prompt_cache", "prompt_cache");
}
export function useAlert(id: string | null, scope: string, enabled: boolean) {
  const options = {
    queryKey: ["alert", getProxyBaseUrl(), scope, id],
    enabled: enabled && id !== null,
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/alerts/{alert_id}", {
        params: { path: { alert_id: id! } },
        signal,
      });
      if (!data) throw new Error("Could not load alert");
      return data;
    },
    staleTime: 60_000,
    retry: false,
  };
  return useQuery(options);
}
