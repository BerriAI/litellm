import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, getProxyBaseUrl, getRuntimeBackendBaseUrl } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import type { QueryParams, HttpMethod } from "@/lib/http/client";

export type Source = components["schemas"]["Source"];
export type Connection = components["schemas"]["SourceConnection"];
export type SyncResult = components["schemas"]["MemberSyncResult"];
export type Bot = components["schemas"]["Agent"];
export type Policy = components["schemas"]["Policy"];
export type Decision = components["schemas"]["Decision"];
export type Summary = components["schemas"]["ExternalAgentSummary"];
export type PolicyInput = components["schemas"]["PolicyPut"];
export type BulkInput = components["schemas"]["AgentPatch"];
export const externalKey = ["external-agents"] as const;
const root = "/v1/external_agents";

export function hookUrl(sourceId: string) {
  return `${(getRuntimeBackendBaseUrl() || getProxyBaseUrl()).replace(/\/$/, "")}/grok_bot/inference_hook/${sourceId}`;
}

export function useExternalQuery<T>(token: string, path: string, query?: QueryParams, enabled = true) {
  const options = {
    queryKey: [...externalKey, path, query],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      apiClient.get<T>(root + path, { accessToken: token, query, signal }),
    enabled: Boolean(token) && enabled,
    refetchInterval: 5000,
  };
  return useQuery<T>(options);
}

export function useExternalMutation(token: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ method, path, body }: { method: HttpMethod; path: string; body: unknown }) =>
      apiClient.request<unknown>(method, root + path, { accessToken: token, body }),
    onSuccess: () => client.invalidateQueries({ queryKey: externalKey }),
  });
}
