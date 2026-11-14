import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";

export type Source = components["schemas"]["Source"];
export type Connection = components["schemas"]["Connection"];
export type AgentPage = components["schemas"]["AgentPage"];
export type SpendReport = components["schemas"]["SpendReport"];
export type CloudAgent = AgentPage["data"][number];

const root = "/v1/external_cloud_agents";
const prefix = ["external-cloud-agents"] as const;

export function useCloudSources(accessToken: string) {
  const options = {
    queryKey: [...prefix, accessToken, "sources"],
    queryFn: () => apiClient.get<Source[]>(root + "/sources", { accessToken }),
    refetchInterval: 3000,
    retry: false,
  };
  return useQuery(options);
}

export function useCloudBots(accessToken: string, sourceId: string, search: string, page: number) {
  const options = {
    queryKey: [...prefix, accessToken, sourceId, "bots", search, page],
    queryFn: () =>
      apiClient.get<AgentPage>(root + "/agents", {
        accessToken,
        query: { source_id: sourceId, search, page, page_size: 20 },
      }),
    refetchInterval: 10000,
    retry: false,
  };
  return useQuery(options);
}

export function useCloudSpend(accessToken: string, source: Source, days: number) {
  const options = {
    queryKey: [...prefix, accessToken, source.source_id, "spend", days, source.last_spend_sync_at],
    queryFn: () =>
      apiClient.get<SpendReport>(root + "/spend", {
        accessToken,
        query: { source_id: source.source_id, days },
      }),
    enabled: source.has_admin_api_key,
    retry: false,
  };
  return useQuery(options);
}

export function useCloudMutation<T>(accessToken: string, method: "POST" | "PUT" | "DELETE" | "GET", path: string) {
  const client = useQueryClient();
  return useMutation<T, Error, Record<string, string> | void>({
    mutationFn: (body) => apiClient.request<T>(method, root + path, { accessToken, body: body || undefined }),
    onSuccess: () => client.invalidateQueries({ queryKey: prefix }),
  });
}
