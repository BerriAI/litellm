import { useQueries, useQuery, useQueryClient, type UseQueryOptions } from "@tanstack/react-query";
import { fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import type { MCPServer, MCPTool, MCPToolset } from "@/components/mcp_tools/types";
import {
  fetchMCPServers,
  fetchMCPToolsets,
  getGuardrailsList,
  getPoliciesList,
  listMCPTools,
  tagListCall,
  vectorStoreListCall,
} from "@/components/networking";
import { getPolicyOptionEntries } from "@/components/policies/PolicySelector";
import type { Policy } from "@/components/policies/types";
import type { Tag } from "@/components/tag_management/types";
import type { VectorStore } from "@/components/vector_store_management/types";
import { fetchAvailableAgents, type Agent } from "../../llm_calls/fetch_agents";

export type PickerOption = Readonly<{ value: string; label: string; description?: string }>;

const EMPTY: readonly never[] = [];

const serverList = (response: MCPServer[] | { data?: MCPServer[] }): readonly MCPServer[] =>
  Array.isArray(response) ? response : response.data ?? [];

export function usePlaygroundModels(requestKey: string | null) {
  const options = {
    queryKey: ["playground", "models", requestKey],
    queryFn: (): Promise<readonly ModelGroup[]> => fetchAvailableModels(requestKey ?? ""),
    enabled: Boolean(requestKey),
    retry: false,
  } satisfies UseQueryOptions<readonly ModelGroup[]>;
  return useQuery(options);
}

export function useMcpCatalog(requestKey: string | null) {
  const queryClient = useQueryClient();
  const servers = useQuery<readonly MCPServer[]>({
    queryKey: ["playground", "mcpServers", requestKey],
    queryFn: async () => serverList(await fetchMCPServers(requestKey ?? "")),
    enabled: Boolean(requestKey),
  });
  const toolsets = useQuery<readonly MCPToolset[] | null>({
    queryKey: ["playground", "mcpToolsets", requestKey],
    queryFn: async () => {
      const response: unknown = await fetchMCPToolsets(requestKey ?? "").catch(() => null);
      if (response === null) return null;
      return Array.isArray(response) ? (response as MCPToolset[]) : [];
    },
    enabled: Boolean(requestKey),
  });
  return {
    servers: servers.data ?? EMPTY,
    toolsets: toolsets.data ?? EMPTY,
    isLoading: servers.isLoading || toolsets.isLoading,
    isReady: servers.isSuccess && Boolean(toolsets.data),
    refreshCredentials: () => {
      void servers.refetch();
      void queryClient.invalidateQueries({ queryKey: ["playground", "mcpTools", requestKey] });
    },
  };
}

export type ServerTools = Readonly<{ tools: readonly MCPTool[]; isLoading: boolean; failed: boolean }>;

export function useServerTools(requestKey: string | null, serverIds: readonly string[]) {
  const results = useQueries({
    queries: serverIds.map((serverId) => ({
      queryKey: ["playground", "mcpTools", requestKey, serverId],
      queryFn: async (): Promise<readonly MCPTool[]> => {
        const response: { tools?: MCPTool[]; error?: string } = await listMCPTools(requestKey ?? "", serverId);
        if (response.error) throw new Error(response.error);
        return response.tools ?? [];
      },
      enabled: Boolean(requestKey),
      staleTime: 60_000,
    })),
  });
  return Object.fromEntries(
    serverIds.map((serverId, index) => [
      serverId,
      {
        tools: results[index]?.data ?? EMPTY,
        isLoading: results[index]?.isLoading ?? false,
        failed: results[index]?.isError ?? false,
      },
    ]),
  ) as Readonly<Record<string, ServerTools>>;
}

export function usePlaygroundAgents(requestKey: string | null, proxyBaseUrl: string, enabled: boolean) {
  return useQuery<readonly Agent[]>({
    queryKey: ["playground", "agents", requestKey, proxyBaseUrl],
    queryFn: () => fetchAvailableAgents(requestKey ?? "", proxyBaseUrl || undefined),
    enabled: enabled && Boolean(requestKey),
  });
}

export function useGuardrailOptions(uiToken: string | null) {
  return useQuery<readonly PickerOption[]>({
    queryKey: ["playground", "guardrails", uiToken],
    queryFn: async () => {
      const response: { guardrails?: { guardrail_name?: string | null }[] } = await getGuardrailsList(uiToken ?? "");
      return (response.guardrails ?? []).flatMap((guardrail) =>
        guardrail.guardrail_name ? [{ value: guardrail.guardrail_name, label: guardrail.guardrail_name }] : [],
      );
    },
    enabled: Boolean(uiToken),
  });
}

export function usePolicyOptions(uiToken: string | null, canView: boolean) {
  return useQuery<readonly PickerOption[]>({
    queryKey: ["playground", "policies", uiToken],
    queryFn: async () => {
      const response: { policies?: Policy[] } = await getPoliciesList(uiToken ?? "");
      return getPolicyOptionEntries(response.policies ?? []);
    },
    enabled: canView && Boolean(uiToken),
  });
}

export function useVectorStoreOptions(uiToken: string | null) {
  return useQuery<readonly PickerOption[]>({
    queryKey: ["playground", "vectorStores", uiToken],
    queryFn: async () => {
      const response: { data?: VectorStore[] } = await vectorStoreListCall(uiToken ?? "");
      return (response.data ?? []).map((store) => ({
        value: store.vector_store_id,
        label: store.vector_store_name || store.vector_store_id,
        description: store.vector_store_name ? store.vector_store_id : undefined,
      }));
    },
    enabled: Boolean(uiToken),
  });
}

export function useTagOptions(uiToken: string | null) {
  return useQuery<readonly string[]>({
    queryKey: ["playground", "tags", uiToken],
    queryFn: async () => {
      const response: Record<string, Tag> = await tagListCall(uiToken ?? "");
      return Object.values(response).map((tag) => tag.name);
    },
    enabled: Boolean(uiToken),
  });
}
