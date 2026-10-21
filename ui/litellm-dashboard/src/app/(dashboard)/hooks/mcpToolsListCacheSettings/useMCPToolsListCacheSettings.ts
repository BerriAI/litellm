import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { createQueryKeys } from "../common/queryKeysFactory";

export type MCPToolsListCacheSettings = components["schemas"]["MCPToolsListCacheSettings"];
export type MCPToolsListCacheSettingsResponse = components["schemas"]["MCPToolsListCacheSettingsResponse"];

const GET_PATH = "/get/mcp_tools_list_cache_settings";
const UPDATE_PATH = "/update/mcp_tools_list_cache_settings";

const mcpToolsListCacheSettingsKeys = createQueryKeys("mcpToolsListCacheSettings");

export const getMCPToolsListCacheSettings = (accessToken: string): Promise<MCPToolsListCacheSettingsResponse> =>
  apiClient.get<MCPToolsListCacheSettingsResponse>(GET_PATH, { accessToken });

export const updateMCPToolsListCacheSettings = (
  accessToken: string,
  settings: MCPToolsListCacheSettings,
): Promise<MCPToolsListCacheSettings> =>
  apiClient.patch<MCPToolsListCacheSettings>(UPDATE_PATH, { accessToken, body: settings });

export const useMCPToolsListCacheSettings = () => {
  const { accessToken } = useAuthorized();
  return useQuery<MCPToolsListCacheSettingsResponse>({
    queryKey: mcpToolsListCacheSettingsKeys.list({}),
    queryFn: () => getMCPToolsListCacheSettings(accessToken),
    enabled: !!accessToken,
  });
};

export const useUpdateMCPToolsListCacheSettings = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useMutation<MCPToolsListCacheSettings, Error, MCPToolsListCacheSettings>({
    mutationFn: (settings) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateMCPToolsListCacheSettings(accessToken, settings);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mcpToolsListCacheSettingsKeys.all });
    },
  });
};
