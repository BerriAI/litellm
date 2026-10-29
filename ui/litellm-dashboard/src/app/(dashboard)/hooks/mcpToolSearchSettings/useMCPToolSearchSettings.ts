import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { createQueryKeys } from "../common/queryKeysFactory";

export type MCPToolSearchSettings = components["schemas"]["MCPToolSearchSettings"];
export type MCPToolSearchSettingsResponse = components["schemas"]["MCPToolSearchSettingsResponse"];

const GET_PATH = "/get/mcp_tool_search_settings";
const UPDATE_PATH = "/update/mcp_tool_search_settings";

const mcpToolSearchSettingsKeys = createQueryKeys("mcpToolSearchSettings");

export const getMCPToolSearchSettings = (accessToken: string): Promise<MCPToolSearchSettingsResponse> =>
  apiClient.get<MCPToolSearchSettingsResponse>(GET_PATH, { accessToken });

export const updateMCPToolSearchSettings = (
  accessToken: string,
  settings: MCPToolSearchSettings,
): Promise<MCPToolSearchSettings> =>
  apiClient.patch<MCPToolSearchSettings>(UPDATE_PATH, { accessToken, body: settings });

export const useMCPToolSearchSettings = () => {
  const { accessToken } = useAuthorized();
  return useQuery<MCPToolSearchSettingsResponse>({
    queryKey: mcpToolSearchSettingsKeys.list({}),
    queryFn: () => getMCPToolSearchSettings(accessToken),
    enabled: !!accessToken,
  });
};

export const useUpdateMCPToolSearchSettings = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useMutation<MCPToolSearchSettings, Error, MCPToolSearchSettings>({
    mutationFn: (settings) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateMCPToolSearchSettings(accessToken, settings);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: mcpToolSearchSettingsKeys.all });
    },
  });
};
