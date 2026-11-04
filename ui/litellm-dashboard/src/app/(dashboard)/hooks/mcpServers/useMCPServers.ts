import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { disableMCPServerCall, enableMCPServerCall, fetchMCPServers } from "@/components/networking";
import { MCPServer } from "@/components/mcp_tools/types";
import useAuthorized from "../useAuthorized";

export const mcpServersKeys = createQueryKeys("mcpServers");

export const useMCPServers = (teamId?: string | null, userId?: string | null) => {
  const { accessToken } = useAuthorized();
  const filters = { ...(teamId ? { teamId } : {}), ...(userId ? { userId } : {}) };
  return useQuery<MCPServer[]>({
    queryKey: mcpServersKeys.list(Object.keys(filters).length > 0 ? { filters } : undefined),
    queryFn: async () => await fetchMCPServers(accessToken!, teamId, undefined, userId),
    enabled: !!accessToken,
  });
};

export interface ToggleMCPServerEnabledVariables {
  serverId: string;
  enabled: boolean;
}

/**
 * Enable/disable an MCP server. Optimistically flips `enabled` in every cached
 * server list, rolls back on error, and refetches the lists when settled.
 */
export const useToggleMCPServerEnabled = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<
    unknown,
    Error,
    ToggleMCPServerEnabledVariables,
    { previous: [readonly unknown[], MCPServer[] | undefined][] }
  >({
    mutationFn: async ({ serverId, enabled }) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return enabled
        ? await enableMCPServerCall(accessToken, serverId)
        : await disableMCPServerCall(accessToken, serverId);
    },
    onMutate: async ({ serverId, enabled }) => {
      await queryClient.cancelQueries({ queryKey: mcpServersKeys.lists() });
      const previous = queryClient.getQueriesData<MCPServer[]>({ queryKey: mcpServersKeys.lists() });
      queryClient.setQueriesData<MCPServer[]>({ queryKey: mcpServersKeys.lists() }, (old) =>
        old?.map((server) => (server.server_id === serverId ? { ...server, enabled } : server)),
      );
      return { previous };
    },
    onError: (_err, _vars, context) => {
      context?.previous.forEach(([key, data]) => queryClient.setQueryData(key, data));
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: mcpServersKeys.lists() });
    },
  });
};
