import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPServers } from "@/components/networking";
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
