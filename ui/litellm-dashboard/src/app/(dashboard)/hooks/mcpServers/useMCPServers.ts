import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPServers } from "@/components/networking";
import { MCPServer } from "@/components/mcp_tools/types";
import useAuthorized from "../useAuthorized";

const mcpServersKeys = createQueryKeys("mcpServers");

export const useMCPServers = (teamId?: string) => {
  const { accessToken } = useAuthorized();
  return useQuery<MCPServer[]>({
    queryKey: mcpServersKeys.list({ teamId }),
    queryFn: async () => await fetchMCPServers(accessToken!, teamId),
    enabled: !!accessToken,
  });
};
