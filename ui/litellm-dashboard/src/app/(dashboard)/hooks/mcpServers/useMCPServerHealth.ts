import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPServerHealth } from "@/components/networking";
import useAuthorized from "../useAuthorized";

const mcpServerHealthKeys = createQueryKeys("mcpServerHealth");

interface MCPServerHealth {
  server_id: string;
  status: string;
}

export const useMCPServerHealth = () => {
  const { accessToken } = useAuthorized();
  return useQuery<MCPServerHealth[]>({
    queryKey: mcpServerHealthKeys.lists(),
    queryFn: async () => await fetchMCPServerHealth(accessToken!),
    enabled: !!accessToken,
    // Refetch health status every 30 seconds to keep it up to date
    refetchInterval: 30000,
  });
};
