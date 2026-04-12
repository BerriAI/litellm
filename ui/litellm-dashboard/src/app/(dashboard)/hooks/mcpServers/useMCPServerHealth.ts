import { useCallback, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
  const queryClient = useQueryClient();
  const [recheckingServerIds, setRecheckingServerIds] = useState<Set<string>>(new Set());

  const query = useQuery<MCPServerHealth[]>({
    queryKey: mcpServerHealthKeys.lists(),
    queryFn: async () => await fetchMCPServerHealth(accessToken!),
    enabled: !!accessToken,
    // Refetch health status every 30 seconds to keep it up to date
    refetchInterval: 30000,
  });

  const recheckServerHealth = useCallback(async (serverId: string) => {
    if (!accessToken) return;

    setRecheckingServerIds((prev) => new Set(prev).add(serverId));

    try {
      const result: MCPServerHealth[] = await fetchMCPServerHealth(accessToken, [serverId]);

      queryClient.setQueriesData<MCPServerHealth[]>(
        { queryKey: mcpServerHealthKeys.lists() },
        (oldData) => {
          if (!oldData) return result;
          return oldData.map((h) => {
            const updated = result.find((r) => r.server_id === h.server_id);
            return updated ?? h;
          });
        },
      );
    } finally {
      setRecheckingServerIds((prev) => {
        const next = new Set(prev);
        next.delete(serverId);
        return next;
      });
    }
  }, [accessToken, queryClient]);

  return {
    ...query,
    recheckServerHealth,
    recheckingServerIds,
  };
};
