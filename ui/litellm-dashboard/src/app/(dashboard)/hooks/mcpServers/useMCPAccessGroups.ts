import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPAccessGroups } from "@/components/networking";

const mcpAccessGroupsKeys = createQueryKeys("mcpAccessGroups");

export const useMCPAccessGroups = (accessToken: string | null) => {
  return useQuery<string[]>({
    queryKey: mcpAccessGroupsKeys.list({}),
    queryFn: async () => await fetchMCPAccessGroups(accessToken!),
    enabled: !!accessToken,
  });
};
