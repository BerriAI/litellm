import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPAccessGroups } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
const mcpAccessGroupsKeys = createQueryKeys("mcpAccessGroups");

export const useMCPAccessGroups = (teamId?: string) => {
  const { accessToken } = useAuthorized();
  return useQuery<string[]>({
    queryKey: mcpAccessGroupsKeys.list({ teamId }),
    queryFn: async () => await fetchMCPAccessGroups(accessToken!, teamId),
    enabled: Boolean(accessToken),
  });
};
