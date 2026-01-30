import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPAccessGroups } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
const mcpAccessGroupsKeys = createQueryKeys("mcpAccessGroups");

export const useMCPAccessGroups = () => {
  const { accessToken } = useAuthorized();
  return useQuery<string[]>({
    queryKey: mcpAccessGroupsKeys.list({}),
    queryFn: async () => await fetchMCPAccessGroups(accessToken!),
    enabled: Boolean(accessToken),
  });
};
