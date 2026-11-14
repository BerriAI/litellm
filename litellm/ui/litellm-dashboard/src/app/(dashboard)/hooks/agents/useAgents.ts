import { getAgentsList } from "@/components/networking";
import { AgentsResponse } from "@/components/agents/types";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const agentsKeys = createQueryKeys("agents");

export const useAgents = () => {
  const { accessToken, userRole } = useAuthorized();
  return useQuery<AgentsResponse>({
    queryKey: agentsKeys.list({}),
    queryFn: async () => await getAgentsList(accessToken!),
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole || ""),
  });
};
