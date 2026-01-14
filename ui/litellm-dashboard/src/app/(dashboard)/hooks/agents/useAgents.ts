import { getAgentsList } from "@/components/networking";
import { AgentsResponse } from "@/components/agents/types";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { all_admin_roles } from "@/utils/roles";

const agentsKeys = createQueryKeys("agents");

export const useAgents = (accessToken: string | null, userRole: string | null) => {
  return useQuery<AgentsResponse>({
    queryKey: agentsKeys.list({}),
    queryFn: async () => await getAgentsList(accessToken!),
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole || ""),
  });
};
