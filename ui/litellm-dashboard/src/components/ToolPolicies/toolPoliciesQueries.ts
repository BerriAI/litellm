import { queryOptions } from "@tanstack/react-query";

import { fetchToolsList, type ToolRow } from "@/components/networking";

export const toolPoliciesKeys = {
  all: ["tool-policies"] as const,
  list: (accessToken: string | null) => [...toolPoliciesKeys.all, accessToken] as const,
};

export const toolPoliciesListOptions = (accessToken: string | null) =>
  queryOptions({
    queryKey: toolPoliciesKeys.list(accessToken),
    queryFn: async (): Promise<ToolRow[]> => (accessToken === null ? [] : fetchToolsList(accessToken)),
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
