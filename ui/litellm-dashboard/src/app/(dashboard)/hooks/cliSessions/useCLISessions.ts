import { keepPreviousData, useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { proxyAdminTierRoles } from "@/utils/roles";

export type CLISessionResponse = components["schemas"]["CLISessionResponse"];

const CLI_SESSION_LIST_KEY = ["get", "/cli/session/list"] as const;

export const useCLISessions = (page: number, pageSize: number) => {
  const { accessToken, userRole } = useAuthorized();

  return $api.useQuery(
    "get",
    "/cli/session/list",
    { params: { query: { page, page_size: pageSize } } },
    {
      enabled: Boolean(accessToken) && proxyAdminTierRoles.includes(userRole || ""),
      staleTime: 30000,
      placeholderData: keepPreviousData,
    },
  );
};

export const useRevokeCLISession = () => {
  const queryClient = useQueryClient();

  return $api.useMutation("post", "/cli/session/{session_id}/revoke", {
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CLI_SESSION_LIST_KEY });
    },
  });
};
