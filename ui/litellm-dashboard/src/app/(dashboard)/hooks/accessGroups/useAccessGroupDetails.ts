import { useQueryClient } from "@tanstack/react-query";
import { $api, authHeader } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { AccessGroupResponse } from "./useAccessGroups";

export const useAccessGroupDetails = (accessGroupId?: string) => {
  const { accessToken, userRole } = useAuthorized();
  const queryClient = useQueryClient();

  return $api.useQuery(
    "get",
    "/v1/access_group/{access_group_id}",
    { params: { path: { access_group_id: accessGroupId ?? "" } }, headers: authHeader(accessToken!) },
    {
      enabled: Boolean(accessToken && accessGroupId) && all_admin_roles.includes(userRole || ""),
      initialData: () => {
        if (!accessGroupId) return undefined;
        const listKey = $api.queryOptions("get", "/v1/access_group", { headers: authHeader(accessToken!) }).queryKey;
        const groups = queryClient.getQueryData<AccessGroupResponse[]>(listKey);
        return groups?.find((g) => g.access_group_id === accessGroupId);
      },
    },
  );
};
