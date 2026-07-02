import type { components } from "@/lib/http/schema";
import { $api, authHeader } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export type AccessGroupResponse = components["schemas"]["AccessGroupResponse"];

export const useAccessGroups = () => {
  const { accessToken, userRole } = useAuthorized();

  return $api.useQuery(
    "get",
    "/v1/access_group",
    { headers: authHeader(accessToken!) },
    { enabled: Boolean(accessToken) && all_admin_roles.includes(userRole || "") },
  );
};
