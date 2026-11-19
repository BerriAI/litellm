import { useQuery } from "@tanstack/react-query";
import useAuthorized from "../../hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { loadApps } from "./appApi";

export const grantedAppsKey = ["apps", "granted"] as const;

export function useGrantedApps() {
  const { accessToken, userRole } = useAuthorized();
  const isProxyAdmin = isProxyAdminRole(userRole);
  const apps = useQuery({
    queryKey: grantedAppsKey,
    queryFn: () => loadApps(),
    enabled: Boolean(accessToken) && !isProxyAdmin,
    retry: false,
  });
  return { isProxyAdmin, apps, hasGrant: (apps.data?.length ?? 0) > 0 };
}
