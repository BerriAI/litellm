import { useQuery } from "@tanstack/react-query";
import useAuthorized from "../../hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import { ApiError } from "@/lib/http/client";
import { loadApps } from "./appApi";

export const grantedAppsKey = ["apps", "granted"] as const;

export const readsEveryApp = (userRole: string, isViewOnly: boolean): boolean =>
  isProxyAdminRole(userRole) && !isViewOnly;

export const grantedAppsFailed = (error: unknown): boolean =>
  error != null && !(error instanceof ApiError && error.status === 403);

export function useGrantedApps() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const isProxyAdmin = readsEveryApp(userRole, isViewOnly);
  const enabled = Boolean(accessToken) && !isProxyAdmin;
  const query = { queryKey: grantedAppsKey, queryFn: () => loadApps(), enabled, retry: false };
  const apps = useQuery(query);
  return { isProxyAdmin, apps, hasGrant: (apps.data?.length ?? 0) > 0, failed: grantedAppsFailed(apps.error) };
}
