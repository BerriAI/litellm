import { useQuery } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { upstreamProxyListCall, type UpstreamProxyList } from "@/components/upstream_proxies/api";
import { all_admin_roles } from "@/utils/roles";

import { createQueryKeys } from "../common/queryKeysFactory";

export const upstreamProxiesKeys = createQueryKeys("upstreamProxies");

export const useUpstreamProxies = ({ enabled = true }: { enabled?: boolean } = {}) => {
  const { accessToken, userRole } = useAuthorized();
  return useQuery<UpstreamProxyList>({
    queryKey: upstreamProxiesKeys.list({}),
    queryFn: async () => await upstreamProxyListCall(accessToken!),
    enabled: enabled && Boolean(accessToken) && all_admin_roles.includes(userRole ?? ""),
  });
};
