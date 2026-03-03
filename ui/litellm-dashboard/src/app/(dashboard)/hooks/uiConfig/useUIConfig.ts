import { getUiConfig, LiteLLMWellKnownUiConfig } from "@/components/networking";
import { useProxyConnection } from "@/contexts/ProxyConnectionContext";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const uiConfigKeys = createQueryKeys("uiConfig");

export const useUIConfig = () => {
  const { isRemoteProxy } = useProxyConnection();

  return useQuery<LiteLLMWellKnownUiConfig>({
    queryKey: uiConfigKeys.list({}),
    queryFn: async () => await getUiConfig(),
    staleTime: 24 * 60 * 60 * 1000, // 24 hours - data rarely changes
    gcTime: 24 * 60 * 60 * 1000, // 24 hours - keep in cache for 24 hours
    enabled: !isRemoteProxy, // Don't fetch UI config from remote proxies — it would overwrite proxyBaseUrl
  });
};
