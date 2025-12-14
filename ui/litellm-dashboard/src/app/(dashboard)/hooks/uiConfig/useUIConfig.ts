import { getUiConfig, LiteLLMWellKnownUiConfig } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const uiConfigKeys = createQueryKeys("uiConfig");

export const useUIConfig = () => {
  return useQuery<LiteLLMWellKnownUiConfig>({
    queryKey: uiConfigKeys.list({}),
    queryFn: async () => await getUiConfig(),
    staleTime: 24 * 60 * 60 * 1000, // 24 hours - data rarely changes
    gcTime: 24 * 60 * 60 * 1000, // 24 hours - keep in cache for 24 hours
  });
};
