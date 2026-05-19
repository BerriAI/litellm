import { getUiSettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const uiSettingsKeys = createQueryKeys("uiSettings");

export const useUISettings = () => {
  return useQuery<Record<string, any>>({
    queryKey: uiSettingsKeys.list({}),
    queryFn: async () => await getUiSettings(),
    staleTime: 60 * 60 * 1000, // 1 hour - data rarely changes
    gcTime: 60 * 60 * 1000, // 1 hour - keep in cache for 1 hour
  });
};
