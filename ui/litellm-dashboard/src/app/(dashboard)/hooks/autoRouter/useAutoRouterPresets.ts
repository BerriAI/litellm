import { AutoRouterPreset, hydratePresets } from "@/lib/autorouter_presets";
import { getAutoRouterPresets } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const presetKeys = createQueryKeys("autoRouterPresets");

export const useAutoRouterPresets = () => {
  const options = {
    queryKey: presetKeys.list({}),
    queryFn: async () => hydratePresets(await getAutoRouterPresets()),
    staleTime: 24 * 60 * 60 * 1000,
    gcTime: 24 * 60 * 60 * 1000,
  };
  return useQuery<AutoRouterPreset[]>(options);
};
