import { getUiSettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const uiSettingsKeys = createQueryKeys("uiSettings");

/**
 * UI settings, cached for an hour by default because they rarely change.
 *
 * Both options are per observer in react-query, so a caller reading a value that tracks
 * proxy process state, rather than a persisted setting, can refresh it on its own cadence
 * without changing how long every other caller caches. `staleTime` alone only marks the
 * cached copy stale; a screen that stays mounted and focused never refetches on its own,
 * so a caller that needs to notice a change also has to poll.
 */
export const useUISettings = (options?: { staleTime?: number; refetchInterval?: number }) => {
  return useQuery<Record<string, any>>({
    queryKey: uiSettingsKeys.list({}),
    queryFn: async () => await getUiSettings(),
    staleTime: options?.staleTime ?? 60 * 60 * 1000, // 1 hour - data rarely changes
    gcTime: 60 * 60 * 1000, // 1 hour - keep in cache for 1 hour
    refetchInterval: options?.refetchInterval,
  });
};
