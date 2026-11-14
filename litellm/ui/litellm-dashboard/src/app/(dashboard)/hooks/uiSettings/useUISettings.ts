import { getProxyBaseUrl, getUiSettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const uiSettingsKeys = createQueryKeys("uiSettings");

export type AriaInternalUserAccess = "none" | "own_usage" | "own_and_team_usage";
type UISettings = {
  values?: Record<string, unknown> & {
    aria_internal_user_access?: AriaInternalUserAccess;
    disable_aria?: boolean;
    enabled_ui_pages_internal_users?: string[] | null;
    usage_default_time_range?: string;
    moyai_url?: string | null;
  };
  field_schema?: {
    description?: string;
    properties?: Record<string, { description?: string }>;
  };
  source?: Record<string, "config" | "db" | "env" | "default" | "unset">;
};

/**
 * UI settings, cached for an hour by default because they rarely change.
 *
 * Both options are per observer in react-query, so a caller reading a value that tracks
 * proxy process state, rather than a persisted setting, can refresh it on its own cadence
 * without changing how long every other caller caches. `staleTime` alone only marks the
 * cached copy stale; a screen that stays mounted and focused never refetches on its own,
 * so a caller that needs to notice a change also has to poll.
 */
export const useUISettings = (options?: { staleTime?: number; refetchInterval?: number; enabled?: boolean }) => {
  const proxyBaseUrl = getProxyBaseUrl();
  return useQuery<UISettings>({
    queryKey: [...uiSettingsKeys.list({}), proxyBaseUrl],
    queryFn: async () => await getUiSettings(proxyBaseUrl),
    staleTime: options?.staleTime ?? 60 * 60 * 1000, // 1 hour - data rarely changes
    gcTime: 60 * 60 * 1000, // 1 hour - keep in cache for 1 hour
    refetchInterval: options?.refetchInterval,
    enabled: options?.enabled,
  });
};
