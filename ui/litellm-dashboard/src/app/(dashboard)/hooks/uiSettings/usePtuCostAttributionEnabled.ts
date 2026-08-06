import { useUISettings } from "./useUISettings";

export const PTU_COST_ATTRIBUTION_SETTING_KEY = "enable_ptu_cost_attribution";

/**
 * Whether the proxy opted into PTU flat-cost attribution.
 *
 * Derived on the proxy from LITELLM_ENABLE_PTU_COST_ATTRIBUTION and returned read-only
 * on /get/ui_settings, so it is not editable from the UI. Anything other than an
 * explicit true (including a settings fetch that has not resolved) counts as off.
 *
 * Re-read far more often than the other UI settings. Those are persisted records, while
 * this tracks the proxy process, so a restart that flips the variable has to reach a
 * dashboard that is already open. A short staleTime is not enough on its own: a model form
 * that stays mounted and focused never refetches, so the flag is polled as well.
 */
/** How long a cached copy of the flag stays fresh, and how often it is re-read, in milliseconds. */
export const PTU_FLAG_REFRESH_MS = 30 * 1000;

export const usePtuCostAttributionEnabled = (): boolean => {
  const { data } = useUISettings({ staleTime: PTU_FLAG_REFRESH_MS, refetchInterval: PTU_FLAG_REFRESH_MS });
  return data?.values?.[PTU_COST_ATTRIBUTION_SETTING_KEY] === true;
};
