import { useUISettings } from "./useUISettings";

export const PTU_COST_ATTRIBUTION_SETTING_KEY = "enable_ptu_cost_attribution";

/**
 * Whether the proxy opted into PTU flat-cost attribution.
 *
 * Derived on the proxy from LITELLM_ENABLE_PTU_COST_ATTRIBUTION and returned read-only on
 * /get/ui_settings, so it is not editable from the UI. Anything other than an explicit
 * true (including a settings fetch that has not resolved) counts as off.
 *
 * Polled only once the flag has been seen on. This tracks the proxy process rather than a
 * persisted setting, so an already-open dashboard has to notice a restart that turns the
 * feature off, and a form that stays mounted and focused never refetches on staleTime
 * alone. A deployment that never opts in is the common case and gets the shared one-hour
 * cache, so the poll costs nothing where the feature is unused; the trade is that turning
 * it on reaches an open dashboard on the next natural refetch rather than within 30s.
 */
export const PTU_FLAG_REFRESH_MS = 30 * 1000;

export const usePtuCostAttributionEnabled = (): boolean => {
  const { data } = useUISettings();
  const enabled = data?.values?.[PTU_COST_ATTRIBUTION_SETTING_KEY] === true;
  useUISettings(enabled ? { staleTime: PTU_FLAG_REFRESH_MS, refetchInterval: PTU_FLAG_REFRESH_MS } : undefined);
  return enabled;
};
