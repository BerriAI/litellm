import { vi } from "vitest";
import bundledPresets from "../../../../litellm/proxy/public_endpoints/autorouter_presets.json";
import { hydratePresets, type AutoRouterPresetsResponse } from "@/lib/autorouter_presets";

// Derived from the real bundled catalog so a preset edit there flows into test expectations
// instead of redding on a stale copy. Exported as vi.fn so a test can override the query state.
export const BUNDLED_PRESETS = hydratePresets(bundledPresets as AutoRouterPresetsResponse);

export const LOADED_PRESETS_QUERY = {
  data: BUNDLED_PRESETS,
  isPending: false,
  isError: false,
  refetch: vi.fn(),
};

export const useAutoRouterPresets = vi.fn(() => LOADED_PRESETS_QUERY);
