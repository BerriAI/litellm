import { readFileSync } from "fs";
import { resolve } from "path";
import { vi } from "vitest";
import { hydratePresets, type AutoRouterPresetsResponse } from "@/lib/autorouter_presets";

// Derived from the real bundled catalog so a preset edit there flows into test expectations
// instead of redding on a stale copy. Exported as vi.fn so a test can override the query state.
const CATALOG_PATH = resolve(__dirname, "../../../../litellm/proxy/public_endpoints/autorouter_presets.json");

export const BUNDLED_PRESETS_RESPONSE = JSON.parse(readFileSync(CATALOG_PATH, "utf8")) as AutoRouterPresetsResponse;

export const BUNDLED_PRESETS = hydratePresets(BUNDLED_PRESETS_RESPONSE);

export const LOADED_PRESETS_QUERY = {
  data: BUNDLED_PRESETS,
  isPending: false,
  isError: false,
  refetch: vi.fn(),
};

export const useAutoRouterPresets = vi.fn(() => LOADED_PRESETS_QUERY);
