import { readFileSync } from "fs";
import { resolve } from "path";
import { vi } from "vitest";
import { hydratePresets, type AutoRouterPresetsResponse } from "@/lib/autorouter_presets";

const BUNDLED_CATALOG_PATH = resolve(__dirname, "../../../../litellm/proxy/public_endpoints/autorouter_presets.json");

export const BUNDLED_PRESETS_RESPONSE = JSON.parse(
  readFileSync(BUNDLED_CATALOG_PATH, "utf8"),
) as AutoRouterPresetsResponse;

export const BUNDLED_PRESETS = hydratePresets(BUNDLED_PRESETS_RESPONSE);

export const LOADED_PRESETS_QUERY = {
  data: BUNDLED_PRESETS,
  isPending: false,
  isError: false,
  refetch: vi.fn(),
};

export const useAutoRouterPresets = vi.fn(() => LOADED_PRESETS_QUERY);
