import { vi } from "vitest";

/**
 * Stubs the proxy's registered classifier plugin names for any test that renders the auto-router tree.
 * Exported as a vi.fn so a test can override the query state, which is how the empty-registry and
 * failed-fetch paths are covered.
 */
export const REGISTERED_CLASSIFIER_PLUGINS = ["spend-aware", "tier-by-team"];

export const LOADED_CLASSIFIER_PLUGINS_QUERY = {
  data: REGISTERED_CLASSIFIER_PLUGINS,
  isPending: false,
  isError: false,
  refetch: vi.fn(),
};

export const useClassifierPlugins = vi.fn(() => LOADED_CLASSIFIER_PLUGINS_QUERY);
