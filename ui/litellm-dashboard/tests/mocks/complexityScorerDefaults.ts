import { vi } from "vitest";
import type { ComplexityScorerDefaults } from "@/components/networking";

/**
 * Stubs the proxy's shipped scorer defaults for any test that renders the auto-router tree.
 *
 * The Advanced scoring panel and the "How Classification Works" card read these over the network, so
 * without a stub every render of that tree pays for a request jsdom cannot serve, which pushed the slowest
 * auto-router tests past their timeout on CI. Exported as a vi.fn so a test can override the query state,
 * which is how the failure path is covered.
 */
export const SHIPPED_SCORER_DEFAULTS: ComplexityScorerDefaults = {
  tier_boundaries: { simple_medium: 0.15, medium_complex: 0.35, complex_reasoning: 0.6 },
  token_thresholds: { simple: 15, complex: 400 },
  dimension_weights: {
    codePresence: 0.3,
    reasoningMarkers: 0.25,
    technicalTerms: 0.25,
    tokenCount: 0.1,
    simpleIndicators: 0.05,
    multiStepPatterns: 0.03,
    questionComplexity: 0.02,
  },
};

export const LOADED_SCORER_DEFAULTS_QUERY = {
  data: SHIPPED_SCORER_DEFAULTS,
  isPending: false,
  isError: false,
  refetch: vi.fn(),
};

export const useComplexityScorerDefaults = vi.fn(() => LOADED_SCORER_DEFAULTS_QUERY);
