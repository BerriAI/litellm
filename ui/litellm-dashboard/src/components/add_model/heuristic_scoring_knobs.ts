// Type aliases rather than interfaces: only an alias gets TypeScript's implicit index signature, which is
// what lets these be read as a Record<string, number> by the generic knob rendering.
export type TierBoundaries = {
  simple_medium: number;
  medium_complex: number;
  complex_reasoning: number;
};

export type TokenThresholds = {
  simple: number;
  complex: number;
};

/**
 * Display order for the scorer's 7 dimensions, highest default weight first. Fixed rather than sorted by
 * current weight so a row cannot jump out from under the cursor mid-drag.
 */
export const DIMENSION_KEYS = [
  "codePresence",
  "reasoningMarkers",
  "technicalTerms",
  "tokenCount",
  "simpleIndicators",
  "multiStepPatterns",
  "questionComplexity",
] as const;

export type DimensionKey = (typeof DIMENSION_KEYS)[number];

export type DimensionWeights = Record<DimensionKey, number>;

export const DIMENSION_LABELS: Record<DimensionKey, string> = {
  codePresence: "Code presence",
  reasoningMarkers: "Reasoning markers",
  technicalTerms: "Technical terms",
  tokenCount: "Token count",
  simpleIndicators: "Simple indicators",
  multiStepPatterns: "Multi-step patterns",
  questionComplexity: "Question complexity",
};

/**
 * Mirrors DEFAULT_TIER_BOUNDARIES / DEFAULT_TOKEN_THRESHOLDS / DEFAULT_DIMENSION_WEIGHTS in
 * litellm/router_strategy/complexity_router/config.py. These are placeholders the controls display, never
 * values a save writes: an untouched knob is omitted from the payload so the router keeps tracking whatever
 * the backend default becomes. Drift here therefore shows a stale number, and cannot pin a router to one.
 */
export const DEFAULT_TIER_BOUNDARIES: TierBoundaries = {
  simple_medium: 0.15,
  medium_complex: 0.35,
  complex_reasoning: 0.6,
};

export const DEFAULT_TOKEN_THRESHOLDS: TokenThresholds = { simple: 15, complex: 400 };

export const DEFAULT_DIMENSION_WEIGHTS: DimensionWeights = {
  codePresence: 0.3,
  reasoningMarkers: 0.25,
  technicalTerms: 0.25,
  tokenCount: 0.1,
  simpleIndicators: 0.05,
  multiStepPatterns: 0.03,
  questionComplexity: 0.02,
};

const asRecord = (raw: unknown): Record<string, unknown> | undefined =>
  typeof raw === "object" && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : undefined;

/**
 * Absent means the router is tracking the backend defaults, so it must hydrate to undefined rather than to a
 * copy of them: hydrating defaults would make an untouched save write them out and pin the router to today's
 * numbers. A stored dict missing individual keys is filled from the defaults, which is what the backend's own
 * `.get(key, default)` reads would have applied at scoring time anyway.
 */
const hydrateNumericMap = <K extends string>(
  raw: unknown,
  defaults: Record<K, number>,
  keys: readonly K[],
): Record<K, number> | undefined => {
  const stored = asRecord(raw);
  if (stored === undefined) return undefined;
  return Object.fromEntries(
    keys.map((key) => {
      const value = stored[key];
      return [key, typeof value === "number" && Number.isFinite(value) ? value : defaults[key]];
    }),
  ) as Record<K, number>;
};

const BOUNDARY_KEYS = ["simple_medium", "medium_complex", "complex_reasoning"] as const;
const THRESHOLD_KEYS = ["simple", "complex"] as const;

export const hydrateTierBoundaries = (raw: unknown): TierBoundaries | undefined =>
  hydrateNumericMap(raw, DEFAULT_TIER_BOUNDARIES, BOUNDARY_KEYS);

export const hydrateTokenThresholds = (raw: unknown): TokenThresholds | undefined =>
  hydrateNumericMap(raw, DEFAULT_TOKEN_THRESHOLDS, THRESHOLD_KEYS);

export const hydrateDimensionWeights = (raw: unknown): DimensionWeights | undefined =>
  hydrateNumericMap(raw, DEFAULT_DIMENSION_WEIGHTS, DIMENSION_KEYS);

export const weightTotal = (weights: DimensionWeights): number =>
  Math.round(DIMENSION_KEYS.reduce((total, key) => total + weights[key], 0) * 100) / 100;
