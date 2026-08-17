import {
  DEFAULT_DIMENSION_WEIGHTS,
  DEFAULT_TIER_BOUNDARIES,
  DEFAULT_TOKEN_THRESHOLDS,
  DIMENSION_KEYS,
  DimensionWeights,
  TierBoundaries,
  TokenThresholds,
} from "./ComplexityRouterConfig";

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
