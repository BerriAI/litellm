export type TierBoundaries = Record<string, number>;

export type TokenThresholds = Record<string, number>;

export type DimensionWeights = Record<string, number>;

/**
 * Display names for the scorer's dimensions. Only the wording lives here; the dimension set and its
 * shipped weights come from the proxy (GET /public/complexity_router/scorer_defaults), so a dimension
 * added backend-side still renders, under its raw key until it is given a label here.
 */
export const DIMENSION_LABELS: Record<string, string> = {
  codePresence: "Code presence",
  reasoningMarkers: "Reasoning markers",
  technicalTerms: "Technical terms",
  tokenCount: "Token count",
  simpleIndicators: "Simple indicators",
  multiStepPatterns: "Multi-step patterns",
  questionComplexity: "Question complexity",
};

export const dimensionLabel = (key: string): string => DIMENSION_LABELS[key] ?? key;

const asRecord = (raw: unknown): Record<string, unknown> | undefined =>
  typeof raw === "object" && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : undefined;

/**
 * Absent means the router is tracking the shipped defaults, so it must hydrate to undefined rather than to
 * a copy of them: hydrating defaults would make an untouched save write them out and pin the router to
 * whatever they were the day the modal was opened. A stored dict is kept exactly as stored, since the
 * backend fills in any key it omits at scoring time.
 */
const hydrateNumericMap = (raw: unknown): Record<string, number> | undefined => {
  const stored = asRecord(raw);
  if (stored === undefined) return undefined;
  return Object.fromEntries(
    Object.entries(stored).filter(([, value]) => typeof value === "number" && Number.isFinite(value)),
  ) as Record<string, number>;
};

export const hydrateTierBoundaries = (raw: unknown): TierBoundaries | undefined => hydrateNumericMap(raw);

export const hydrateTokenThresholds = (raw: unknown): TokenThresholds | undefined => hydrateNumericMap(raw);

export const hydrateDimensionWeights = (raw: unknown): DimensionWeights | undefined => hydrateNumericMap(raw);

/**
 * The scalar counterpart of hydrateNumericMap: absent hydrates to undefined so an untouched save keeps the
 * floor tracking tier_boundaries.simple_medium, while a stored 0 hydrates to 0, which is a real floor.
 */
export const hydrateReasoningOverrideMinScore = (raw: unknown): number | undefined =>
  typeof raw === "number" && Number.isFinite(raw) ? raw : undefined;

export const weightTotal = (weights: DimensionWeights): number =>
  Math.round(Object.values(weights).reduce((total, weight) => total + weight, 0) * 100) / 100;
