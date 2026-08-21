import type { ComplexityTiers } from "./ComplexityRouterConfig";

export interface TierDraft {
  /** List identity: the React key and the fallback and plan-mode pointers' target. Never serialized. */
  id: string;
  name: string;
  /** The tier's rubric bullet. Blank on a built-in name inherits the built-in criteria. */
  definition: string;
  models: string[];
}

// The draft IS the wire list (severity order = tier_definitions order); absence means the
// built-in four-tier router and a payload identical to before this field existed.
export interface CustomTierSet {
  tiers: TierDraft[];
  fallback_tier_id: string;
}

const BUILT_IN_TIER_NAMES = Object.keys({
  SIMPLE: null,
  MEDIUM: null,
  COMPLEX: null,
  REASONING: null,
} satisfies Record<keyof ComplexityTiers, null>);

export const isBuiltInTierName = (name: string): boolean =>
  BUILT_IN_TIER_NAMES.some((tier) => tier.toLowerCase() === name.trim().toLowerCase());

export const activeTierNames = (customTierSet: CustomTierSet | undefined): string[] =>
  customTierSet ? customTierSet.tiers.map((tier) => tier.name.trim()).filter(Boolean) : [...BUILT_IN_TIER_NAMES];

// Per-row errors the backend cannot phrase; everything payload-level is the write gate's job,
// dry-run before every save. The Done button and both submit gates read this one guard.
export const getCustomTierRowsError = (customTierSet: CustomTierSet): string | null => {
  const rows = customTierSet.tiers;
  if (rows.some((row) => !row.name.trim())) return "Name every tier";
  if (rows.some((row) => !row.definition.trim() && !isBuiltInTierName(row.name)))
    return "Every custom tier needs a definition: it is the rubric the classifier routes on";
  if (rows.some((row) => row.models.length === 0)) return "Select at least one model for every tier";
  if (!rows.some((row) => row.id === customTierSet.fallback_tier_id))
    return "Pick a Fallback Tier for classifier failures";
  return null;
};

// Params are name-keyed; keys no live row owns are stale and drop.
export const scopeTierParamsToRows = <T>(
  params: Record<string, T> | undefined,
  rows: { name: string }[],
): Record<string, T> | undefined => {
  if (!params) return params;
  const names = new Set(rows.map((row) => row.name.trim()));
  const scoped = Object.fromEntries(Object.entries(params).filter(([key]) => names.has(key)));
  return Object.keys(scoped).length > 0 ? scoped : undefined;
};
