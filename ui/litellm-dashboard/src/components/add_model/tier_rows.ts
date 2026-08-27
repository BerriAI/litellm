import type { ComplexityTiers } from "./ComplexityRouterConfig";
import type { ComplexityTier } from "./KeywordTierRules";

export const TIER_ORDER: ComplexityTier[] = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"];

export interface TierRow {
  id: string;
  name: string;
  models: string[];
}

export interface ActiveTierSet {
  tiers: ComplexityTiers;
}

export const activeTierName = (row: TierRow): string => row.name.trim();

export const sameTierIdentity = (left: string, right: string): boolean =>
  left.trim().toLowerCase() === right.trim().toLowerCase();

export const isBuiltInTierName = (name: string): boolean => TIER_ORDER.some((tier) => sameTierIdentity(tier, name));

// The only reader of the tier set. A row's id is the canonical tier key, so anything pointing into
// the set (the plan-mode floor, per-model params) points at a row rather than at a position.
export const activeTierRows = (value: ActiveTierSet): TierRow[] =>
  TIER_ORDER.map((tier) => ({ id: tier, name: tier, models: value.tiers[tier] ?? [] }));

export const tierRowById = (rows: readonly TierRow[], id: string | undefined): TierRow | undefined =>
  id === undefined ? undefined : rows.find((row) => row.id === id);

export const tierRowByName = (rows: readonly TierRow[], name: string): TierRow | undefined =>
  rows.find((row) => sameTierIdentity(row.name, name));

// Mirrors init_complexity_router_deployment (litellm/router.py): a pin wins, then MEDIUM or SIMPLE
// looked up by exact name.
export const resolveComplexityDefaultModel = (value: ActiveTierSet, pinned?: string): string | undefined => {
  const rows = activeTierRows(value);
  const named = (name: string) => rows.find((row) => activeTierName(row) === name)?.models[0];
  return pinned?.trim() || named("MEDIUM") || named("SIMPLE");
};
