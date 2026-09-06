import type { KeywordTierRule } from "./KeywordTierRules";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { pruneTierModelParams } from "./complexity_router_tiers";
import {
  type ActiveTierRow,
  type TierRow,
  TIER_ORDER,
  activeTierName,
  activeTierRows,
  rowParamsByTier,
  sameTierIdentity,
  tierRowById,
  tierRowByName,
} from "./tier_rows";

export type TierSetAction =
  | { kind: "models"; id: string; models: string[] }
  | { kind: "patch"; id: string; patch: Partial<Omit<TierRow, "id">> }
  | { kind: "add" }
  | { kind: "remove"; id: string }
  | { kind: "restore" };

/** What an action produces: the next config, and the keyword rules that followed their rows. */
export interface TierSetResult {
  value: ComplexityRouterConfigValue;
  keywordTierRules: readonly KeywordTierRule[];
}

// The sole tier-set writer: it owns where rows live and reconciles both row-id pointers, so a
// fallback re-points and a floor whose row is gone turns off in the same write.
const commitTierRows = (
  rows: TierRow[],
  fallbackTierId: string,
  base: ComplexityRouterConfigValue,
): ComplexityRouterConfigValue => {
  const floorGone = base.plan_mode_min_tier !== undefined && !rows.some((row) => row.id === base.plan_mode_min_tier);
  const next = floorGone ? { ...base, plan_mode_min_tier: undefined } : base;
  if (!next.custom_tier_set) {
    return { ...next, tiers: { ...next.tiers, ...Object.fromEntries(rows.map((row) => [row.id, row.models])) } };
  }
  const fallback_tier_id = rows.some((row) => row.id === fallbackTierId)
    ? fallbackTierId
    : (tierRowByName(rows, "MEDIUM") ?? rows[0])?.id ?? "";
  return { ...next, custom_tier_set: { tiers: rows, fallback_tier_id } };
};

const asCustomBase = (base: ComplexityRouterConfigValue): ComplexityRouterConfigValue =>
  base.custom_tier_set
    ? base
    : { ...base, custom_tier_set: { tiers: activeTierRows(base), fallback_tier_id: "MEDIUM" } };

// A rule follows the row holding its name through any action, so neither a rename nor a restore can
// orphan it. A rule whose name is ambiguous across rows, or whose row the action deleted, stays put
// for the save gate to name loudly rather than being dropped or rewritten by guess.
const followedRuleTier = (before: readonly TierRow[], after: readonly TierRow[], tier: string): string | undefined => {
  const holders = before.filter((row) => sameTierIdentity(row.name, tier));
  if (holders.length !== 1 || activeTierName(holders[0]) !== tier) return undefined;
  const target = tierRowById(after, holders[0].id);
  return target === undefined ? undefined : activeTierName(target);
};

const rulesFollowingRows = (
  before: readonly TierRow[],
  after: readonly TierRow[],
  rules: readonly KeywordTierRule[],
): readonly KeywordTierRule[] => {
  const followed = rules.map((rule) => {
    const tier = followedRuleTier(before, after, rule.tier);
    return tier === undefined || tier === rule.tier ? rule : { ...rule, tier };
  });
  return followed.every((rule, index) => rule === rules[index]) ? rules : followed;
};

// Models and params both come from these rows, so the two cannot be keyed differently.
const exitToBuiltInTiers = (value: ComplexityRouterConfigValue, rows: readonly ActiveTierRow[]) => {
  const { custom_tier_set: _dropped, ...rest } = value;
  const builtInRows: ActiveTierRow[] = TIER_ORDER.map(
    (tier) =>
      tierRowById(rows, tier) ?? {
        id: tier,
        name: tier,
        definition: "",
        models: value.tiers[tier],
        params: value.tier_model_params?.[tier] ?? {},
      },
  );
  const restored: ComplexityRouterConfigValue = {
    ...rest,
    tier_model_params: rowParamsByTier(builtInRows),
    tiers: { ...value.tiers, ...Object.fromEntries(builtInRows.map((row) => [row.id, row.models])) },
  };
  return commitTierRows(activeTierRows(restored), "", restored);
};

const nextTierSetValue = (
  value: ComplexityRouterConfigValue,
  rows: ActiveTierRow[],
  action: TierSetAction,
): ComplexityRouterConfigValue => {
  const fallbackId = value.custom_tier_set?.fallback_tier_id ?? "MEDIUM";

  switch (action.kind) {
    case "models":
      return commitTierRows(
        rows.map((row) => (row.id === action.id ? { ...row, models: action.models } : row)),
        fallbackId,
        { ...value, tier_model_params: pruneTierModelParams(value.tier_model_params, action.id, action.models) },
      );
    case "patch":
      return commitTierRows(
        rows.map((row) => (row.id === action.id ? { ...row, ...action.patch } : row)),
        fallbackId,
        asCustomBase(value),
      );
    case "add":
      return commitTierRows(
        [...rows, { id: crypto.randomUUID(), name: "", definition: "", models: [] }],
        fallbackId,
        asCustomBase(value),
      );
    case "remove": {
      const removed = tierRowById(rows, action.id);
      const snapshot =
        removed && (TIER_ORDER as string[]).includes(action.id)
          ? { ...value, tiers: { ...value.tiers, [action.id]: removed.models } }
          : value;
      return commitTierRows(
        rows.filter((row) => row.id !== action.id),
        fallbackId,
        asCustomBase(snapshot),
      );
    }
    case "restore":
      return exitToBuiltInTiers(value, rows);
  }
};

export const applyTierSetAction = (
  value: ComplexityRouterConfigValue,
  keywordTierRules: readonly KeywordTierRule[],
  action: TierSetAction,
): TierSetResult => {
  const rows = activeTierRows(value);
  const next = nextTierSetValue(value, rows, action);
  return { value: next, keywordTierRules: rulesFollowingRows(rows, activeTierRows(next), keywordTierRules) };
};

/** The fallback-tier select is the only writer that re-points rather than reconciling. */
export const setFallbackTier = (value: ComplexityRouterConfigValue, id: string): ComplexityRouterConfigValue =>
  commitTierRows(activeTierRows(value), id, value);
