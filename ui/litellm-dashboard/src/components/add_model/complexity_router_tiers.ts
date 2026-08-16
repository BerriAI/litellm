import type { ComplexityTiers } from "./ComplexityRouterConfig";
import type { ComplexityTier } from "./KeywordTierRules";

/**
 * A complexity tier maps to `str | list[str] | object | list[object]` on the backend
 * (litellm/router_strategy/complexity_router/config.py: "string = pin; list = random pick"),
 * and the router widens the bare string with `models if isinstance(models, list) else [models]`.
 *
 * Every UI reader of a STORED complexity_router_config must widen the same way, so this is the
 * single owner of that rule. Readers of in-memory ComplexityTiers state are already string[]
 * and do not need it.
 */
export const normalizeTierModels = (value: unknown): string[] => {
  const entries = Array.isArray(value) ? value : [value];
  return entries.flatMap((entry) => {
    if (typeof entry === "string" && entry) return [entry];
    if (
      typeof entry === "object" &&
      entry !== null &&
      !Array.isArray(entry) &&
      typeof (entry as { model_name?: unknown }).model_name === "string"
    ) {
      return [(entry as { model_name: string }).model_name];
    }
    return [];
  });
};

/**
 * Mirrors `init_complexity_router_deployment` (litellm/router.py): an explicit pin wins, otherwise
 * the default is `MEDIUM or SIMPLE`. Deriving past SIMPLE would name a model the backend never
 * picks, and it raises rather than falling through to COMPLEX/REASONING.
 */
export const resolveComplexityDefaultModel = (tiers: ComplexityTiers, pinned?: string): string | undefined =>
  pinned?.trim() || tiers.MEDIUM[0] || tiers.SIMPLE[0];

export const DEFAULT_TIER_LABELS: Record<ComplexityTier, string> = {
  SIMPLE: "Simple",
  MEDIUM: "Medium",
  COMPLEX: "Complex",
  REASONING: "Reasoning",
};

export const TIER_ORDER: ComplexityTier[] = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"];

export const tierOptions = (
  tierLabels: Partial<Record<ComplexityTier, string>> | undefined,
): { value: ComplexityTier; label: string }[] =>
  TIER_ORDER.map((tier) => ({ value: tier, label: tierLabels?.[tier]?.trim() || DEFAULT_TIER_LABELS[tier] }));

export type TierModelParams = Record<string, unknown>;
export type SerializedTierModel = string | { model_name: string; litellm_params: TierModelParams };

export type TierModelParamsByTier = Partial<Record<keyof ComplexityTiers, Record<string, TierModelParams>>>;

export const extractTierModelParams = (value: unknown): Record<string, TierModelParams> => {
  const entries = Array.isArray(value) ? value : [value];
  return Object.fromEntries(
    entries.flatMap((entry) => {
      if (
        typeof entry !== "object" ||
        entry === null ||
        Array.isArray(entry) ||
        typeof (entry as { model_name?: unknown }).model_name !== "string"
      ) {
        return [];
      }
      const params = (entry as { litellm_params?: unknown }).litellm_params;
      if (typeof params !== "object" || params === null || Array.isArray(params)) return [];
      return [[(entry as { model_name: string }).model_name, params as TierModelParams] as const];
    }),
  );
};

export const serializeTierModels = (
  models: string[],
  paramsByModel: Record<string, TierModelParams> | undefined,
): SerializedTierModel[] => {
  const entries = models.map((model) => {
    const params = paramsByModel?.[model];
    return params && Object.keys(params).length > 0 ? { model_name: model, litellm_params: params } : model;
  });
  return entries;
};

export type SerializedTierConfig = Partial<Record<keyof ComplexityTiers, SerializedTierModel[]>>;

export const serializeTierConfig = (
  tiers: Partial<ComplexityTiers>,
  paramsByTier: TierModelParamsByTier | undefined,
): SerializedTierConfig =>
  Object.fromEntries(
    Object.entries(tiers).map(([tier, models]) => [
      tier,
      serializeTierModels(models ?? [], paramsByTier?.[tier as keyof ComplexityTiers]),
    ]),
  ) as SerializedTierConfig;
