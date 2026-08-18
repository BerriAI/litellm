import { customTierDefaultModel, normalizeTierModels, resolveComplexityDefaultModel } from "./complexity_router_tiers";
import { hydrateCustomTierSet } from "./build_complexity_router_config";
import { ComplexityTiers } from "./ComplexityRouterConfig";

export type AutoRouterTestMode = "chat" | "embedding";

export interface AutoRouterTestTarget {
  labels: string[];
  modelGroup: string;
  mode: AutoRouterTestMode;
}

export interface BuildAutoRouterTestTargetsParams {
  /** With a custom tier set, pass the EFFECTIVE record (removed built-ins emptied) - see effectiveComplexityTiers. */
  tiers: ComplexityTiers;
  additionalTiers?: { name: string; models: string[] }[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  /** The resolved default model - see resolveComplexityDefaultModel. A live fallback destination,
   * so it is probed even when no tier lists it. */
  defaultModel?: string;
}

// Keys drive iteration order; `satisfies Record<keyof ComplexityTiers, null>` makes it a
// compile error to add a tier to ComplexityTiers without listing it here (and vice versa).
const TIER_ORDER = Object.keys({
  SIMPLE: null,
  MEDIUM: null,
  COMPLEX: null,
  REASONING: null,
} satisfies Record<keyof ComplexityTiers, null>) as (keyof ComplexityTiers)[];

export const buildAutoRouterTestTargets = ({
  tiers,
  additionalTiers = [],
  semanticMatchingEnabled,
  embeddingModel,
  defaultModel,
}: BuildAutoRouterTestTargetsParams): AutoRouterTestTarget[] => {
  const tierPools: [string, string[]][] = [
    ...TIER_ORDER.map((tier): [string, string[]] => [tier, tiers[tier] ?? []]),
    ...additionalTiers.map((tier): [string, string[]] => [tier.name, tier.models]),
  ];
  const tieredByModel = tierPools.reduce<Record<string, string[]>>((acc, [tier, models]) => {
    return models.reduce((tierAcc, rawModel) => {
      const modelGroup = rawModel?.trim();
      if (!modelGroup) return tierAcc;
      return { ...tierAcc, [modelGroup]: [...(tierAcc[modelGroup] ?? []), tier] };
    }, acc);
  }, {});

  // The default is a live destination whenever the chosen tier has no model, and when an LLM
  // classifier fails with "Route to the default model", so a green test that skipped it would be
  // reporting on a router it had not fully reached.
  const resolvedDefault = defaultModel?.trim();
  const groupedByModel =
    resolvedDefault && !(resolvedDefault in tieredByModel)
      ? { ...tieredByModel, [resolvedDefault]: ["Default"] }
      : tieredByModel;

  const tierTargets: AutoRouterTestTarget[] = Object.entries(groupedByModel).map(([modelGroup, labels]) => ({
    labels,
    modelGroup,
    mode: "chat" as const,
  }));

  const embeddingTarget: AutoRouterTestTarget[] =
    semanticMatchingEnabled && embeddingModel?.trim()
      ? [{ labels: ["Embedding"], modelGroup: embeddingModel.trim(), mode: "embedding" as const }]
      : [];

  return [...tierTargets, ...embeddingTarget];
};

interface ComplexityRouterTierConfig {
  tiers?: {
    SIMPLE?: unknown;
    MEDIUM?: unknown;
    COMPLEX?: unknown;
    REASONING?: unknown;
  };
  tier_definitions?: unknown;
  fallback_tier?: unknown;
  semantic_keyword_matching?: boolean;
  embedding_model?: string;
  default_model?: string;
}

interface ComplexityRouterModelData {
  litellm_params?: {
    complexity_router_config?: ComplexityRouterTierConfig | string;
    complexity_router_default_model?: string;
  };
}

export const buildComplexityRouterTestTargets = (
  modelData: ComplexityRouterModelData | null | undefined,
): AutoRouterTestTarget[] => {
  const rawConfig = modelData?.litellm_params?.complexity_router_config;
  let config: ComplexityRouterTierConfig = {};
  if (typeof rawConfig === "string") {
    try {
      config = JSON.parse(rawConfig);
    } catch {
      config = {};
    }
  } else if (rawConfig) {
    config = rawConfig;
  }

  const rawTiers = {
    SIMPLE: normalizeTierModels(config.tiers?.SIMPLE),
    MEDIUM: normalizeTierModels(config.tiers?.MEDIUM),
    COMPLEX: normalizeTierModels(config.tiers?.COMPLEX),
    REASONING: normalizeTierModels(config.tiers?.REASONING),
  };
  const customTierSet = hydrateCustomTierSet(config);
  const tiers = customTierSet ? { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] } : rawTiers;

  // Mirrors init_complexity_router_deployment (litellm/router.py): litellm_params wins, otherwise
  // pure tier-derivation. complexity_router_config.default_model is a UI-only marker the backend
  // never reads — folding it in here could point Test Connection at a model the router never
  // calls (see PR #36615 discussion).
  const effectiveDefaultModel = modelData?.litellm_params?.complexity_router_default_model || undefined;

  const testTargetParams = {
    tiers,
    additionalTiers: customTierSet?.tiers.map((row) => ({ name: row.name, models: row.models })),
    semanticMatchingEnabled: Boolean(config.semantic_keyword_matching),
    embeddingModel: config.embedding_model,
    defaultModel: customTierSet
      ? customTierDefaultModel(customTierSet, effectiveDefaultModel)
      : resolveComplexityDefaultModel(tiers, effectiveDefaultModel),
  };
  return buildAutoRouterTestTargets(testTargetParams);
};
