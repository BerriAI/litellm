import { normalizeTierModels } from "./complexity_router_tiers";

export type AutoRouterTestMode = "chat" | "embedding";

export interface AutoRouterTestTarget {
  labels: string[];
  modelGroup: string;
  mode: AutoRouterTestMode;
  requestParams?: Record<string, unknown>;
}

export interface BuildAutoRouterTestTargetsParams {
  /** Ordered [tier name, model groups] entries of the active tier set. */
  tiers: readonly (readonly [string, string[]])[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  /** The resolved default model - see resolveComplexityDefaultModel. A live fallback destination,
   * so it is probed even when no tier lists it. */
  defaultModel?: string;
  classifier?: { model: string; reasoningEffort?: string };
}

export const buildAutoRouterTestTargets = ({
  tiers,
  semanticMatchingEnabled,
  embeddingModel,
  defaultModel,
  classifier,
}: BuildAutoRouterTestTargetsParams): AutoRouterTestTarget[] => {
  const tieredByModel = tiers.reduce<Record<string, string[]>>((acc, [tier, models]) => {
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

  const classifierModel = classifier?.model.trim();
  const classifierTarget: AutoRouterTestTarget[] = classifierModel
    ? [
        {
          labels: ["Classifier"],
          modelGroup: classifierModel,
          mode: "chat",
          ...(classifier?.reasoningEffort && { requestParams: { reasoning_effort: classifier.reasoningEffort } }),
        },
      ]
    : [];

  return [...tierTargets, ...embeddingTarget, ...classifierTarget];
};

interface ComplexityRouterTierConfig {
  tiers?: {
    SIMPLE?: unknown;
    MEDIUM?: unknown;
    COMPLEX?: unknown;
    REASONING?: unknown;
  };
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

  const tiers: [string, string[]][] =
    config.tiers && typeof config.tiers === "object"
      ? Object.entries(config.tiers).map(([tier, models]) => [tier, normalizeTierModels(models)])
      : [];

  // Mirrors init_complexity_router_deployment (litellm/router.py): litellm_params wins, otherwise
  // pure tier-derivation. complexity_router_config.default_model is a UI-only marker the backend
  // never reads — folding it in here could point Test Connection at a model the router never
  // calls (see PR #36615 discussion).
  const effectiveDefaultModel = modelData?.litellm_params?.complexity_router_default_model || undefined;

  const testTargetParams = {
    tiers,
    semanticMatchingEnabled: Boolean(config.semantic_keyword_matching),
    embeddingModel: config.embedding_model,
    defaultModel: effectiveDefaultModel,
  };
  return buildAutoRouterTestTargets(testTargetParams);
};
