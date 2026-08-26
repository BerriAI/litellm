export type AutoRouterTestMode = "chat" | "embedding";

export interface AutoRouterTestTarget {
  labels: string[];
  modelGroup: string;
  mode: AutoRouterTestMode;
}

export interface BuildAutoRouterTestTargetsParams {
  /** Ordered [tier name, model groups] entries of the active tier set. */
  tiers: readonly (readonly [string, string[]])[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  /** The resolved default model - see resolveComplexityDefaultModel. A live fallback destination,
   * so it is probed even when no tier lists it. */
  defaultModel?: string;
}

export const buildAutoRouterTestTargets = ({
  tiers,
  semanticMatchingEnabled,
  embeddingModel,
  defaultModel,
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

  return [...tierTargets, ...embeddingTarget];
};
