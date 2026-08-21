export type AutoRouterTestMode = "chat" | "embedding";

export interface AutoRouterTestTarget {
  labels: string[];
  modelGroup: string;
  mode: AutoRouterTestMode;
}

export interface BuildAutoRouterTestTargetsParams {
  /** Ordered [tier name, model groups] entries of the active tier set. */
  tiers: [string, string[]][];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
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

  // Classifier failures land on the default, so it is probed even when no tier lists it.
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
