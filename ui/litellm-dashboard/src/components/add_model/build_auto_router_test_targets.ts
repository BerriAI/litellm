import { ComplexityTiers } from "./ComplexityRouterConfig";

export type AutoRouterTestMode = "chat" | "embedding";

export interface AutoRouterTestTarget {
  labels: string[];
  modelGroup: string;
  mode: AutoRouterTestMode;
}

export interface BuildAutoRouterTestTargetsParams {
  tiers: ComplexityTiers;
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
}

const TIER_ORDER: (keyof ComplexityTiers)[] = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"];

export const buildAutoRouterTestTargets = ({
  tiers,
  semanticMatchingEnabled,
  embeddingModel,
}: BuildAutoRouterTestTargetsParams): AutoRouterTestTarget[] => {
  const groupedByModel = TIER_ORDER.reduce<Record<string, string[]>>((acc, tier) => {
    const modelGroup = tiers[tier]?.trim();
    if (!modelGroup) return acc;
    return { ...acc, [modelGroup]: [...(acc[modelGroup] ?? []), tier] };
  }, {});

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
