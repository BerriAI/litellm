import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export type SetupMode = "auto" | "manual";

export interface RecommendationStatus {
  setupMode: SetupMode;
  recommendationEnabled: boolean;
  recommendationPending: boolean;
  recommendationFailed: boolean;
  hasRecommendation: boolean;
}

export const getRecommendationBlockedReason = ({
  setupMode,
  recommendationEnabled,
  recommendationPending,
  recommendationFailed,
  hasRecommendation,
}: RecommendationStatus): string | null => {
  if (setupMode !== "auto") return null;
  if (!recommendationEnabled) return "Select a team so Auto can use that team's available models";
  if (recommendationPending && !hasRecommendation) return "Building the best setup for your available models";
  if (recommendationFailed && !hasRecommendation) {
    return "Auto setup could not build a router for your available models";
  }
  return null;
};

const autoSetupPoolShape = (value: ComplexityRouterConfigValue): string =>
  JSON.stringify({
    tiers: value.tiers,
    customTierSet: value.custom_tier_set,
    tierModelParams: value.tier_model_params,
  });

export const detachAutoSetupAfterPoolEdit = (
  previous: ComplexityRouterConfigValue,
  next: ComplexityRouterConfigValue,
): ComplexityRouterConfigValue =>
  previous.auto_setup && autoSetupPoolShape(previous) !== autoSetupPoolShape(next)
    ? { ...next, auto_setup: undefined }
    : next;
