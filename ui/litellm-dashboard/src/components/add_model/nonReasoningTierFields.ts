import type { ClassifierType, ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

/** The NON_REASONING keys a classifier switch carries forward, or clears for a classifier that
 * cannot emit the tier. Leaving them set there is a config the backend refuses on save. */
export const nonReasoningTierFields = (
  classifierType: ClassifierType,
  value: ComplexityRouterConfigValue,
): Pick<ComplexityRouterConfigValue, "enable_non_reasoning_tier" | "tiers"> => {
  if (classifierType === "llm") {
    return { enable_non_reasoning_tier: value.enable_non_reasoning_tier, tiers: value.tiers };
  }
  const { NON_REASONING: _cleared, ...keptTiers } = value.tiers;
  return { enable_non_reasoning_tier: undefined, tiers: keptTiers };
};
