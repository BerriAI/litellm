import type { ClassifierType, ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const NON_REASONING = "NON_REASONING";

/** The NON_REASONING keys a classifier switch carries forward, or clears for a classifier that
 * cannot emit the tier. Leaving them set there is a config the backend refuses on save. The floor
 * goes with them: it is rejected on save while it names an inactive tier, and the switch is
 * disabled once the classifier changes, so the operator could not clear it themselves.
 * An orphaned keyword rule is left for getKeywordTierRulesError to name, matching how a removed
 * custom tier already behaves. */
export const nonReasoningTierFields = (
  classifierType: ClassifierType,
  value: ComplexityRouterConfigValue,
): Pick<ComplexityRouterConfigValue, "enable_non_reasoning_tier" | "tiers" | "plan_mode_min_tier"> => {
  if (classifierType === "llm") {
    return {
      enable_non_reasoning_tier: value.enable_non_reasoning_tier,
      tiers: value.tiers,
      plan_mode_min_tier: value.plan_mode_min_tier,
    };
  }
  const { [NON_REASONING]: _cleared, ...tiers } = value.tiers;
  return {
    enable_non_reasoning_tier: undefined,
    tiers,
    plan_mode_min_tier: value.plan_mode_min_tier === NON_REASONING ? undefined : value.plan_mode_min_tier,
  };
};
