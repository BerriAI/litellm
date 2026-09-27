import { describe, expect, it } from "vitest";

import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { nonReasoningTierFields } from "./nonReasoningTierFields";

const enabledValue: ComplexityRouterConfigValue = {
  classifier_type: "llm",
  enable_non_reasoning_tier: true,
  tiers: {
    NON_REASONING: ["relay-cheap"],
    SIMPLE: ["gpt-4o-mini"],
    MEDIUM: ["gpt-4o"],
    COMPLEX: ["sonnet"],
    REASONING: ["opus"],
  },
};

describe("nonReasoningTierFields", () => {
  it("keeps the tier and its pool while the classifier stays LLM", () => {
    expect(nonReasoningTierFields("llm", enabledValue)).toEqual({
      enable_non_reasoning_tier: true,
      tiers: enabledValue.tiers,
    });
  });

  it.each(["heuristic", "heuristic_v2", "heuristic_first", "hybrid"] as const)(
    "clears the flag and the tier when the classifier becomes %s",
    (classifierType) => {
      // Leaving the flag set under a classifier that cannot emit the tier is a config the backend
      // refuses, and the switch is disabled there, so the operator could never undo it.
      const cleared = nonReasoningTierFields(classifierType, enabledValue);
      expect(cleared.enable_non_reasoning_tier).toBeUndefined();
      expect(cleared.tiers).not.toHaveProperty("NON_REASONING");
    },
  );

  it("leaves the other tiers untouched when it clears", () => {
    const { NON_REASONING: _dropped, ...expectedTiers } = enabledValue.tiers;
    expect(nonReasoningTierFields("heuristic", enabledValue).tiers).toEqual(expectedTiers);
  });

  it("is a no-op for a router that never enabled the tier", () => {
    const fourTier: ComplexityRouterConfigValue = {
      classifier_type: "heuristic",
      tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: ["sonnet"], REASONING: ["opus"] },
    };
    expect(nonReasoningTierFields("heuristic", fourTier)).toEqual({
      enable_non_reasoning_tier: undefined,
      tiers: fourTier.tiers,
    });
  });
});

describe("stale references to the cleared tier", () => {
  const withFloorOnTierZero: ComplexityRouterConfigValue = { ...enabledValue, plan_mode_min_tier: "NON_REASONING" };

  it("releases a plan-mode floor pointing at the tier it just cleared", () => {
    // The backend rejects a floor naming an inactive tier, and the switch is disabled once the
    // classifier changes, so a floor left behind is a config the operator cannot save or undo.
    expect(nonReasoningTierFields("heuristic", withFloorOnTierZero).plan_mode_min_tier).toBeUndefined();
  });

  it("leaves a floor on another tier alone", () => {
    const floorOnComplex: ComplexityRouterConfigValue = { ...enabledValue, plan_mode_min_tier: "COMPLEX" };
    expect(nonReasoningTierFields("heuristic", floorOnComplex).plan_mode_min_tier).toBe("COMPLEX");
  });

  it("keeps the floor while the classifier can still emit the tier", () => {
    expect(nonReasoningTierFields("llm", withFloorOnTierZero).plan_mode_min_tier).toBe("NON_REASONING");
  });
});
