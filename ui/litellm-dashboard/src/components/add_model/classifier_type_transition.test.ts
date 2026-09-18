import { describe, expect, it } from "vitest";
import { effectiveClassifierType, type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { transitionClassifierType } from "./classifier_type_transition";
import { applyTierSetAction } from "./tier_set_actions";

const standard: ComplexityRouterConfigValue = {
  classifier_type: "llm",
  classifier_llm_config: { model: "judge", timeout_ms: 20000, classification_rubric: "business" },
  classifier_context_window_size: 8,
  classifier_context_budget_chars: 16000,
  classifier_context_include_assistant_turns: true,
  classifier_fallback: "default_model",
  tiers: { SIMPLE: ["efficient"], MEDIUM: ["middle"], COMPLEX: [], REASONING: ["capable"] },
};

describe("transitionClassifierType", () => {
  it("switches between LLM and JEV without losing shared routing settings or leaking opposite config", () => {
    const initial = {
      ...standard,
      classification_prompt: "LLM only",
      classification_examples: "LLM examples",
      enable_non_reasoning_tier: true,
      tiers: { ...standard.tiers, NON_REASONING: ["fast"] },
      plan_mode_min_tier: "NON_REASONING",
      adaptive: true,
    };
    const jev = transitionClassifierType(initial, "jev");
    expect(jev).toMatchObject({
      classifier_type: "jev",
      jev_classifier_config: { model: "jev-latest", timeout_ms: 3000 },
      classifier_context_window_size: 8,
      classifier_context_budget_chars: 16000,
      classifier_context_include_assistant_turns: true,
      classifier_fallback: "default_model",
      adaptive: true,
      enable_non_reasoning_tier: true,
      plan_mode_min_tier: "NON_REASONING",
      tiers: initial.tiers,
    });
    expect(jev.classifier_llm_config).toBeUndefined();
    expect(jev.classification_prompt).toBeUndefined();
    expect(jev.classification_examples).toBeUndefined();
    const custom = applyTierSetAction(jev, [], { kind: "patch", id: "SIMPLE", patch: { name: "QUICK" } }).value;
    expect(effectiveClassifierType(custom)).toBe("jev");
    const restored = applyTierSetAction(custom, [], { kind: "restore" }).value;
    expect(effectiveClassifierType(restored)).toBe("jev");
    expect(restored.jev_classifier_config).toEqual(jev.jev_classifier_config);
    const llm = transitionClassifierType(custom, "llm");
    expect(llm.jev_classifier_config).toBeUndefined();
    expect(llm.classifier_llm_config).toMatchObject({ model: "" });
    expect(llm.custom_tier_set).toEqual(custom.custom_tier_set);
    expect(llm.classifier_context_window_size).toBe(8);
  });

  it.each(["heuristic_first", "hybrid"] as const)("keeps existing LLM settings when switching to %s", (target) => {
    const result = transitionClassifierType(standard, target);
    const expectedSettings = {
      classifier_type: target,
      classifier_llm_config: standard.classifier_llm_config,
      classifier_context_window_size: 8,
      classifier_context_budget_chars: 16000,
      classifier_context_include_assistant_turns: true,
      classifier_fallback: "default_model",
    };
    expect(result).toMatchObject(expectedSettings);
  });

  it.each(["capability", "llm_v2"] as const)("requires explicit policy input for a new %s classifier", (target) => {
    const result = transitionClassifierType(standard, target);
    expect(result.classifier_llm_config).toEqual({ model: "judge", timeout_ms: 20000 });
    expect(result.classifier_fallback).toBeUndefined();
    if (target === "capability") {
      expect(result.capability_classifier_config?.base_threshold).toBeNaN();
    } else {
      expect(result.llm_v2_config).toMatchObject({ efficient_profile: "", capable_profile: "", harness: "" });
      expect(result.llm_v2_config?.max_quality_gap).toBeNaN();
    }
    expect(standard.tiers.MEDIUM).toEqual(["middle"]);
    expect(standard.classifier_llm_config?.classification_rubric).toBe("business");
  });

  it.each([
    ["capability", "llm"],
    ["capability", "heuristic_first"],
    ["capability", "hybrid"],
    ["llm_v2", "llm"],
    ["llm_v2", "heuristic_first"],
    ["llm_v2", "hybrid"],
  ] as const)("restores the complexity rubric from %s to %s while preserving the judge", (source, target) => {
    const forecast = transitionClassifierType(standard, source);
    const result = transitionClassifierType(forecast, target);
    expect(result.classifier_llm_config).toEqual({
      model: "judge",
      timeout_ms: 20000,
      classification_rubric: "agentic",
    });
    expect(result.capability_classifier_config).toBeUndefined();
    expect(result.llm_v2_config).toBeUndefined();
  });

  it("clears the inactive non-reasoning pool and plan floor when switching to local classification", () => {
    const initial: ComplexityRouterConfigValue = {
      ...standard,
      tiers: { ...standard.tiers, NON_REASONING: ["chat"] },
      enable_non_reasoning_tier: true,
      plan_mode_min_tier: "NON_REASONING",
    };
    const result = transitionClassifierType(initial, "heuristic");
    expect(result.classifier_llm_config).toBeUndefined();
    expect(result.classifier_context_window_size).toBeUndefined();
    expect(result.classifier_context_budget_chars).toBeUndefined();
    expect(result.classifier_context_include_assistant_turns).toBeUndefined();
    expect(result.classifier_fallback).toBeUndefined();
    expect(result.tiers.NON_REASONING).toBeUndefined();
    expect(result.enable_non_reasoning_tier).toBeUndefined();
    expect(result.plan_mode_min_tier).toBeUndefined();
    expect(result.tiers.SIMPLE).toEqual(["efficient"]);
  });
});
