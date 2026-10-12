import { describe, expect, it } from "vitest";
import { effectiveClassifierType, type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { transitionClassifierType } from "./classifier_type_transition";
import { applyTierSetAction } from "./tier_set_actions";
import { buildUpdatedComplexityRouterConfig } from "../edit_auto_router/edit_auto_router_modal";

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
    const expectedJevConfig = {
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
    };
    expect(jev).toMatchObject(expectedJevConfig);
    expect(buildUpdatedComplexityRouterConfig({}, jev).classifier_llm_config).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, jev).classification_prompt).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, jev).classification_examples).toBeUndefined();
    const custom = applyTierSetAction(jev, [], { kind: "patch", id: "SIMPLE", patch: { name: "QUICK" } }).value;
    expect(effectiveClassifierType(custom)).toBe("jev");
    const restored = applyTierSetAction(custom, [], { kind: "restore" }).value;
    expect(effectiveClassifierType(restored)).toBe("jev");
    expect(restored.jev_classifier_config).toEqual(jev.jev_classifier_config);
    const llm = transitionClassifierType(custom, "llm");
    expect(buildUpdatedComplexityRouterConfig({}, llm).opensource_classifier_config).toBeUndefined();
    expect(llm.classifier_llm_config).toEqual(standard.classifier_llm_config);
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
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_llm_config).toEqual({
      model: "judge",
      timeout_ms: 20000,
    });
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_fallback).toBeUndefined();
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
      classification_rubric: "business",
    });
    expect(buildUpdatedComplexityRouterConfig({}, result).capability_classifier_config).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, result).llm_v2_config).toBeUndefined();
  });

  it("clears the inactive non-reasoning pool and plan floor when switching to local classification", () => {
    const initial: ComplexityRouterConfigValue = {
      ...standard,
      tiers: { ...standard.tiers, NON_REASONING: ["chat"] },
      enable_non_reasoning_tier: true,
      plan_mode_min_tier: "NON_REASONING",
    };
    const result = transitionClassifierType(initial, "heuristic");
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_llm_config).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_context_window_size).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_context_budget_chars).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_context_include_assistant_turns).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, result).classifier_fallback).toBeUndefined();
    expect(result.tiers.NON_REASONING).toBeUndefined();
    expect(result.enable_non_reasoning_tier).toBeUndefined();
    expect(result.plan_mode_min_tier).toBeUndefined();
    expect(result.tiers.SIMPLE).toEqual(["efficient"]);
  });
});

describe("classifier draft preservation", () => {
  it("keeps the decision deployment and LLM tuning while serializing only the active classifier", () => {
    const initial = {
      ...standard,
      classifier_type: "jev" as const,
      jev_classifier_config: { deployment_name: "decision", model: "jev-latest", timeout_ms: 3000 },
    };
    const judge = transitionClassifierType(initial, "llm");
    const heuristic = transitionClassifierType(judge, "heuristic");
    const decision = transitionClassifierType(heuristic, "jev");
    expect(decision.jev_classifier_config).toEqual(initial.jev_classifier_config);
    expect(transitionClassifierType(decision, "llm").classifier_llm_config).toEqual(standard.classifier_llm_config);
    expect(decision.classifier_context_window_size).toBe(standard.classifier_context_window_size);
    expect(buildUpdatedComplexityRouterConfig({}, decision).classifier_llm_config).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, heuristic).opensource_classifier_config).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, heuristic).classifier_llm_config).toBeUndefined();
    expect(buildUpdatedComplexityRouterConfig({}, judge).opensource_classifier_config).toBeUndefined();
  });
});

it.each(["heuristic", "heuristic_v2", "llm", "jev", "capability", "llm_v2"] as const)(
  "preserves classifier tuning through %s and excludes inactive settings from save",
  (target) => {
    const initial: ComplexityRouterConfigValue = {
      ...standard,
      heuristic_v2_success_threshold: 0.65,
      capability_classifier_config: { efficient_tier: "SIMPLE", capable_tier: "REASONING", base_threshold: 0.8 },
      llm_v2_config: { efficient_profile: "fast", capable_profile: "thorough", harness: "tools", max_quality_gap: 0.1 },
    };
    const selected = transitionClassifierType(initial, target);
    const restored = transitionClassifierType(selected, "llm");
    expect(restored.classifier_llm_config).toEqual(initial.classifier_llm_config);
    expect(restored.classifier_context_window_size).toBe(initial.classifier_context_window_size);
    expect(restored.heuristic_v2_success_threshold).toBe(initial.heuristic_v2_success_threshold);
    expect(restored.capability_classifier_config).toEqual(initial.capability_classifier_config);
    expect(restored.llm_v2_config).toEqual(initial.llm_v2_config);
    const saved = buildUpdatedComplexityRouterConfig({}, restored);
    expect(saved.capability_classifier_config).toBeUndefined();
    expect(saved.llm_v2_config).toBeUndefined();
    expect(saved.opensource_classifier_config).toBeUndefined();
    expect(saved.previous_classifier_types).toBeUndefined();
  },
);

it.each([
  ["capability", "llm_v2"],
  ["llm_v2", "capability"],
] as const)("keeps the solver pair and policy when switching from %s to %s and back", (source, target) => {
  const initial: ComplexityRouterConfigValue = {
    ...standard,
    classifier_type: source,
    tiers: { SIMPLE: [], MEDIUM: ["efficient"], COMPLEX: ["capable"], REASONING: [] },
    capability_classifier_config: { efficient_tier: "MEDIUM", capable_tier: "COMPLEX", base_threshold: 0.8 },
    llm_v2_config: {
      efficient_tier: "MEDIUM",
      capable_tier: "COMPLEX",
      efficient_profile: "fast",
      capable_profile: "thorough",
      harness: "tools",
      max_quality_gap: 0.1,
    },
  };
  const selected = transitionClassifierType(initial, target);
  const restored = transitionClassifierType(selected, source);
  expect(restored.tiers).toEqual(initial.tiers);
  expect(restored.capability_classifier_config).toEqual(initial.capability_classifier_config);
  expect(restored.llm_v2_config).toEqual(initial.llm_v2_config);
});
