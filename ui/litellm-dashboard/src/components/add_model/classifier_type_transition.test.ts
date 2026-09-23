import { describe, expect, it } from "vitest";
import { type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { transitionClassifierType } from "./classifier_type_transition";

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
      adaptive: true,
    };
    const jev = transitionClassifierType(initial, "jev");
    const expectedJev = {
      classifier_type: "jev",
      classifier_context_window_size: 8,
      classifier_context_budget_chars: 16000,
      classifier_context_include_assistant_turns: true,
      classifier_fallback: "default_model",
      adaptive: true,
      tiers: initial.tiers,
    };
    expect(jev).toMatchObject(expectedJev);
    expect(jev.jev_classifier_config).toBeDefined();
    expect(jev.classifier_llm_config).toBeUndefined();
    expect(jev.classification_prompt).toBeUndefined();
    const llm = transitionClassifierType(jev, "llm");
    expect(llm.jev_classifier_config).toBeUndefined();
    expect(llm.classifier_llm_config).toMatchObject({ model: "" });
    expect(llm.classifier_context_window_size).toBe(8);
  });

  it("clears classifier settings when switching to the heuristic", () => {
    const result = transitionClassifierType(standard, "heuristic");
    expect(result.classifier_llm_config).toBeUndefined();
    expect(result.jev_classifier_config).toBeUndefined();
    expect(result.classifier_context_window_size).toBeUndefined();
    expect(result.classifier_context_budget_chars).toBeUndefined();
    expect(result.classifier_context_include_assistant_turns).toBeUndefined();
    expect(result.classifier_fallback).toBeUndefined();
    expect(result.tiers.SIMPLE).toEqual(["efficient"]);
  });
});
