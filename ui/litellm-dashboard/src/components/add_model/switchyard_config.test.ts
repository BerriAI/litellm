import { describe, expect, it } from "vitest";
import {
  emptySwitchyardConfig,
  isDecisionModelMode,
  parseToolSemantics,
  setClassifierKind,
  stageConfidence,
  switchyardSaveError,
  switchyardConfigSchema,
  switchyardRouterPayload,
  toggleClassifier,
} from "./switchyard_config";

describe("Switchyard configuration", () => {
  it("changes the confidence default when the classifier is toggled", () => {
    const config = emptySwitchyardConfig();
    expect(stageConfidence(config)).toBe(0.5);
    expect(stageConfidence(toggleClassifier(config, true))).toBe(0.7);
    expect(stageConfidence(toggleClassifier(toggleClassifier(config, true), false))).toBe(0.5);
  });
  it("preserves an admin's confidence choice across toggles and save/reload", () => {
    const config = {
      ...emptySwitchyardConfig(),
      models: { capable: "strong", efficient: "fast" },
      options: { picker: "efficient_first" as const, confidence_threshold: 0.64 },
    };
    const enabled = { ...toggleClassifier(config, true), models: { ...config.models, judge: "judge" } };
    const saved = switchyardConfigSchema.parse(JSON.parse(JSON.stringify(enabled)));
    expect(stageConfidence(saved)).toBe(0.64);
    expect(stageConfidence(toggleClassifier(saved, false))).toBe(0.64);
  });
  it("requires a judge only with a classifier and removes it when disabled", () => {
    const config = { ...emptySwitchyardConfig(), models: { capable: "strong", efficient: "fast" } };
    expect(switchyardConfigSchema.safeParse(config).success).toBe(true);
    expect(switchyardConfigSchema.safeParse(toggleClassifier(config, true)).success).toBe(false);
    const enabled = { ...toggleClassifier(config, true), models: { ...config.models, judge: "judge" } };
    expect(switchyardConfigSchema.safeParse(enabled).success).toBe(true);
    expect(toggleClassifier(enabled, false).models).toEqual(config.models);
  });
  it.each(["escalation_note", "deescalation_note", "capable_system_prompt", "efficient_system_prompt"])(
    "rejects the unsupported rewrite option %s",
    (option) => {
      expect(
        switchyardConfigSchema.safeParse({
          models: { capable: "strong", efficient: "fast" },
          options: { [option]: "ignored" },
        }).success,
      ).toBe(false);
    },
  );
  it("rejects unsupported algorithms and settings the stage router judge ignores", () => {
    const models = { capable: "strong", efficient: "fast", judge: "judge" };
    expect(switchyardConfigSchema.safeParse({ ...emptySwitchyardConfig(), algorithm: "random" }).success).toBe(false);
    expect(switchyardConfigSchema.safeParse({ models, classifier: { session_affinity: true } }).success).toBe(false);
  });
  it("reads a stored classifier without a kind as an LLM judge", () => {
    const parsed = switchyardConfigSchema.parse({
      models: { capable: "strong", efficient: "fast", judge: "judge" },
      classifier: { base_threshold: 0.4 },
    });
    expect(parsed.classifier).toMatchObject({ kind: "llm", base_threshold: 0.4 });
  });
  it("opens routers saved through the API with nulls and structured decision instructions", () => {
    const instructions = { system: "Prefer the capable model for schema migrations" };
    const unsetOptions = {
      picker: "efficient_first",
      confidence_threshold: null,
      recent_window: null,
      tool_semantics: null,
    };
    const apiSaved = {
      algorithm: "stage_router",
      models: { capable: "strong", efficient: "fast", judge: "decider" },
      options: unsetOptions,
      classifier: { kind: "decision", cutoff: 0.6, instructions, evidence: null },
    };
    const decision = switchyardConfigSchema.parse(apiSaved);
    expect(switchyardSaveError("route", decision)).toBeNull();
    expect(switchyardRouterPayload("route", decision).litellm_params.switchyard_router_config.classifier).toMatchObject(
      {
        instructions,
      },
    );
    const plain = switchyardConfigSchema.parse({
      models: { capable: "strong", efficient: "fast", judge: null },
      classifier: null,
    });
    expect(switchyardSaveError("route", plain)).toBeNull();
  });
  it("accepts tool semantics that list only some categories", () => {
    expect(parseToolSemantics('{"observe": ["Grep"], "mutate": ["Deploy"]}')).toEqual({
      observe: ["Grep"],
      mutate: ["Deploy"],
    });
    expect(() => parseToolSemantics('{"unknown": ["Grep"]}')).toThrow();
  });
  it("switches classifier kind with fresh settings and clears a judge of the wrong kind", () => {
    const llm = {
      ...toggleClassifier(emptySwitchyardConfig(), true),
      models: { capable: "a", efficient: "b", judge: "gpt" },
    };
    const decision = setClassifierKind(llm, "decision");
    expect(decision.classifier).toEqual({ kind: "decision", cutoff: 0.6 });
    expect(decision.models.judge).toBe("");
    expect(setClassifierKind(decision, "decision")).toBe(decision);
    const saved = switchyardConfigSchema.parse({ ...decision, models: { ...decision.models, judge: "jev" } });
    expect(saved.classifier).toEqual({ kind: "decision", cutoff: 0.6 });
  });
  it("rejects LLM-only settings on a decision classifier and out-of-range cutoffs", () => {
    const models = { capable: "a", efficient: "b", judge: "jev" };
    expect(
      switchyardConfigSchema.safeParse({ models, classifier: { kind: "decision", base_threshold: 0.5 } }).success,
    ).toBe(false);
    expect(switchyardConfigSchema.safeParse({ models, classifier: { kind: "decision", cutoff: 1.2 } }).success).toBe(
      false,
    );
  });
  it.each([
    ["evaluation", true],
    ["decisions", true],
    ["chat", false],
    [undefined, false],
  ])("treats mode %s as a decision model: %s", (mode, expected) => {
    expect(isDecisionModelMode(mode)).toBe(expected);
  });
  it("blocks saving without a name or with the router as one of its own targets", () => {
    const config = { ...emptySwitchyardConfig(), models: { capable: "strong", efficient: "fast" } };
    expect(switchyardSaveError("route", config)).toBeNull();
    expect(switchyardSaveError("  ", config)).toBe("Enter a router name");
    expect(switchyardSaveError("strong", config)).toBe("The router cannot select itself");
  });
});
