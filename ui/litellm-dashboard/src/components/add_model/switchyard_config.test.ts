import { describe, expect, it } from "vitest";
import {
  hydrateSwitchyardConfig,
  restoreSwitchyardDefaults,
  SWITCHYARD_DEFAULTS,
  switchyardConfigError,
  switchyardConfigSchema,
} from "./switchyard_config";

const models = { efficient: ["small"], capable: ["large", "backup"], judge: ["judge"] };

describe("Switchyard configuration", () => {
  it("preserves unknown configuration and parses JSON editing fields on save", () => {
    const stored = {
      algorithm: "escalation",
      models,
      future_option: "keep",
      escalation: { future_judge_option: 9, deescalation: { strong_min_calls: 3, confirmations: 2 } },
      custom: null,
    };
    const draft = hydrateSwitchyardConfig(JSON.stringify(stored));
    const edited = {
      ...draft,
      escalation: { ...stored.escalation, confirmations: 4, deescalation: '{"strong_min_calls":5,"confirmations":3}' },
    };
    expect(switchyardConfigSchema.parse(edited)).toEqual({
      ...stored,
      escalation: { ...stored.escalation, confirmations: 4, deescalation: { strong_min_calls: 5, confirmations: 3 } },
    });
  });

  it("restores tuning defaults while preserving models and future settings", () => {
    const customConfig = {
      algorithm: "capability",
      models,
      future: true,
      capability: { base_threshold: 0.9, prompt: "override", future_setting: 4 },
    };
    const restored = restoreSwitchyardDefaults(customConfig);
    expect(restored).toMatchObject({
      models,
      future: true,
      capability: { ...SWITCHYARD_DEFAULTS.capability, future_setting: 4 },
    });
  });

  it("validates random weights against candidates rather than group count", () => {
    expect(
      switchyardConfigError({ algorithm: "random", models: { any: ["small", "large"] }, weights: "[1, 2]" }),
    ).toBeNull();
    expect(
      switchyardConfigError({ algorithm: "random", models: { any: ["small", "large"] }, weights: "[1]" }),
    ).toContain("one weight per model");
    expect(switchyardConfigError({ algorithm: "random", models: { any: ["small"] }, weights: "[0]" })).toContain(
      "positive weight",
    );
  });

  it("requires judge for classifiers but permits stage without a classifier", () => {
    const solvers = { efficient: ["small"], capable: ["large"] };
    expect(switchyardConfigError({ algorithm: "capability", models: solvers })).toContain("judge");
    expect(switchyardConfigError({ algorithm: "stage", models: solvers })).toBeNull();
    expect(switchyardConfigError({ algorithm: "stage", models: solvers, stage: { classifier: "{}" } })).toContain(
      "judge",
    );
  });

  it("rejects invalid JSON, missing custom fields, and incompatible affinity", () => {
    expect(switchyardConfigError({ algorithm: "random", models: { any: ["small"] }, weights: "bad" })).toContain(
      "weights",
    );
    expect(switchyardConfigError({ algorithm: "custom", models })).toContain("response_schema");
    expect(
      switchyardConfigError({ algorithm: "capability", models, capability: { message_hash_fallback: true } }),
    ).toContain("requires session affinity");
    expect(switchyardConfigError({ algorithm: "capability", models, capability: { base_threshold: 2 } })).toContain(
      "base_threshold",
    );
  });

  it("omits random weights when switching to a classifier and keeps the draft intact", () => {
    const draft = { algorithm: "capability", models, weights: [1, 2] };
    expect(switchyardConfigSchema.parse(draft)).not.toHaveProperty("weights");
    expect(draft.weights).toEqual([1, 2]);
  });
});
