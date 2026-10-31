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

  it("keeps restored defaults editable after create, save and model-info reload", () => {
    const initial = {
      algorithm: "escalation",
      models,
      escalation: { prompt: "override", deescalation: { strong_min_calls: 2, confirmations: 1 } },
    };
    const created = switchyardConfigSchema.parse(initial);
    const saved = switchyardConfigSchema.parse(restoreSwitchyardDefaults(hydrateSwitchyardConfig(created)));
    expect(saved.escalation).not.toHaveProperty("prompt");
    expect(saved.escalation).not.toHaveProperty("deescalation");
    const reloaded = hydrateSwitchyardConfig(JSON.stringify(saved));
    expect(switchyardConfigError(reloaded)).toBeNull();
    const edited = { ...reloaded, escalation: { ...(saved.escalation as Record<string, unknown>), confirmations: 4 } };
    expect(switchyardConfigSchema.parse(edited)).toMatchObject({ escalation: { confirmations: 4 } });
  });

  it("normalizes legacy model-info nulls only in schema-known non-string nullable fields", () => {
    const legacy = {
      algorithm: "escalation",
      models,
      weights: "None",
      seed: "None",
      custom: "None",
      capability: { recent_turn_window: "None", prompt: "None" },
      escalation: { deescalation: "None", prompt: "None" },
      stage: { recent_window: "None", classifier: "None", tool_semantics: "None", escalation_note: "None" },
      future: { nullable: "None" },
    };
    const hydrated = hydrateSwitchyardConfig(JSON.stringify(legacy));
    expect(switchyardConfigError(hydrated)).toBeNull();
    const expected = {
      algorithm: "escalation",
      models,
      capability: { prompt: "None" },
      escalation: { prompt: "None" },
      stage: { escalation_note: "None" },
      future: { nullable: "None" },
    };
    expect(hydrated).toEqual(expected);
  });

  it("preserves schema data and literal text while normalizing nested nullable options", () => {
    const legacy = {
      algorithm: "custom",
      models,
      custom: {
        prompt: "None",
        selector: "target",
        default_target: "efficient",
        recent_turn_window: "None",
        response_schema: { description: "None", default: null },
      },
      stage: { classifier: { recent_turn_window: "None", prompt: "None" } },
      escalation: { deescalation: { strong_min_calls: 2, confirmations: 1, strong_max_calls: "None" } },
      future: null,
    };
    const payload = switchyardConfigSchema.parse(hydrateSwitchyardConfig(legacy));
    const expected = {
      custom: { prompt: "None", response_schema: { description: "None", default: null } },
      stage: { classifier: { prompt: "None" } },
      escalation: { deescalation: { strong_min_calls: 2, confirmations: 1 } },
      future: null,
    };
    expect(payload).toMatchObject(expected);
    expect(payload.custom).not.toHaveProperty("recent_turn_window");
    expect((payload.escalation as Record<string, unknown>).deescalation).not.toHaveProperty("strong_max_calls");
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
