import { describe, expect, it } from "vitest";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import {
  fuseProfileFields,
  fuseSettingsSchema,
  getForecastConfigError,
  prepareForecastClassifier,
  selectFuseProfile,
  type FuseSettings,
} from "./forecast_classifier_config";
import { getKeywordTierRulesError } from "./build_complexity_router_config";
import { activeTierRows } from "./tier_rows";
import {
  buildUpdatedComplexityRouterConfig,
  hydrateComplexityRouterConfig,
} from "../edit_auto_router/edit_auto_router_modal";

const capability: ComplexityRouterConfigValue = {
  classifier_type: "capability",
  classifier_llm_config: { model: "judge", timeout_ms: 20000 },
  tiers: { SIMPLE: ["efficient"], MEDIUM: [], COMPLEX: [], REASONING: ["capable"] },
  capability_classifier_config: {
    efficient_tier: "SIMPLE",
    capable_tier: "REASONING",
    base_threshold: 0.7,
    threshold_step: 0.1,
  },
};
const fuse: ComplexityRouterConfigValue = {
  ...capability,
  classifier_type: "llm_v2",
  capability_classifier_config: undefined,
  adaptive: false,
  llm_v2_config: {
    efficient_profile: "A concise solver",
    capable_profile: "A solver with more reasoning budget",
    harness: "One attempt with shell and tests",
    max_quality_gap: 0.05,
  },
};

describe("Fuse profile presets", () => {
  const refs: FuseSettings = {
    efficient_profile_preset: "efficient-v1",
    capable_profile_preset: "capable-v1",
    harness_preset: "runtime-v1",
    max_quality_gap: 0.05,
    max_output_tokens: 1024,
    response_format: "json_object",
    calibration: {
      version: "fitted-pair",
      prompt_version: "llm-v2-1",
      efficient: { slope: 1.1, intercept: -0.1 },
      capable: { slope: 0.9, intercept: 0.2 },
    },
  };

  it.each(fuseProfileFields)("validates both sources of %s without needing catalog availability", (field) => {
    const presetField = `${field}_preset` as const;
    expect(fuseSettingsSchema.safeParse(refs).success).toBe(true);
    expect(fuseSettingsSchema.safeParse({ ...refs, [presetField]: undefined }).success).toBe(false);
    expect(fuseSettingsSchema.safeParse({ ...refs, [field]: null, [presetField]: null }).success).toBe(false);
    expect(fuseSettingsSchema.safeParse({ ...refs, [field]: " \n " }).success).toBe(false);
    expect(fuseSettingsSchema.safeParse({ ...refs, [field]: "a".repeat(4001) }).success).toBe(false);
    expect(fuseSettingsSchema.safeParse({ ...refs, [field]: "a".repeat(4000) }).success).toBe(true);
    expect(fuseSettingsSchema.safeParse({ ...refs, [field]: "Override", [presetField]: "" }).success).toBe(false);
  });

  it.each([
    { ...fuse.llm_v2_config!, efficient_profile: "  Custom solver\n" },
    refs,
    { ...refs, efficient_profile: null, capable_profile: null, harness: null },
    { ...fuse.llm_v2_config!, efficient_profile_preset: null, capable_profile_preset: null, harness_preset: null },
    { ...refs, efficient_profile: "  Explicit override\n", capable_profile: "More budget", harness: "No shell" },
  ])("preserves references, literal overrides and calibration across hydration and unchanged saves: %j", (settings) => {
    const stored = { ...fuse, llm_v2_config: settings };
    const hydrated = hydrateComplexityRouterConfig(stored, undefined);
    expect(hydrated.llm_v2_config).toEqual(settings);
    expect(getForecastConfigError(hydrated)).toBeNull();
    const saved = buildUpdatedComplexityRouterConfig(stored, {
      ...hydrated,
      tiers: { ...hydrated.tiers, SIMPLE: ["arbitrary-new-group"] },
    });
    expect(saved.llm_v2_config).toEqual(settings);
  });

  it.each(fuseProfileFields)("changes only %s ownership on explicit selection", (field) => {
    const overridden = { ...refs, [field]: "Override" };
    const selected = selectFuseProfile(overridden, field, "replacement-v2", "Override");
    expect(selected).toEqual({ ...refs, [field]: undefined, [`${field}_preset`]: "replacement-v2" });
    expect(JSON.parse(JSON.stringify(selected))).not.toHaveProperty(field);
    const custom = selectFuseProfile(selected, field, undefined, "Effective preset text");
    expect(custom).toEqual({ ...refs, [field]: "Effective preset text", [`${field}_preset`]: undefined });
    expect(JSON.parse(JSON.stringify(custom))).not.toHaveProperty(`${field}_preset`);
  });
});

describe("forecast classifier configuration", () => {
  it.each([
    { version: "eval", slope: 21, intercept: 0 },
    { version: "eval", slope: 1, intercept: -21 },
    { version: " eval ", slope: 1, intercept: 0 },
  ])("rejects capability calibration outside the server contract: %j", (calibration) => {
    const value = {
      ...capability,
      capability_classifier_config: { ...capability.capability_classifier_config!, calibration },
    };
    expect(getForecastConfigError(value)).toContain("calibration");
  });

  it.each([capability, fuse])("accepts a complete $classifier_type configuration", (value) => {
    expect(getForecastConfigError(value)).toBeNull();
  });

  it("requires both solvers without requiring unused middle tiers", () => {
    expect(getForecastConfigError({ ...capability, tiers: { ...capability.tiers, REASONING: [] } })).toContain("both");
    expect(getForecastConfigError(capability)).toBeNull();
  });

  it("rejects a stepped threshold that exceeds one", () => {
    expect(
      getForecastConfigError({
        ...capability,
        capability_classifier_config: { ...capability.capability_classifier_config!, threshold_step: 0.2 },
      }),
    ).toContain("twice");
  });

  it.each([Number.NaN, -0.1, 1.1])("rejects an invalid probability %s", (base_threshold) => {
    expect(
      getForecastConfigError({
        ...capability,
        capability_classifier_config: { ...capability.capability_classifier_config!, base_threshold },
      }),
    ).not.toBeNull();
  });

  it.each(["efficient_profile", "capable_profile", "harness"] as const)("requires %s for Fuse", (field) => {
    expect(getForecastConfigError({ ...fuse, llm_v2_config: { ...fuse.llm_v2_config!, [field]: " " } })).not.toBeNull();
  });

  it("requires distinct single model groups and disables adaptive selection", () => {
    expect(getForecastConfigError({ ...fuse, tiers: { ...fuse.tiers, SIMPLE: ["capable"] } })).toContain("distinct");
    expect(getForecastConfigError({ ...fuse, tiers: { ...fuse.tiers, SIMPLE: ["efficient", "second"] } })).toContain(
      "distinct",
    );
    expect(getForecastConfigError({ ...fuse, adaptive: true })).toContain("adaptive");
  });

  it("removes incompatible prompt and tier settings when switching to Fuse", () => {
    const previous: ComplexityRouterConfigValue = {
      ...fuse,
      adaptive: true,
      classifier_fallback: "default_model",
      classifier_llm_config: {
        model: "judge",
        timeout_ms: 20000,
        system_prompt: "old rubric",
        classification_rubric: "business",
      },
      classification_prompt: "old prompt",
      classification_examples: "old example",
      tiers: { ...fuse.tiers, MEDIUM: ["extra"], SIMPLE: ["efficient", "second"] },
    };
    const next = prepareForecastClassifier(previous);
    const saved = buildUpdatedComplexityRouterConfig({}, next);
    expect(getForecastConfigError(next)).toBeNull();
    expect(saved.adaptive).toBe(false);
    expect(saved.tiers).toEqual({ SIMPLE: ["efficient"], REASONING: ["capable"] });
    expect(saved.classifier_llm_config).toEqual({ model: "judge", timeout_ms: 20000 });
    expect(saved).not.toHaveProperty("classification_prompt");
    expect(saved).not.toHaveProperty("classification_examples");
    expect(saved).not.toHaveProperty("classifier_fallback");
  });

  it.each([capability, fuse])(
    "switching to $classifier_type removes hidden pools and their overrides while retaining the solver settings",
    (value) => {
      const efficientParams = { reasoning_effort: "low", speed: "fast", max_tokens: 1024 };
      const secondEfficientParams = { reasoning_effort: "medium", max_tokens: 2048 };
      const capableParams = { reasoning_effort: "high", max_tokens: 4096 };
      const secondCapableParams = { speed: "fast" };
      const previous: ComplexityRouterConfigValue = {
        ...value,
        adaptive: true,
        plan_mode_min_tier: "MEDIUM",
        enable_non_reasoning_tier: true,
        tiers: {
          NON_REASONING: ["relay"],
          SIMPLE: ["efficient", "second-efficient"],
          MEDIUM: ["leftover-medium"],
          COMPLEX: ["leftover-complex"],
          REASONING: ["capable", "second-capable"],
        },
        tier_model_params: {
          NON_REASONING: { relay: { max_tokens: 128 } },
          SIMPLE: { efficient: efficientParams, "second-efficient": secondEfficientParams },
          MEDIUM: { "leftover-medium": { speed: "fast" } },
          COMPLEX: { "leftover-complex": { reasoning_effort: "high" } },
          REASONING: { capable: capableParams, "second-capable": secondCapableParams },
          LEGACY_CUSTOM: { "leftover-custom": { max_tokens: 512 } },
        },
      };
      const next = prepareForecastClassifier(previous);
      const preservesPools = value.classifier_type === "capability";
      const expectedTiers = {
        SIMPLE: preservesPools ? ["efficient", "second-efficient"] : ["efficient"],
        MEDIUM: [],
        COMPLEX: [],
        REASONING: preservesPools ? ["capable", "second-capable"] : ["capable"],
      };
      expect(next.tiers).toEqual(expectedTiers);
      expect(next.tier_model_params).toEqual({
        SIMPLE: {
          efficient: efficientParams,
          ...(preservesPools && { "second-efficient": secondEfficientParams }),
        },
        REASONING: {
          capable: capableParams,
          ...(preservesPools && { "second-capable": secondCapableParams }),
        },
      });
      expect(next.adaptive).toBe(preservesPools);
      expect(next.plan_mode_min_tier).toBeUndefined();
      expect(getForecastConfigError(next)).toBeNull();
      expect(activeTierRows(next).map((row) => row.name)).toEqual(["SIMPLE", "REASONING"]);
      expect(
        getKeywordTierRulesError([{ id: "old-middle", keywords: ["invoice"], tier: "MEDIUM" }], activeTierRows(next)),
      ).toContain("no longer has");
      expect(
        getKeywordTierRulesError([{ id: "solver", keywords: ["audit"], tier: "REASONING" }], activeTierRows(next)),
      ).toBeNull();
      const saved = buildUpdatedComplexityRouterConfig(previous, next);
      expect(saved.tiers).toEqual({ SIMPLE: expectedTiers.SIMPLE, REASONING: expectedTiers.REASONING });
      expect(saved.tier_model_configs).toEqual({
        SIMPLE: [
          { model_name: "efficient", litellm_params: efficientParams },
          ...(preservesPools ? [{ model_name: "second-efficient", litellm_params: secondEfficientParams }] : []),
        ],
        REASONING: [
          { model_name: "capable", litellm_params: capableParams },
          ...(preservesPools ? [{ model_name: "second-capable", litellm_params: secondCapableParams }] : []),
        ],
      });
      expect(saved).not.toHaveProperty("plan_mode_min_tier");
    },
  );

  it("preserves fitted calibration through edits and removes it when disabled", () => {
    const stored = {
      ...capability,
      capability_classifier_config: {
        ...capability.capability_classifier_config!,
        calibration: { version: "eval-a", slope: 1.2, intercept: -0.3 },
      },
    };
    const hydrated = hydrateComplexityRouterConfig(stored, undefined);
    const edited = {
      ...hydrated,
      capability_classifier_config: { ...hydrated.capability_classifier_config!, base_threshold: 0.6 },
    };
    expect(buildUpdatedComplexityRouterConfig(stored, edited).capability_classifier_config).toEqual({
      ...stored.capability_classifier_config,
      base_threshold: 0.6,
    });
    const disabled = {
      ...edited,
      capability_classifier_config: { ...edited.capability_classifier_config, calibration: undefined },
    };
    expect(buildUpdatedComplexityRouterConfig(stored, disabled).capability_classifier_config).toHaveProperty(
      "calibration",
      undefined,
    );
  });

  it("preserves non-default tier assignments", () => {
    const value = {
      ...capability,
      capability_classifier_config: {
        ...capability.capability_classifier_config!,
        efficient_tier: "MEDIUM",
        capable_tier: "COMPLEX",
      },
      tiers: { SIMPLE: [], MEDIUM: ["efficient"], COMPLEX: ["capable"], REASONING: [] },
    };
    expect(getForecastConfigError(value)).toBeNull();
    expect(buildUpdatedComplexityRouterConfig({}, value).capability_classifier_config).toEqual(
      value.capability_classifier_config,
    );
    const previous = {
      ...value,
      tiers: { ...value.tiers, SIMPLE: ["leftover-simple"], REASONING: ["leftover-reasoning"] },
      plan_mode_min_tier: "COMPLEX",
      tier_model_params: {
        MEDIUM: { efficient: { max_tokens: 1024 } },
        COMPLEX: { capable: { speed: "fast" } },
        SIMPLE: { "leftover-simple": { speed: "fast" } },
      },
    };
    const next = prepareForecastClassifier(previous);
    expect(next.tiers).toEqual(value.tiers);
    expect(next.plan_mode_min_tier).toBe("COMPLEX");
    expect(next.tier_model_params).toEqual({
      MEDIUM: { efficient: { max_tokens: 1024 } },
      COMPLEX: { capable: { speed: "fast" } },
    });
  });

  it("keeps configured extra Capability pools through hydration and an unrelated edit", () => {
    const stored = {
      ...capability,
      adaptive: true,
      plan_mode_min_tier: "MEDIUM",
      tiers: { ...capability.tiers, MEDIUM: ["middle"] },
      tier_model_configs: { MEDIUM: [{ model_name: "middle", litellm_params: { speed: "fast", max_tokens: 1024 } }] },
    };
    const hydrated = hydrateComplexityRouterConfig(stored, undefined);
    expect(hydrated.tiers.MEDIUM).toEqual(["middle"]);
    expect(hydrated.plan_mode_min_tier).toBe("MEDIUM");
    expect(activeTierRows(hydrated).map((row) => row.name)).toEqual(["SIMPLE", "MEDIUM", "REASONING"]);
    expect(
      getKeywordTierRulesError([{ id: "kept", keywords: ["invoice"], tier: "MEDIUM" }], activeTierRows(hydrated)),
    ).toBeNull();
    const saved = buildUpdatedComplexityRouterConfig(stored, {
      ...hydrated,
      capability_classifier_config: { ...hydrated.capability_classifier_config!, base_threshold: 0.6 },
    });
    expect(saved.tiers).toEqual({ SIMPLE: ["efficient"], MEDIUM: ["middle"], REASONING: ["capable"] });
    expect(saved.tier_model_configs).toEqual(stored.tier_model_configs);
    expect(saved.adaptive).toBe(false);
    expect(saved.plan_mode_min_tier).toBe("MEDIUM");
  });

  it("keeps empty built-in tiers available to standard classifiers", () => {
    const standard: ComplexityRouterConfigValue = { ...capability, classifier_type: "llm" };
    expect(activeTierRows(standard).map((row) => row.name)).toEqual(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]);
    expect(
      getKeywordTierRulesError([{ id: "middle", keywords: ["invoice"], tier: "MEDIUM" }], activeTierRows(standard)),
    ).toBeNull();
  });

  it.each([capability, fuse])("drops $classifier_type settings when switching to the heuristic", (value) => {
    const saved = buildUpdatedComplexityRouterConfig(value, { ...value, classifier_type: "heuristic" });
    expect(saved).not.toHaveProperty("capability_classifier_config");
    expect(saved).not.toHaveProperty("llm_v2_config");
  });
});
