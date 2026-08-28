import { describe, expect, it } from "vitest";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { applyTierSetAction, setFallbackTier } from "./tier_set_actions";

const tiers = {
  SIMPLE: ["gpt-3.5-turbo"],
  MEDIUM: ["gpt-3.5-turbo"],
  COMPLEX: ["gpt-4"],
  REASONING: ["claude-3-opus"],
};
const builtIn: ComplexityRouterConfigValue = { tiers, classifier_type: "llm" };
const custom: ComplexityRouterConfigValue = {
  ...builtIn,
  custom_tier_set: {
    tiers: [
      { id: "CASUAL", name: "CASUAL", definition: "small talk", models: ["gpt-3.5-turbo"] },
      { id: "sec", name: "SECURITY_REVIEW", definition: "audits", models: ["gpt-4"] },
    ],
    fallback_tier_id: "CASUAL",
  },
};
const apply = (value: ComplexityRouterConfigValue, action: Parameters<typeof applyTierSetAction>[2], rules = []) =>
  applyTierSetAction(value, rules, action);

describe("applyTierSetAction", () => {
  it("adds a row and moves the form into an edited set, which the built-in record never leaves", () => {
    const { value } = apply(builtIn, { kind: "add" });
    expect(value.custom_tier_set?.tiers).toHaveLength(5);
    expect(value.tiers).toEqual(tiers);
  });

  it("renames a built-in tier, which is what makes the set custom", () => {
    const { value } = apply(builtIn, { kind: "patch", id: "COMPLEX", patch: { name: "SECURITY_REVIEW" } });
    expect(value.custom_tier_set?.tiers.map((row) => row.name)).toEqual([
      "SIMPLE",
      "MEDIUM",
      "SECURITY_REVIEW",
      "REASONING",
    ]);
    expect(value.tiers).toEqual(tiers);
  });

  it("carries a renamed tier's keyword rules across, so a rule cannot be orphaned by a rename", () => {
    const rules = [{ id: "r1", keywords: ["audit"], tier: "COMPLEX" }];
    const next = applyTierSetAction(builtIn, rules, { kind: "patch", id: "COMPLEX", patch: { name: "AUDIT" } });
    expect(next.keywordTierRules).toEqual([{ id: "r1", keywords: ["audit"], tier: "AUDIT" }]);
  });

  it("leaves the rules alone when another row still answers to the old name", () => {
    const shared: ComplexityRouterConfigValue = {
      ...builtIn,
      custom_tier_set: {
        tiers: [
          { id: "a", name: "AUDIT", definition: "d", models: ["gpt-4"] },
          { id: "b", name: "audit", definition: "d", models: ["gpt-4"] },
        ],
        fallback_tier_id: "a",
      },
    };
    const rules = [{ id: "r1", keywords: ["x"], tier: "AUDIT" }];
    const next = applyTierSetAction(shared, rules, { kind: "patch", id: "a", patch: { name: "RENAMED" } });
    expect(next.keywordTierRules).toBe(rules);
  });

  it("prunes the per-model params of a model dropped from a row", () => {
    const withParams: ComplexityRouterConfigValue = {
      ...custom,
      tier_model_params: { sec: { "gpt-4": { reasoning_effort: "high" } } },
    };
    const { value } = apply(withParams, { kind: "models", id: "sec", models: [] });
    expect(value.tier_model_params?.sec ?? {}).toEqual({});
  });

  it("snapshots a removed built-in row's models back into the record it came from", () => {
    const { value } = apply(builtIn, { kind: "remove", id: "COMPLEX" });
    expect(value.custom_tier_set?.tiers.map((row) => row.id)).toEqual(["SIMPLE", "MEDIUM", "REASONING"]);
    expect(value.tiers.COMPLEX).toEqual(["gpt-4"]);
  });

  it("re-points a fallback whose row was removed rather than leaving it dangling", () => {
    const { value } = apply(custom, { kind: "remove", id: "CASUAL" });
    expect(value.custom_tier_set?.fallback_tier_id).toBe("sec");
  });

  it("turns the plan-mode floor off when its row is gone, in the same write", () => {
    const withFloor: ComplexityRouterConfigValue = { ...custom, plan_mode_min_tier: "sec" };
    const { value } = apply(withFloor, { kind: "remove", id: "sec" });
    expect(value.plan_mode_min_tier).toBeUndefined();
  });

  it("leaves no custom row's effort settings behind when restoring the built-in tiers", () => {
    const withEfforts: ComplexityRouterConfigValue = {
      ...custom,
      tier_model_params: { sec: { "gpt-4": { reasoning_effort: "high" } }, SIMPLE: { "gpt-3.5-turbo": {} } },
    };
    const { value } = apply(withEfforts, { kind: "restore" });
    expect(value.custom_tier_set).toBeUndefined();
    expect(Object.keys(value.tier_model_params ?? {})).toEqual(["SIMPLE"]);
  });

  it("keeps a renamed built-in row's effort settings across a restore", () => {
    const renamed: ComplexityRouterConfigValue = {
      ...custom,
      tier_model_params: { COMPLEX: { "gpt-4": { reasoning_effort: "high" } } },
      custom_tier_set: {
        tiers: [
          { id: "SIMPLE", name: "SIMPLE", definition: "", models: ["gpt-3.5-turbo"] },
          { id: "COMPLEX", name: "DEEP_WORK", definition: "renamed built-in", models: ["gpt-4"] },
        ],
        fallback_tier_id: "SIMPLE",
      },
    };
    const { value } = apply(renamed, { kind: "restore" });
    expect(value.tiers.COMPLEX).toEqual(["gpt-4"]);
    expect(value.tier_model_params).toEqual({ COMPLEX: { "gpt-4": { reasoning_effort: "high" } } });
  });

  it("resets a full set to the four built-ins rather than stacking them on top", () => {
    const six: ComplexityRouterConfigValue = {
      ...custom,
      custom_tier_set: {
        tiers: Array.from({ length: 6 }, (_, i) => ({
          id: `row-${i}`,
          name: `TIER_${i}`,
          definition: "d",
          models: ["gpt-4"],
        })),
        fallback_tier_id: "row-0",
      },
    };
    const { value } = apply(six, { kind: "restore" });
    expect(value.custom_tier_set).toBeUndefined();
    expect(Object.keys(value.tiers)).toEqual(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]);
  });
});

describe("setFallbackTier", () => {
  it("re-points the fallback without disturbing the rows", () => {
    const next = setFallbackTier(custom, "sec");
    expect(next.custom_tier_set?.fallback_tier_id).toBe("sec");
    expect(next.custom_tier_set?.tiers).toHaveLength(2);
  });
});
