import { describe, expect, it } from "vitest";
import { modelThrottlingError, modelThrottlingUpdate, modelsWithinAccess } from "./modelThrottling";

describe("budget model throttling", () => {
  const config = { thresholds: [{ spend: 60, models: ["premium"] }] };

  it("keeps thresholds strictly inside the final budget", () => {
    expect(modelThrottlingError(config, 100)).toBeNull();
    expect(modelThrottlingError(config, 60)).toBe("Each threshold must be above $0 and below the final budget");
    expect(modelThrottlingError(config, null)).toBe("Set a final budget before enabling model throttling");
    expect(modelThrottlingError(null, null)).toBeNull();
  });

  it("requires a positive threshold and models before saving", () => {
    expect(modelThrottlingError({ thresholds: [{ spend: 0, models: [] }] }, 100)).toBe(
      "Each threshold must be above $0 and below the final budget",
    );
    expect(modelThrottlingError({ thresholds: [{ spend: 60, models: [] }] }, 100)).toBe(
      "Select models to pause at each threshold",
    );
  });

  it("prevents duplicate amounts and models across cumulative thresholds", () => {
    expect(modelThrottlingError({ thresholds: [...config.thresholds, { spend: 60, models: ["standard"] }] }, 100)).toBe(
      "Combine models at the same spend into one threshold",
    );
    expect(modelThrottlingError({ thresholds: [...config.thresholds, { spend: 85, models: ["premium"] }] }, 100)).toBe(
      "Select each model at only one threshold",
    );
  });

  it("offers concrete models within the entity's access", () => {
    const models = ["premium", "economy", "openai/a", "openai/b", "openai/*", "all-team-models"];
    expect(modelsWithinAccess(models, ["economy"])).toEqual(["economy"]);
    expect(modelsWithinAccess(models, ["openai/*"])).toEqual(["openai/a", "openai/b"]);
    expect(modelsWithinAccess(models, ["no-default-models"])).toEqual([]);
    expect(modelsWithinAccess(models, ["all-team-models"])).toEqual(["premium", "economy", "openai/a", "openai/b"]);
  });

  it("expands group grants without offering group names or models outside the parent scope", () => {
    const groups = { premium_access: ["premium", "standard"] };
    expect(modelsWithinAccess(["premium_access", "economy"], ["premium_access"], groups)).toEqual([
      "premium",
      "standard",
    ]);
    expect(modelsWithinAccess(["economy", "standard"], ["economy"], groups, ["premium", "standard"])).toEqual([
      "economy",
      "standard",
    ]);
    expect(modelsWithinAccess(["economy", "premium"], [], groups, ["premium"])).toEqual(["premium"]);
  });

  it("preserves unchanged rules and sends an explicit clear when disabled", () => {
    expect(modelThrottlingUpdate(config, structuredClone(config))).toEqual({});
    expect(modelThrottlingUpdate(null, undefined)).toEqual({});
    expect(modelThrottlingUpdate(null, config)).toEqual({ model_throttling: null });
    expect(modelThrottlingUpdate(config, null)).toEqual({ model_throttling: config });
  });
});
