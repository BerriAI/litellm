import { describe, it, expect } from "vitest";
import {
  getAllPresets,
  getPresetByKey,
  getRequiredModelsInPreset,
  getMissingModelsInPreset,
} from "./autorouter_presets";

describe("autorouter_presets", () => {
  it("loads exactly the two model-family presets (sample_spec excluded)", () => {
    const presets = getAllPresets();
    expect(presets.map((p) => p.label).sort()).toEqual(["Anthropic Family", "OpenAI Family"]);
    // Every preset carries all four fields the UI relies on; a JSON typo dropping one fails here.
    for (const p of presets) {
      expect(p).toMatchObject({ key: expect.any(String), label: expect.any(String), description: expect.any(String) });
      expect(p.complexity_router_config.tiers).toBeTruthy();
    }
  });

  it("resolves a preset by its stable JSON key, not its display label", () => {
    expect(getPresetByKey("anthropic_family")?.label).toBe("Anthropic Family");
    expect(getPresetByKey("does_not_exist")).toBeUndefined();
    // sample_spec is filtered out, so it is not resolvable by key either.
    expect(getPresetByKey("sample_spec")).toBeUndefined();
  });

  it("keeps every preset a plain heuristic complexity router (no adaptive/quality settings)", () => {
    for (const { complexity_router_config: config } of getAllPresets()) {
      expect(config.classifier_type).toBe("heuristic");
      expect(config.adaptive).toBeUndefined();
      expect(config.adaptive_weights).toBeUndefined();
      expect(config.adaptive_eligible).toBeUndefined();
      expect(config.tier_distance_penalty).toBeUndefined();
    }
  });

  it("collects every tier model as a required model", () => {
    const preset = getPresetByKey("anthropic_family")!;
    const required = getRequiredModelsInPreset(preset);
    const tierModels = Object.values(preset.complexity_router_config.tiers).flat();
    expect(tierModels.length).toBeGreaterThan(0);
    for (const model of tierModels) expect(required.has(model)).toBe(true);
  });

  it("reports only the models the caller is missing, and none when the family is fully available", () => {
    const preset = getPresetByKey("openai_family")!;
    const required = [...getRequiredModelsInPreset(preset)];

    expect(getMissingModelsInPreset(preset, new Set(["gpt-5-nano"]))).toEqual(
      required.filter((m) => m !== "gpt-5-nano").sort(),
    );
    expect(getMissingModelsInPreset(preset, new Set(required))).toEqual([]);
  });
});
