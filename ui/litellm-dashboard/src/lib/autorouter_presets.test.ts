import { describe, it, expect } from "vitest";
import {
  getAllPresets,
  getPresetByKey,
  getRequiredModelsInPreset,
  getMissingModelsInPreset,
  getRequiredModels,
  getMissingModels,
  getReferencedModelsError,
  buildEmptyPrefill,
  buildPresetPrefill,
} from "./autorouter_presets";
import { DEFAULT_MATCH_THRESHOLD } from "@/components/add_model/SemanticKeywordMatching";
import { DEFAULT_ESCALATION_KEYWORDS } from "@/components/add_model/EscalationKeywords";

describe("autorouter_presets", () => {
  it("loads exactly the two model-family presets", () => {
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
    const [held] = required;

    expect(getMissingModelsInPreset(preset, new Set([held]))).toEqual(required.filter((m) => m !== held).sort());
    expect(getMissingModelsInPreset(preset, new Set(required))).toEqual([]);
  });

  // Admins spell version numbers with either "-" or "." (claude-sonnet-4-5 vs claude-sonnet-4.5);
  // a caller who only registered one form still satisfies a preset that names the other. The
  // caller's spellings are derived from the preset itself so that renaming a preset's models in
  // autorouter_presets.json can't quietly turn this into a no-op (the inequality below fails
  // instead, if no preset model carries a version number at all).
  it("treats a preset's model as available under either version-separator spelling", () => {
    const preset = getPresetByKey("anthropic_family")!;
    const required = [...getRequiredModelsInPreset(preset)];
    const dottedSpellings = required.map((model) => model.replace(/(\d)-(\d)/g, "$1.$2"));

    expect(dottedSpellings).not.toEqual(required);
    expect(getMissingModelsInPreset(preset, new Set(dottedSpellings))).toEqual([]);
  });

  // The two-arm mirror: a differently-punctuated preset model must not be reported missing.
  it("does not flag a differently-punctuated model as missing via getMissingModels directly", () => {
    const missing = getMissingModels(
      { tiers: { SIMPLE: ["claude-sonnet-4-5"], MEDIUM: [], COMPLEX: [], REASONING: [] } },
      new Set(["claude-sonnet-4.5"]),
    );
    expect(missing).toEqual([]);
  });

  // A classifier_llm_config placeholder is seeded with model: "" before a caller picks one; an
  // empty string is not a real model reference and must not be reported as an unavailable model.
  it("does not treat an empty-string classifier or embedding model as a required model", () => {
    const required = getRequiredModels({
      tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] },
      classifier_llm_config: { model: "", timeout_ms: 5000 },
      embedding_model: "",
    });
    expect(required).toEqual(new Set(["gpt-5-nano"]));
  });

  describe("getReferencedModelsError", () => {
    const tiers = { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] };
    const available = new Set(["gpt-5-nano"]);
    // Both fields are always populated with a model missing from `available`; only the
    // enabled/disabled toggles below decide whether that missing model gets reported.
    const params = {
      classifierLlmConfig: { model: "missing-classifier", timeout_ms: 5000 },
      embeddingModel: "missing-embed",
    };

    // Bugbot-found bug class from #35199's history: a classifier/embedding model left selected
    // from a prior toggle must not block submit once that toggle is off again, since
    // buildComplexityRouterConfig never emits the field in that state - only a model whose toggle
    // is on should ever be reported.
    it.each([
      ["both toggles off", "heuristic", false, null],
      ["classifier type llm, semantic matching off", "llm", false, "missing-classifier"],
      ["classifier type heuristic, semantic matching on", "heuristic", true, "missing-embed"],
      ["both toggles on", "llm", true, "missing-classifier, missing-embed"],
    ] as const)("%s", (_label, classifierType, semanticMatchingEnabled, missingModels) => {
      const config = { tiers, classifierType, semanticMatchingEnabled, ...params };
      const error = getReferencedModelsError(config, available);
      expect(error).toBe(missingModels ? `Model(s) no longer available: ${missingModels}` : null);
    });
  });

  describe("buildEmptyPrefill", () => {
    it("resets every field to its default, empty state", () => {
      const expected = {
        complexityRouterConfig: {
          tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
          classifier_type: "heuristic",
        },
        customTechnicalKeywords: [],
        keywordTierRules: [],
        semanticMatchingEnabled: false,
        embeddingModel: undefined,
        matchThreshold: DEFAULT_MATCH_THRESHOLD,
        escalationKeywords: DEFAULT_ESCALATION_KEYWORDS,
      };
      expect(buildEmptyPrefill()).toEqual(expected);
    });
  });

  describe("buildPresetPrefill", () => {
    it("prefills a real bundled preset's tiers into the config", () => {
      const preset = getPresetByKey("anthropic_family")!;
      const prefill = buildPresetPrefill(preset.complexity_router_config, getRequiredModelsInPreset(preset));
      expect(prefill.complexityRouterConfig.tiers).toEqual(preset.complexity_router_config.tiers);
    });

    // `??`, not `||`: match_threshold: 0 and an empty escalation_keywords array are deliberate,
    // falsy preset values. A prefill that used `||` would silently replace both with the default,
    // which is exactly the kind of bug this test would have caught before either bundled preset
    // happened to avoid the case.
    it("keeps a preset's falsy match_threshold and escalation_keywords instead of defaulting them", () => {
      const config = {
        tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic" as const,
        session_affinity: false,
        match_threshold: 0,
        escalation_keywords: [],
      };
      const prefill = buildPresetPrefill(config, new Set(["gpt-5-nano"]));
      expect(prefill.matchThreshold).toBe(0);
      expect(prefill.escalationKeywords).toEqual([]);
    });

    it("falls back to the defaults when a preset omits match_threshold and escalation_keywords", () => {
      const prefill = buildPresetPrefill(
        {
          tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] },
          classifier_type: "heuristic",
          session_affinity: false,
        },
        new Set(["gpt-5-nano"]),
      );
      expect(prefill.matchThreshold).toBe(DEFAULT_MATCH_THRESHOLD);
      expect(prefill.escalationKeywords).toEqual(DEFAULT_ESCALATION_KEYWORDS);
    });

    // The whole point of the separator normalization: a caller whose proxy only registered the
    // dotted form of a version number still gets that model written into the tier, not the
    // preset's own hyphenated spelling (which the caller never actually registered).
    it("rewrites a preset's model name to the caller's differently-punctuated registered spelling", () => {
      const config = {
        tiers: { SIMPLE: ["claude-sonnet-4-5"], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic" as const,
        session_affinity: false,
      };
      const prefill = buildPresetPrefill(config, new Set(["claude-sonnet-4.5"]));
      expect(prefill.complexityRouterConfig.tiers.SIMPLE).toEqual(["claude-sonnet-4.5"]);
    });
  });
});
