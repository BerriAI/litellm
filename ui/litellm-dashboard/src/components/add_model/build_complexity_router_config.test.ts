import {
  buildComplexityRouterConfig,
  getMissingTiersError,
  getSemanticConfigError,
  BuildComplexityRouterConfigParams,
} from "./build_complexity_router_config";

const tiers = {
  SIMPLE: ["gpt-4o-mini"],
  MEDIUM: ["gpt-4o"],
  COMPLEX: ["claude-sonnet-4"],
  REASONING: ["o1-preview"],
};

const baseParams: BuildComplexityRouterConfigParams = {
  tiers,
  classifierType: "heuristic",
  classifierLlmConfig: undefined,
  customTechnicalKeywords: [],
  keywordTierRules: [],
  semanticMatchingEnabled: false,
  embeddingModel: undefined,
  matchThreshold: 0.5,
  escalationKeywords: ["LITELLM ESCALATE"],
  adaptive: false,
  adaptiveWeights: { quality: 0.3, cost: 0.7 },
  tierDistancePenalty: 0.5,
  adaptiveEligible: "all",
  returnRawModelName: false,
};

describe("buildComplexityRouterConfig", () => {
  it("emits tiers, classifier_type, and escalation_keywords when nothing else is configured", () => {
    const config = buildComplexityRouterConfig(baseParams);
    expect(config).toEqual({ tiers, classifier_type: "heuristic", escalation_keywords: ["LITELLM ESCALATE"] });
  });

  it("trims escalation keywords and drops blank entries", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      escalationKeywords: [" LITELLM ESCALATE ", "", "  ", "MAKE IT BETTER"],
    });
    expect(config.escalation_keywords).toEqual(["LITELLM ESCALATE", "MAKE IT BETTER"]);
  });

  it("emits an empty escalation_keywords list so clearing the field disables escalation", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, escalationKeywords: [] });
    expect(config.escalation_keywords).toEqual([]);
  });

  it("passes through a tier configured with more than one model as a pool", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      tiers: { ...tiers, SIMPLE: ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5"] },
    });
    expect(config.tiers.SIMPLE).toEqual(["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5"]);
  });

  it("includes classifier_llm_config only when classifier_type is llm", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "llm",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_type).toBe("llm");
    expect(config.classifier_llm_config).toEqual({ model: "gpt-4o-mini", timeout_ms: 3000 });
  });

  it("omits classifier_llm_config when classifier_type is heuristic even if config lingers in state", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "heuristic",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_llm_config).toBeUndefined();
  });

  it("sends keyword_tier_rules with their per-tier targeting preserved (not flattened)", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      customTechnicalKeywords: ["udp"],
      keywordTierRules: [
        { id: "r1", keywords: ["deploy to k8s"], tier: "REASONING" },
        { id: "r2", keywords: ["invoice", "refund"], tier: "SIMPLE" },
      ],
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.keyword_tier_rules).toEqual([
      { keywords: ["deploy to k8s"], tier: "REASONING" },
      { keywords: ["invoice", "refund"], tier: "SIMPLE" },
    ]);
    // custom technical keywords stay their own list, not merged with rule keywords
    expect(config.custom_technical_keywords).toEqual(["udp"]);
    expect(config.semantic_keyword_matching).toBeUndefined();
  });

  it("includes semantic fields only when semantic matching is enabled", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["k8s"], tier: "REASONING" }],
      semanticMatchingEnabled: true,
      embeddingModel: "openai/text-embedding-3-small",
      matchThreshold: 0.42,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.semantic_keyword_matching).toBe(true);
    expect(config.embedding_model).toBe("openai/text-embedding-3-small");
    expect(config.match_threshold).toBe(0.42);
  });

  it("omits semantic fields when the toggle is off even if an embedding model lingers in state", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["k8s"], tier: "REASONING" }],
      semanticMatchingEnabled: false,
      embeddingModel: "openai/text-embedding-3-small",
      matchThreshold: 0.42,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.semantic_keyword_matching).toBeUndefined();
    expect(config.embedding_model).toBeUndefined();
    expect(config.match_threshold).toBeUndefined();
  });

  it("omits empty optional lists", () => {
    const config = buildComplexityRouterConfig(baseParams);
    expect(config.custom_technical_keywords).toBeUndefined();
    expect(config.keyword_tier_rules).toBeUndefined();
  });

  it("trims keywords and drops rules left empty, so unfilled rows never 400 the backend", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [
        { id: "r1", keywords: [" deploy to k8s ", "", "  "], tier: "REASONING" },
        { id: "r2", keywords: [], tier: "COMPLEX" }, // seeded by "Add keyword rule", never filled
        { id: "r3", keywords: ["   "], tier: "SIMPLE" }, // whitespace only
      ],
    };
    const config = buildComplexityRouterConfig(params);
    // r1 keeps only its real keyword (trimmed); r2 and r3 are dropped entirely.
    expect(config.keyword_tier_rules).toEqual([{ keywords: ["deploy to k8s"], tier: "REASONING" }]);
  });

  it("omits keyword_tier_rules entirely when every rule is empty", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["", "  "], tier: "COMPLEX" }],
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.keyword_tier_rules).toBeUndefined();
  });

  it("omits adaptive fields when adaptive is disabled even if weights linger in state", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      adaptive: false,
      adaptiveWeights: { quality: 0.9, cost: 0.1 },
      tierDistancePenalty: 2,
      adaptiveEligible: "classified_tier",
    });
    expect(config.adaptive).toBeUndefined();
    expect(config.adaptive_weights).toBeUndefined();
    expect(config.tier_distance_penalty).toBeUndefined();
    expect(config.adaptive_eligible).toBeUndefined();
  });

  it("omits return_raw_model_name when disabled", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, returnRawModelName: false });
    expect(config.return_raw_model_name).toBeUndefined();
  });

  it("includes return_raw_model_name when enabled", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, returnRawModelName: true });
    expect(config.return_raw_model_name).toBe(true);
  });

  it("includes tier_distance_penalty when adaptive is enabled with eligible='all'", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      adaptive: true,
      adaptiveWeights: { quality: 0.6, cost: 0.4 },
      tierDistancePenalty: 0.75,
      adaptiveEligible: "all",
    });
    expect(config.adaptive).toBe(true);
    expect(config.adaptive_weights).toEqual({ quality: 0.6, cost: 0.4 });
    expect(config.tier_distance_penalty).toBe(0.75);
    expect(config.adaptive_eligible).toBe("all");
  });

  it("omits tier_distance_penalty when eligible='classified_tier', since the penalty doesn't apply there", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      adaptive: true,
      adaptiveWeights: { quality: 0.6, cost: 0.4 },
      tierDistancePenalty: 0.75,
      adaptiveEligible: "classified_tier",
    });
    expect(config.adaptive).toBe(true);
    expect(config.adaptive_eligible).toBe("classified_tier");
    expect(config.tier_distance_penalty).toBeUndefined();
  });
});

describe("getMissingTiersError", () => {
  it("returns null when all four tiers have a model", () => {
    expect(getMissingTiersError(tiers)).toBeNull();
  });

  it("names the specific missing tier when only one is blank", () => {
    expect(getMissingTiersError({ ...tiers, REASONING: [] })).toBe(
      "Select a model for the following tier(s): REASONING",
    );
  });

  it("names multiple missing tiers in SIMPLE/MEDIUM/COMPLEX/REASONING order", () => {
    expect(getMissingTiersError({ ...tiers, SIMPLE: [], REASONING: [] })).toBe(
      "Select a model for the following tier(s): SIMPLE, REASONING",
    );
  });

  it("names all four tiers when none are filled", () => {
    const noTiers = { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] };
    expect(getMissingTiersError(noTiers)).toBe(
      "Select a model for the following tier(s): SIMPLE, MEDIUM, COMPLEX, REASONING",
    );
  });

  it("treats a tier with more than one model as filled", () => {
    expect(getMissingTiersError({ ...tiers, SIMPLE: ["gpt-4o-mini", "gpt-4o"] })).toBeNull();
  });
});

describe("getSemanticConfigError", () => {
  const rule = { id: "r1", keywords: ["k8s"], tier: "REASONING" as const };

  it("returns null when semantic matching is disabled (even with gaps)", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: false, embeddingModel: undefined, keywordTierRules: [] }),
    ).toBeNull();
  });

  it("errors when enabled without an embedding model", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: undefined, keywordTierRules: [rule] }),
    ).toMatch(/embedding model/i);
  });

  it("errors when enabled with an embedding model but no keyword tier rules", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: "voyage-3-5", keywordTierRules: [] }),
    ).toMatch(/keyword tier rule/i);
  });

  it("errors when a rule has no non-empty keywords", () => {
    const emptyRule = { id: "r2", keywords: ["", "  "], tier: "SIMPLE" as const };
    expect(
      getSemanticConfigError({
        semanticMatchingEnabled: true,
        embeddingModel: "voyage-3-5",
        keywordTierRules: [emptyRule],
      }),
    ).toMatch(/at least one keyword/i);
  });

  it("returns null when enabled with both an embedding model and rules", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: "voyage-3-5", keywordTierRules: [rule] }),
    ).toBeNull();
  });
});
